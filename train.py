"""
Free-Energy Curriculum Selection via Fitted Soft Values.

Simplified formulation: the policy over training candidates is the Boltzmann
distribution pi(x) = softmax(w . phi(x) / beta) where w is a learned value
vector in embedding space, updated by online regression toward observed
one-step held-out improvements.

The algorithm alternates two steps (coordinate descent on the free energy):
  M-step: train on selected example   theta <- theta + alpha * grad log p(x_k)
  E-step: regress value vector         w <- w - eta * (w . phi(x_k) - r_k) * phi(x_k)

No Q-function, no V baseline, no Bellman loss, no parameter snapshots.

Uses GPT-2 (small) as the transformer and a local sentence-transformers model
as the embedding function phi.
"""

import os
import math
import json
import time
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Model
    model_name: str = "gpt2"
    context_length: int = 256  # tokens per context window

    # Embedding model
    embedding_backend: str = "local"  # "local" (sentence-transformers) or "openai"
    embedding_model_local: str = "all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"
    d_phi: int = 0  # set automatically based on backend

    # Data streaming
    num_candidates: int = 900  # |S_t|
    num_heldout: int = 400     # |D_t|
    outer_steps: int = 200     # number of outer-loop stream steps
    inner_steps: int = 200     # K inner-loop steps per outer step

    # Free-energy / soft-value parameters
    beta: float = 1.0          # Boltzmann temperature (higher = more exploration)
    value_lr: float = 0.01     # learning rate eta for value vector regression
    reward_scale: float = 1000.0  # multiply raw r_k; raw r_k ~ O(0.001)

    # LM training
    lr: float = 1e-4           # learning rate alpha for LM gradient step
    grad_clip: float = 1.0     # max gradient norm (0 = no clipping)

    # Curriculum mode
    curriculum: str = "rl"     # "rl" (free-energy) | "random" | "loss" | "uncertainty"

    # Eval
    num_eval_windows: int = 200  # fixed eval set size for perplexity tracking
    eval_every_outer: int = 5    # evaluate every N outer steps

    # Batching (0 = all-at-once; set lower if OOM)
    batch_size: int = 200        # max examples per GPU forward pass

    # Checkpointing
    save_every_outer: int = 0    # save model every N outer steps (0 = never)

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_every: int = 1         # log every N inner steps
    data_name: str = "openwebtext"
    data_split: str = "train"
    embedding_cache_path: str = "embedding_cache.json"
    checkpoint_dir: str = "checkpoints"


def config_from_args() -> Config:
    """Parse command-line arguments into a Config."""
    cfg = Config()
    parser = argparse.ArgumentParser(description="Free-Energy Curriculum Training")
    for k, v in cfg.__dict__.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true", default=v)
            parser.add_argument(f"--no_{k}", dest=k, action="store_false")
        else:
            ty = type(v) if v is not None else str
            parser.add_argument(f"--{k}", type=ty, default=v)
    args = parser.parse_args()
    return Config(**vars(args))


# ---------------------------------------------------------------------------
# Data streaming
# ---------------------------------------------------------------------------

class DataStream:
    """Streams context-window-sized chunks from a text dataset."""

    def __init__(self, cfg: Config, tokenizer: GPT2Tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.ctx = cfg.context_length

        logging.info("Loading dataset %s (streaming) ...", cfg.data_name)
        self.dataset = load_dataset(
            cfg.data_name, split=cfg.data_split, streaming=True,
        )
        self.iterator = iter(self.dataset)
        self.token_buffer: list[int] = []

    def _refill_buffer(self, need: int):
        """Tokenise more documents until we have at least *need* tokens."""
        while len(self.token_buffer) < need:
            try:
                doc = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                doc = next(self.iterator)
            tokens = self.tokenizer.encode(doc["text"])
            self.token_buffer.extend(tokens)

    def get_batch(self, n: int) -> list[list[int]]:
        """Return *n* context windows, each of length cfg.context_length tokens."""
        total = n * self.ctx
        self._refill_buffer(total)
        windows = []
        for _ in range(n):
            window = self.token_buffer[: self.ctx]
            self.token_buffer = self.token_buffer[self.ctx :]
            windows.append(window)
        return windows

    def get_stream_step(self) -> tuple[list[list[int]], list[list[int]]]:
        """Return (S_t, D_t) for one outer-loop step."""
        total = self.cfg.num_candidates + self.cfg.num_heldout
        windows = self.get_batch(total)
        S_t = windows[: self.cfg.num_candidates]
        D_t = windows[self.cfg.num_candidates :]
        return S_t, D_t


# ---------------------------------------------------------------------------
# Embedding: local (sentence-transformers) or OpenAI API (with disk cache)
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """Embeds text using a local sentence-transformers model."""

    def __init__(self, cfg: Config, tokenizer: GPT2Tokenizer):
        from sentence_transformers import SentenceTransformer
        self.tokenizer = tokenizer
        logging.info("Loading local embedding model: %s", cfg.embedding_model_local)
        self.model = SentenceTransformer(cfg.embedding_model_local)
        self.d_phi = self.model.get_sentence_embedding_dimension()
        logging.info("Local embedding dimension: %d", self.d_phi)

    def embed_batch(self, token_id_lists: list[list[int]]) -> torch.Tensor:
        texts = [self.tokenizer.decode(ids) for ids in token_id_lists]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.float()  # (N, d_phi)


class OpenAIEmbedder:
    """Embeds text using OpenAI API with a disk cache."""

    def __init__(self, cfg: Config, tokenizer: GPT2Tokenizer):
        import openai
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.client = openai.OpenAI()  # uses OPENAI_API_KEY env var
        self.cache: dict[str, list[float]] = {}
        self._load_cache()
        resp = self.client.embeddings.create(
            model=cfg.embedding_model_openai, input=["dimension probe"]
        )
        self.d_phi = len(resp.data[0].embedding)
        logging.info("OpenAI embedding dimension: %d", self.d_phi)

    def _load_cache(self):
        p = Path(self.cfg.embedding_cache_path)
        if p.exists():
            with open(p) as f:
                self.cache = json.load(f)
            logging.info("Loaded %d cached embeddings.", len(self.cache))

    def _save_cache(self):
        with open(self.cfg.embedding_cache_path, "w") as f:
            json.dump(self.cache, f)

    def embed_batch(self, token_id_lists: list[list[int]]) -> torch.Tensor:
        texts = [self.tokenizer.decode(ids) for ids in token_id_lists]
        keys = [t[:128] for t in texts]
        miss_indices = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            response = self.client.embeddings.create(
                model=self.cfg.embedding_model_openai, input=miss_texts,
            )
            for idx, emb_obj in zip(miss_indices, response.data):
                self.cache[keys[idx]] = emb_obj.embedding
            self._save_cache()
        embeddings = [self.cache[k] for k in keys]
        return torch.tensor(embeddings, dtype=torch.float32)


def make_embedder(cfg: Config, tokenizer: GPT2Tokenizer):
    """Factory: return the right embedder and set cfg.d_phi."""
    if cfg.embedding_backend == "local":
        emb = LocalEmbedder(cfg, tokenizer)
    elif cfg.embedding_backend == "openai":
        emb = OpenAIEmbedder(cfg, tokenizer)
    else:
        raise ValueError(f"Unknown embedding backend: {cfg.embedding_backend}")
    cfg.d_phi = emb.d_phi
    return emb


# ---------------------------------------------------------------------------
# Model: GPT-2 (language model only — no RL heads)
# ---------------------------------------------------------------------------

class CurriculumGPT2(nn.Module):
    """
    GPT-2 backbone for language modeling.  No policy or Q-value heads;
    the selection policy is parameterized by a separate value vector w
    in embedding space (not part of this module).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.gpt2 = GPT2LMHeadModel.from_pretrained(cfg.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward_lm(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Return mean_log_prob: scalar, mean log p(x) over tokens (teacher-forced).
        """
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Shift for causal LM: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask[:, 1:].contiguous().float()
            mean_lp = (token_log_probs * mask).sum() / mask.sum().clamp(min=1)
        else:
            mean_lp = token_log_probs.mean()

        return mean_lp


# ---------------------------------------------------------------------------
# Policy: Boltzmann distribution over fitted soft values
# ---------------------------------------------------------------------------

class SoftValuePolicy:
    """
    Policy pi(x) = softmax(w . phi(x) / beta) over candidates S_t.

    w is a value vector in embedding space (d_phi,), updated by online
    linear regression toward observed rewards:
        w <- w - eta * (w . phi(x_k) - r_k) * phi(x_k)

    This is the Boltzmann distribution over estimated values V(x) = w . phi(x).
    The entropy H(pi) is implicitly controlled by beta (temperature).
    """

    def __init__(self, d_phi: int, beta: float, lr: float, device: torch.device):
        self.beta = beta
        self.lr = lr
        self.device = device
        # Initialize w to zero: uniform policy at start (all logits equal)
        self.w = torch.zeros(d_phi, device=device)
        # Running stats for logging
        self.n_updates = 0
        self.reward_ema = 0.0  # exponential moving average of rewards

    def log_pi(self, E_S: torch.Tensor) -> torch.Tensor:
        """
        Compute log pi(x) for all x in S_t.
        E_S: (|S_t|, d_phi) — candidate embeddings.
        Returns: (|S_t|,) log-probabilities.
        """
        logits = E_S @ self.w / max(self.beta, 1e-8)  # (|S_t|,)
        return F.log_softmax(logits, dim=0)

    def sample(self, E_S: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Sample one candidate index from pi.
        Returns (idx, log_pi) where log_pi is the full (|S_t|,) vector.
        """
        lp = self.log_pi(E_S)
        probs = torch.exp(lp)
        idx = torch.multinomial(probs, num_samples=1).item()
        return idx, lp

    def update(self, phi_x: torch.Tensor, r_k: float):
        """
        Online linear regression step:
            w <- w - eta * (w . phi(x) - r_k) * phi(x)

        phi_x: (d_phi,) — embedding of selected candidate.
        r_k: scalar reward (scaled held-out improvement).
        """
        predicted = self.w @ phi_x  # scalar
        error = predicted - r_k
        self.w = self.w - self.lr * error * phi_x
        self.n_updates += 1
        # EMA of reward for baseline / logging
        alpha = min(1.0, 2.0 / (self.n_updates + 1))
        self.reward_ema = (1 - alpha) * self.reward_ema + alpha * r_k

    def stats(self, log_pi: torch.Tensor) -> dict:
        """Compute policy diagnostics."""
        pi = torch.exp(log_pi.detach())
        entropy = -(pi * log_pi.detach()).sum().item()
        max_prob = pi.max().item()
        min_prob = pi.min().item()
        effective_n = math.exp(entropy) if entropy > 0 else 1.0
        return {
            "pi_entropy": entropy,
            "pi_max_prob": max_prob,
            "pi_min_prob": min_prob,
            "pi_effective_n": effective_n,
            "w_norm": self.w.norm().item(),
            "w_mean": self.w.mean().item(),
            "reward_ema": self.reward_ema,
            "value_pred": 0.0,  # filled in by caller
        }


# ---------------------------------------------------------------------------
# Utility: batch-tensorize token lists
# ---------------------------------------------------------------------------

def _tokens_to_tensor(
    token_id_lists: list[list[int]], device: torch.device,
) -> torch.Tensor:
    """Stack token lists into a (B, T) tensor.  All lists must have the same length."""
    return torch.tensor(token_id_lists, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Utility: compute mean log-prob of a set of context windows (batched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_log_prob_dataset(
    model: CurriculumGPT2,
    token_ids,
    device: torch.device,
    batch_size: int = 0,
) -> float:
    """
    Compute (1/|D|) sum_x log p_theta(x) over a set of context windows.
    batch_size=0 means process all at once.
    """
    if isinstance(token_ids, torch.Tensor):
        all_ids = token_ids
    else:
        all_ids = _tokens_to_tensor(token_ids, device)
    N = all_ids.size(0)
    bs = N if batch_size <= 0 else batch_size
    total_lp = 0.0
    for start in range(0, N, bs):
        chunk = all_ids[start : start + bs]
        lp = model.forward_lm(chunk)
        total_lp += lp.item() * chunk.size(0)
    return total_lp / N


# ---------------------------------------------------------------------------
# Baseline curricula (non-RL)
# ---------------------------------------------------------------------------

def select_random(S_t_tokens: list[list[int]], **_kw) -> int:
    """Uniform random selection."""
    return torch.randint(len(S_t_tokens), (1,)).item()


def select_loss_based(
    model: CurriculumGPT2,
    S_t_tokens: list[list[int]],
    device: torch.device,
    **_kw,
) -> int:
    """Select the candidate with the highest LM loss (lowest log-prob)."""
    worst_lp = float("inf")
    worst_idx = 0
    with torch.no_grad():
        for i, ids in enumerate(S_t_tokens):
            input_ids = torch.tensor([ids], device=device)
            lp = model.forward_lm(input_ids)
            if lp.item() < worst_lp:
                worst_lp = lp.item()
                worst_idx = i
    return worst_idx


def select_uncertainty_based(
    model: CurriculumGPT2,
    S_t_tokens: list[list[int]],
    device: torch.device,
    **_kw,
) -> int:
    """Select the candidate with highest predictive entropy."""
    best_ent = -float("inf")
    best_idx = 0
    with torch.no_grad():
        for i, ids in enumerate(S_t_tokens):
            input_ids = torch.tensor([ids], device=device)
            outputs = model.gpt2(input_ids)
            logits = outputs.logits
            probs = F.softmax(logits[:, :-1, :], dim=-1)
            ent = -(probs * probs.clamp(min=1e-12).log()).sum(-1).mean().item()
            if ent > best_ent:
                best_ent = ent
                best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _grad_norm_lm(model: CurriculumGPT2) -> float:
    """Compute total gradient norm of the LM model."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return math.sqrt(total)


def _gpu_stats() -> dict:
    """GPU memory usage (if CUDA available)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        "gpu_mem_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }


def _sync():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Evaluation: perplexity on a fixed set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: CurriculumGPT2,
    token_ids,
    device: torch.device,
    batch_size: int = 0,
) -> float:
    """Compute perplexity = exp(-mean_log_prob) on a fixed eval set."""
    mlp = mean_log_prob_dataset(model, token_ids, device, batch_size=batch_size)
    return math.exp(-mlp)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    logging.info("Device: %s", device)
    logging.info("Curriculum mode: %s", cfg.curriculum)

    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Embedder (sets cfg.d_phi) ---
    embedder = make_embedder(cfg, tokenizer)
    logging.info("Embedding dimension (d_phi): %d", cfg.d_phi)

    # --- Model (pure LM, no RL heads) ---
    model = CurriculumGPT2(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- Soft-value policy (separate from model) ---
    policy = SoftValuePolicy(
        d_phi=cfg.d_phi, beta=cfg.beta, lr=cfg.value_lr, device=device,
    )
    logging.info(
        "Soft-value policy: beta=%.3f, value_lr=%.4f, d_phi=%d",
        cfg.beta, cfg.value_lr, cfg.d_phi,
    )

    # --- Data ---
    stream = DataStream(cfg, tokenizer)

    # --- Fixed eval set ---
    logging.info("Drawing %d fixed eval windows ...", cfg.num_eval_windows)
    eval_tokens_list = stream.get_batch(cfg.num_eval_windows)
    eval_ids = _tokens_to_tensor(eval_tokens_list, device)
    init_ppl = evaluate_perplexity(model, eval_ids, device, batch_size=cfg.batch_size)
    logging.info("Initial eval perplexity: %.2f", init_ppl)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Logging ---
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = Path(cfg.checkpoint_dir) / "log.jsonl"
    if log_path.exists():
        log_path.unlink()
        logging.info("Cleared old log file: %s", log_path)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    logging.info("Run ID: %s", run_id)
    with open(Path(cfg.checkpoint_dir) / "config.json", "w") as f:
        cfg_dict = cfg.__dict__.copy()
        cfg_dict["run_id"] = run_id
        cfg_dict["formulation"] = "free_energy_softvalue"
        json.dump(cfg_dict, f, indent=2)

    logging.info(
        "Starting training: %d outer steps x %d inner steps  (curriculum=%s)",
        cfg.outer_steps, cfg.inner_steps, cfg.curriculum,
    )

    total_examples_seen = 0
    cumulative_reward = 0.0
    train_start_time = time.time()

    for t in range(cfg.outer_steps):
        outer_t0 = time.time()

        # === Outer loop: refresh S_t, D_t ===
        t_data = time.time()
        S_t_tokens, D_t_tokens = stream.get_stream_step()
        D_t_ids = _tokens_to_tensor(D_t_tokens, device)
        dt_data = time.time() - t_data

        # Pre-compute embeddings for S_t (needed for rl mode)
        E_S = None
        dt_embed = 0.0
        if cfg.curriculum == "rl":
            t_emb = time.time()
            E_S = embedder.embed_batch(S_t_tokens).to(device)  # (|S_t|, d_phi)
            dt_embed = time.time() - t_emb

        # Cache held-out log-prob before inner loop
        t_held = time.time()
        prev_heldout_lp = mean_log_prob_dataset(model, D_t_ids, device, batch_size=cfg.batch_size)
        dt_heldout_init = time.time() - t_held

        for k in range(cfg.inner_steps):
            step_t0 = time.time()
            timings = {}

            # ==============================================================
            # Step 1 (E-step): Select a training example from pi
            # ==============================================================
            _sync(); t0 = time.time()
            log_pi_k = None
            if cfg.curriculum == "rl":
                idx_k, log_pi_k = policy.sample(E_S)
            elif cfg.curriculum == "random":
                idx_k = select_random(S_t_tokens)
            elif cfg.curriculum == "loss":
                idx_k = select_loss_based(model, S_t_tokens, device)
            elif cfg.curriculum == "uncertainty":
                idx_k = select_uncertainty_based(model, S_t_tokens, device)
            else:
                raise ValueError(f"Unknown curriculum: {cfg.curriculum}")
            _sync(); timings["t_select"] = round(time.time() - t0, 4)

            x_k_tokens = S_t_tokens[idx_k]
            x_k_text_snippet = tokenizer.decode(x_k_tokens[:60]).replace("\n", " ")[:200]
            total_examples_seen += 1

            # ==============================================================
            # Step 2 (M-step): Language-modeling update
            #   theta <- theta + alpha * grad log p(x_k)
            # ==============================================================
            _sync(); t0 = time.time()
            input_ids_xk = torch.tensor([x_k_tokens], device=device)
            lm_logp = model.forward_lm(input_ids_xk)
            lm_loss = -lm_logp

            optimizer.zero_grad()
            lm_loss.backward()
            lm_grad_total = _grad_norm_lm(model)
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            _sync(); timings["t_lm_update"] = round(time.time() - t0, 4)

            # ==============================================================
            # Step 3: Reward  r_k = mean_lp(D, theta') - mean_lp(D, theta)
            # ==============================================================
            _sync(); t0 = time.time()
            new_heldout_lp = mean_log_prob_dataset(model, D_t_ids, device, batch_size=cfg.batch_size)
            r_k_raw = new_heldout_lp - prev_heldout_lp
            cumulative_reward += r_k_raw
            r_k = r_k_raw * cfg.reward_scale
            _sync(); timings["t_reward"] = round(time.time() - t0, 4)

            # ==============================================================
            # Step 4: Value regression update on w
            #   w <- w - eta * (w . phi(x_k) - r_k) * phi(x_k)
            # ==============================================================
            value_stats = {}
            if cfg.curriculum == "rl":
                _sync(); t0 = time.time()
                phi_xk = E_S[idx_k]  # (d_phi,)
                value_pred = (policy.w @ phi_xk).item()
                policy.update(phi_xk, r_k)
                _sync(); timings["t_value_update"] = round(time.time() - t0, 4)

                # Policy diagnostics (recompute log_pi after w update)
                log_pi_after = policy.log_pi(E_S)
                pol_stats = policy.stats(log_pi_after)
                pol_stats["value_pred"] = value_pred
                pol_stats["value_error"] = value_pred - r_k
                value_stats = pol_stats

            # Update cached held-out log-prob
            prev_heldout_lp = new_heldout_lp

            # --- Logging ---
            step_time = time.time() - step_t0
            if k % cfg.log_every == 0:
                wall_clock = time.time() - train_start_time
                log_entry = {
                    "outer_step": t,
                    "inner_step": k,
                    "global_step": t * cfg.inner_steps + k,
                    "total_examples": total_examples_seen,
                    "wall_clock": round(wall_clock, 2),
                    "step_time": round(step_time, 4),
                    "curriculum": cfg.curriculum,
                    # -- Timings --
                    **timings,
                    # -- Selected example --
                    "selected_idx": idx_k,
                    "selected_text": x_k_text_snippet,
                    # -- Core metrics --
                    "r_k_raw": r_k_raw,
                    "r_k": r_k,
                    "cumulative_reward": cumulative_reward,
                    "heldout_lp": new_heldout_lp,
                    "lm_loss": lm_loss.item(),
                    # -- LM gradient norm --
                    "lm_grad_total": lm_grad_total,
                    # -- GPU --
                    **_gpu_stats(),
                    # -- Soft-value policy stats --
                    **value_stats,
                }
                logging.info(
                    "[t=%d k=%d] r=%.4f heldout=%.4f lm=%.4f grad=%.3f (%.2fs)"
                    "  w_norm=%.3f ent=%.2f eff_n=%.0f",
                    t, k, r_k, new_heldout_lp, lm_loss.item(),
                    lm_grad_total, step_time,
                    value_stats.get("w_norm", 0),
                    value_stats.get("pi_entropy", 0),
                    value_stats.get("pi_effective_n", 0),
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        outer_time = time.time() - outer_t0

        # --- End of outer step: eval + checkpoint ---
        if t % cfg.eval_every_outer == 0:
            t0 = time.time()
            ppl = evaluate_perplexity(model, eval_ids, device, batch_size=cfg.batch_size)
            eval_lp = mean_log_prob_dataset(model, eval_ids, device, batch_size=cfg.batch_size)
            dt_eval = time.time() - t0
            wall_clock = time.time() - train_start_time
            logging.info(
                "[Outer %d] ppl=%.2f eval_lp=%.4f (ex=%d, %.0fs) outer=%.1fs"
                " (data=%.2f emb=%.2f held=%.2f eval=%.2f)",
                t, ppl, eval_lp, total_examples_seen, wall_clock,
                outer_time, dt_data, dt_embed, dt_heldout_init, dt_eval,
            )
            eval_entry = {
                "type": "eval",
                "outer_step": t,
                "total_examples": total_examples_seen,
                "wall_clock": round(wall_clock, 2),
                "perplexity": ppl,
                "eval_lp": eval_lp,
                "cumulative_reward": cumulative_reward,
                "outer_time": round(outer_time, 3),
                "t_data_stream": round(dt_data, 4),
                "t_embed": round(dt_embed, 4),
                "t_heldout_init": round(dt_heldout_init, 4),
                "t_eval": round(dt_eval, 4),
                "w_norm": policy.w.norm().item() if cfg.curriculum == "rl" else 0.0,
                **_gpu_stats(),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(eval_entry) + "\n")

        if cfg.save_every_outer > 0 and (t + 1) % cfg.save_every_outer == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"model_outer_{t:04d}.pt"
            save_dict = {
                "model_state_dict": model.state_dict(),
                "w": policy.w.cpu(),
                "beta": policy.beta,
            }
            torch.save(save_dict, ckpt_path)
            logging.info("[Outer %d] Checkpoint saved to %s", t, ckpt_path)

    logging.info("Training complete.  Total examples seen: %d", total_examples_seen)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = config_from_args()
    train(cfg)
    # Workaround for HuggingFace tokenizer GIL cleanup crash on exit
    import gc
    gc.collect()
    os._exit(0)
