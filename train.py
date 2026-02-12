"""
Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models.

Implementation of the algorithm described in project.pdf / project.tex.
Uses GPT-2 (small) as the transformer and either OpenAI text-embedding-3-small
or a local sentence-transformers model as the embedding function phi.
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
    num_heldout: int = 100     # |D_t|
    outer_steps: int = 200     # number of outer-loop stream steps
    inner_steps: int = 100     # K inner-loop steps per outer step

    # RL
    gamma: float = 0.99        # discount factor
    lr: float = 1e-4           # shared learning rate alpha
    M: int = 16                # number of samples for V estimation

    # Stability
    grad_clip: float = 1.0     # max gradient norm (0 = no clipping)
    fix_sigma: bool = True     # if True, sigma^2 = 1 (W_gamma ignored)
    policy_temp: float = 0.0   # temperature for policy logits (0 = auto: 1/d_phi)

    # Curriculum mode
    curriculum: str = "rl"     # "rl" | "random" | "loss" | "uncertainty"

    # Eval
    num_eval_windows: int = 200  # fixed eval set size for perplexity tracking
    eval_every_outer: int = 1    # evaluate every N outer steps

    # Batching (0 = all-at-once; set lower if OOM)
    batch_size: int = 0          # max examples per GPU forward pass

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
    parser = argparse.ArgumentParser(description="Curriculum RL Training")
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
        # Probe dimension
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
                model=self.cfg.embedding_model_openai,
                input=miss_texts,
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
# Model: GPT-2 + policy / Q heads
# ---------------------------------------------------------------------------

class CurriculumGPT2(nn.Module):
    """
    GPT-2 backbone with three extra linear heads:
      - W_mu:    d -> d_phi   (policy mean)
      - W_gamma: d -> d_phi   (policy log-variance)
      - W_Q:     d -> 1       (Q-value)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Load pretrained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(cfg.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        d_model = self.gpt2.config.n_embd  # 768 for gpt2
        n_layers = self.gpt2.config.n_layer  # 12 for gpt2

        # Second-to-last layer: paper calls this ell = L-1 (1-indexed).
        # In 0-indexed terms this is layer (n_layers - 2), i.e. index 10 for GPT-2.
        self.hook_layer = n_layers - 2

        # Projection heads (d_phi must be set before calling this)
        assert cfg.d_phi > 0, "cfg.d_phi must be set before constructing CurriculumGPT2"
        self.W_mu = nn.Linear(d_model, cfg.d_phi, bias=False)
        self.W_gamma = nn.Linear(d_model, cfg.d_phi, bias=False)
        self.W_Q = nn.Linear(d_model, 1, bias=False)

        # Init heads small so they don't dominate early
        nn.init.normal_(self.W_mu.weight, std=0.01)
        nn.init.normal_(self.W_gamma.weight, std=0.01)
        nn.init.normal_(self.W_Q.weight, std=0.01)

    def _get_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run GPT-2 and return:
          - lm_logits: (B, T, vocab)
          - hidden:    (B, T, d) from the second-to-last layer
        """
        outputs = self.gpt2(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # outputs.hidden_states is a tuple of (n_layer+1) tensors
        # index 0 = embedding output, index i = output of layer i-1
        # second-to-last layer activation = index (n_layer - 1) = hook_layer + 1
        hidden = outputs.hidden_states[self.hook_layer + 1]  # (B, T, d)
        return outputs.logits, hidden

    def forward_lm(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (mean_log_prob, hidden_last_token).
        mean_log_prob: scalar, mean log p(x) over tokens (teacher-forced).
        hidden_last_token: (B, d) activation at last real token from layer L-1.
        """
        logits, hidden = self._get_hidden_states(input_ids, attention_mask)
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

        # Last-token hidden state (for Q-head)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            h_last = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        else:
            h_last = hidden[:, -1, :]  # (B, d)

        return mean_lp, h_last

    def get_policy_state(self, device: torch.device) -> torch.Tensor:
        """
        Forward pass on just <s> token to get z = f_theta^{(L-1)}(<s>).
        Returns z: (d,)
        """
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.eos_token_id
        input_ids = torch.tensor([[bos_id]], device=device)
        _, hidden = self._get_hidden_states(input_ids)
        return hidden[0, 0, :]  # (d,)

    def policy_params(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Given z (d,), return mu (d_phi,) and log_sigma2 (d_phi,)."""
        mu = self.W_mu(z)                # (d_phi,)
        log_sigma2 = self.W_gamma(z)     # (d_phi,)
        return mu, log_sigma2

    def q_value(self, h_last: torch.Tensor) -> torch.Tensor:
        """
        Given last-token hidden state(s), return scalar Q value(s).
        h_last: (d,) or (B, d) -> scalar or (B,)
        """
        return self.W_Q(h_last).squeeze(-1)


# ---------------------------------------------------------------------------
# Policy: Gaussian in embedding space normalised over S_t
# ---------------------------------------------------------------------------

def compute_log_pi(
    mu: torch.Tensor,
    log_sigma2: torch.Tensor,
    E_S: torch.Tensor,
    fix_sigma: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute log pi(x) for all x in S_t.

    mu:         (d_phi,)
    log_sigma2: (d_phi,)
    E_S:        (|S_t|, d_phi)
    fix_sigma:  if True, use sigma^2 = 1 (ignore log_sigma2)
    temperature: scale logits by this factor before softmax (higher = more uniform)

    Returns log_pi: (|S_t|,)  (log-normalised over S_t)

    The constant terms in the Gaussian log-density cancel in the softmax
    normalisation, so we only need the quadratic part as logits:
        logit(x) = -0.5 * sum_d (phi(x)_d - mu_d)^2 / sigma^2_d
    """
    diff = E_S - mu.unsqueeze(0)  # (|S_t|, d_phi)
    if fix_sigma:
        logits = -0.5 * (diff ** 2).sum(dim=-1)  # (|S_t|,)
    else:
        sigma2 = torch.exp(log_sigma2).clamp(min=1e-8)  # (d_phi,)
        logits = -0.5 * (diff ** 2 / sigma2.unsqueeze(0)).sum(dim=-1)  # (|S_t|,)
    # Temperature scaling: divide logits by temperature
    logits = logits / max(temperature, 1e-8)
    log_pi = F.log_softmax(logits, dim=0)  # (|S_t|,)
    return log_pi


def sample_from_pi(log_pi: torch.Tensor) -> int:
    """Sample one index from log_pi. Returns int index into S_t."""
    probs = torch.exp(log_pi)
    idx = torch.multinomial(probs, num_samples=1).item()
    return idx


def sample_M_from_pi(log_pi: torch.Tensor, M: int) -> torch.Tensor:
    """Sample M indices (with replacement) from log_pi."""
    probs = torch.exp(log_pi)
    return torch.multinomial(probs, num_samples=M, replacement=True)


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
    Uses batched forward passes for GPU efficiency.
    batch_size=0 means process all at once.

    token_ids: list[list[int]] OR pre-built (N, T) tensor on *device*.
    """
    if isinstance(token_ids, torch.Tensor):
        all_ids = token_ids  # already (N, T) on device
    else:
        all_ids = _tokens_to_tensor(token_ids, device)
    N = all_ids.size(0)
    bs = N if batch_size <= 0 else batch_size
    total_lp = 0.0
    for start in range(0, N, bs):
        chunk = all_ids[start : start + bs]           # (B, T)
        lp, _ = model.forward_lm(chunk)
        # forward_lm returns a single scalar mean over all tokens in the batch,
        # so we weight by the number of examples in this chunk.
        total_lp += lp.item() * chunk.size(0)
    return total_lp / N


# ---------------------------------------------------------------------------
# Utility: compute Q-values for a batch of context windows (batched)
# ---------------------------------------------------------------------------

def compute_q_values(
    model: CurriculumGPT2,
    token_ids,
    device: torch.device,
    batch_size: int = 0,
) -> torch.Tensor:
    """
    Compute Q_theta(x) for each x in the batch.
    Returns (N,) tensor of Q-values (with grad through W_Q and backbone).
    Uses batched forward passes for GPU efficiency.

    token_ids: list[list[int]] OR pre-built (N, T) tensor on *device*.
    """
    if isinstance(token_ids, torch.Tensor):
        all_ids = token_ids
    else:
        all_ids = _tokens_to_tensor(token_ids, device)
    N = all_ids.size(0)
    bs = N if batch_size <= 0 else batch_size
    q_chunks = []
    for start in range(0, N, bs):
        chunk = all_ids[start : start + bs]             # (B, T)
        _, h_last = model.forward_lm(chunk)              # (B, d)
        q = model.q_value(h_last)                        # (B,)
        q_chunks.append(q)
    return torch.cat(q_chunks)


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
            lp, _ = model.forward_lm(input_ids)
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
            logits, _ = model._get_hidden_states(input_ids)
            probs = F.softmax(logits[:, :-1, :], dim=-1)
            ent = -(probs * probs.clamp(min=1e-12).log()).sum(-1).mean().item()
            if ent > best_ent:
                best_ent = ent
                best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Diagnostics: gradient norms, weight norms, policy stats
# ---------------------------------------------------------------------------

def _grad_norm(model: CurriculumGPT2) -> dict:
    """Compute per-component gradient norms (after backward, before step)."""
    backbone_grads, mu_grads, gamma_grads, q_grads = [], [], [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().norm().item()
        if name.startswith("W_mu"):
            mu_grads.append(g)
        elif name.startswith("W_gamma"):
            gamma_grads.append(g)
        elif name.startswith("W_Q"):
            q_grads.append(g)
        else:
            backbone_grads.append(g)
    def _rms(xs):
        return math.sqrt(sum(x**2 for x in xs) / max(len(xs), 1)) if xs else 0.0
    return {
        "grad_norm_backbone": _rms(backbone_grads),
        "grad_norm_W_mu": _rms(mu_grads),
        "grad_norm_W_gamma": _rms(gamma_grads),
        "grad_norm_W_Q": _rms(q_grads),
    }


def _weight_norms(model: CurriculumGPT2) -> dict:
    """Compute weight norms for the projection heads."""
    return {
        "wnorm_W_mu": model.W_mu.weight.detach().norm().item(),
        "wnorm_W_gamma": model.W_gamma.weight.detach().norm().item(),
        "wnorm_W_Q": model.W_Q.weight.detach().norm().item(),
    }


def _policy_stats(log_pi: torch.Tensor, log_sigma2: torch.Tensor, fix_sigma: bool = False) -> dict:
    """Compute policy distribution diagnostics."""
    pi = torch.exp(log_pi.detach())
    entropy = -(pi * log_pi.detach()).sum().item()
    max_prob = pi.max().item()
    min_prob = pi.min().item()
    effective_n = math.exp(entropy) if entropy > 0 else 1.0
    stats = {
        "pi_entropy": entropy,
        "pi_max_prob": max_prob,
        "pi_min_prob": min_prob,
        "pi_effective_n": effective_n,
    }
    # Only log sigma2 stats when sigma is learned (not fixed)
    if not fix_sigma:
        sigma2 = torch.exp(log_sigma2.detach())
        stats.update({
            "sigma2_mean": sigma2.mean().item(),
            "sigma2_std": sigma2.std().item(),
            "sigma2_min": sigma2.min().item(),
            "sigma2_max": sigma2.max().item(),
            "log_sigma2_mean": log_sigma2.detach().mean().item(),
        })
    return stats


def _gpu_stats() -> dict:
    """GPU memory usage (if CUDA available)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        "gpu_mem_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }


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

    # --- Tokenizer (needed before model for embedder) ---
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Embedder (sets cfg.d_phi) ---
    embedder = make_embedder(cfg, tokenizer)
    logging.info("Embedding dimension (d_phi): %d", cfg.d_phi)

    # --- Model (needs d_phi set) ---
    model = CurriculumGPT2(cfg).to(device)

    # Single optimizer for all parameters (shared lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- Data ---
    stream = DataStream(cfg, tokenizer)

    # --- Fixed eval set (drawn once, used throughout) ---
    logging.info("Drawing %d fixed eval windows ...", cfg.num_eval_windows)
    eval_tokens_list = stream.get_batch(cfg.num_eval_windows)
    eval_ids = _tokens_to_tensor(eval_tokens_list, device)  # (num_eval, T) on GPU
    init_ppl = evaluate_perplexity(model, eval_ids, device, batch_size=cfg.batch_size)
    logging.info("Initial eval perplexity: %.2f", init_ppl)

    # --- Logging ---
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = Path(cfg.checkpoint_dir) / "log.jsonl"
    # Clear log from previous runs so plots don't overlay stale data
    if log_path.exists():
        log_path.unlink()
        logging.info("Cleared old log file: %s", log_path)
    # Generate a unique run ID
    run_id = time.strftime("%Y%m%d_%H%M%S")
    logging.info("Run ID: %s", run_id)
    # Write config
    with open(Path(cfg.checkpoint_dir) / "config.json", "w") as f:
        cfg_dict = cfg.__dict__.copy()
        cfg_dict["run_id"] = run_id
        json.dump(cfg_dict, f, indent=2)

    logging.info(
        "Starting training: %d outer steps x %d inner steps  (curriculum=%s)",
        cfg.outer_steps, cfg.inner_steps, cfg.curriculum,
    )

    # Policy temperature: auto = scale by d_phi so logits are O(1)
    if cfg.policy_temp > 0:
        policy_temp = cfg.policy_temp
    else:
        policy_temp = float(cfg.d_phi)  # auto: divide by d_phi
    logging.info("Policy temperature: %.1f  (d_phi=%d)", policy_temp, cfg.d_phi)

    total_examples_seen = 0
    cumulative_reward = 0.0
    train_start_time = time.time()

    for t in range(cfg.outer_steps):
        outer_t0 = time.time()

        # === Outer loop: refresh S_t, D_t ===
        t_data = time.time()
        S_t_tokens, D_t_tokens = stream.get_stream_step()
        D_t_ids = _tokens_to_tensor(D_t_tokens, device)  # (|D_t|, T) on GPU
        dt_data = time.time() - t_data

        # Pre-compute embeddings for S_t  (only needed for RL mode)
        E_S = None
        dt_embed = 0.0
        if cfg.curriculum == "rl":
            t_emb = time.time()
            E_S = embedder.embed_batch(S_t_tokens).to(device)  # (|S_t|, d_phi)
            dt_embed = time.time() - t_emb

        # Cache held-out log-prob before inner loop (for reward computation)
        t_held = time.time()
        prev_heldout_lp = mean_log_prob_dataset(model, D_t_ids, device, batch_size=cfg.batch_size)
        dt_heldout_init = time.time() - t_held

        for k in range(cfg.inner_steps):
            step_t0 = time.time()
            timings = {}

            # ==============================================================
            # Step 1: Select a training example
            # ==============================================================
            t0 = time.time()
            if cfg.curriculum == "rl":
                idx_k, log_pi_k, z_k, mu_k, log_sigma2_k = _rl_select(
                    model, E_S, device, fix_sigma=cfg.fix_sigma,
                    temperature=policy_temp,
                )
            elif cfg.curriculum == "random":
                idx_k = select_random(S_t_tokens)
            elif cfg.curriculum == "loss":
                idx_k = select_loss_based(model, S_t_tokens, device)
            elif cfg.curriculum == "uncertainty":
                idx_k = select_uncertainty_based(model, S_t_tokens, device)
            else:
                raise ValueError(f"Unknown curriculum: {cfg.curriculum}")
            timings["t_select"] = round(time.time() - t0, 4)

            x_k_tokens = S_t_tokens[idx_k]
            x_k_text_snippet = tokenizer.decode(x_k_tokens[:60]).replace("\n", " ")[:200]
            total_examples_seen += 1

            # ==============================================================
            # Step 6 (pre-computed): Baseline V_k under theta_k
            #   Must be computed here, before the LM step changes theta.
            #   We sample M candidates from pi_{theta_k} and average Q_{theta_k}.
            # ==============================================================
            V_k_val = None
            if cfg.curriculum == "rl":
                t0 = time.time()
                sample_indices_k = sample_M_from_pi(log_pi_k.detach(), cfg.M)
                sampled_tokens_k = [S_t_tokens[i] for i in sample_indices_k]
                with torch.no_grad():
                    q_samples_k = compute_q_values(model, sampled_tokens_k, device, batch_size=cfg.batch_size)
                    V_k_val = q_samples_k.mean().item()
                timings["t_V_k"] = round(time.time() - t0, 4)

            # ==============================================================
            # Step 2: Language-modeling update (MDP state transition)
            #   theta_{k+1} = theta_k + alpha * grad log p(x_k)
            # ==============================================================
            t0 = time.time()
            input_ids_xk = torch.tensor([x_k_tokens], device=device)
            lm_logp, _ = model.forward_lm(input_ids_xk)
            lm_loss = -lm_logp  # minimise negative log-prob

            optimizer.zero_grad()
            lm_loss.backward()
            lm_grad_stats = _grad_norm(model)
            lm_grad_total = math.sqrt(sum(
                p.grad.detach().norm().item()**2
                for p in model.parameters() if p.grad is not None
            ))
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            timings["t_lm_update"] = round(time.time() - t0, 4)
            # model now holds theta_{k+1}

            # ==============================================================
            # Step 3: Reward  r_k = mean_lp(D, theta_{k+1}) - mean_lp(D, theta_k)
            # ==============================================================
            t0 = time.time()
            new_heldout_lp = mean_log_prob_dataset(model, D_t_ids, device, batch_size=cfg.batch_size)
            r_k = new_heldout_lp - prev_heldout_lp
            cumulative_reward += r_k
            timings["t_reward"] = round(time.time() - t0, 4)

            # ==============================================================
            # Steps 4-7: RL loss (only in RL curriculum mode)
            # ==============================================================
            rl_stats = {}
            if cfg.curriculum == "rl":
                t0 = time.time()
                rl_stats = _rl_update(
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    device=device,
                    S_t_tokens=S_t_tokens,
                    E_S=E_S,
                    x_k_tokens=x_k_tokens,
                    idx_k=idx_k,
                    r_k=r_k,
                    V_k=V_k_val,
                    fix_sigma=cfg.fix_sigma,
                    temperature=policy_temp,
                )
                timings["t_rl_total"] = round(time.time() - t0, 4)

            # Update cached held-out log-prob for next step
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
                    "r_k": r_k,
                    "cumulative_reward": cumulative_reward,
                    "heldout_lp": new_heldout_lp,
                    "lm_loss": lm_loss.item(),
                    # -- LM gradient norms --
                    "lm_grad_total": lm_grad_total,
                    **{f"lm_{k2}": v for k2, v in lm_grad_stats.items()},
                    # -- Weight norms --
                    **_weight_norms(model),
                    # -- GPU --
                    **_gpu_stats(),
                    # -- RL-specific --
                    **rl_stats,
                }
                logging.info(
                    "[t=%d k=%d] r=%.5f heldout=%.4f lm=%.4f grad=%.3f (%.2fs: sel=%.3f lm=%.3f rew=%.3f rl=%.3f)%s",
                    t, k, r_k, new_heldout_lp, lm_loss.item(),
                    lm_grad_total, step_time,
                    timings.get("t_select", 0), timings.get("t_lm_update", 0),
                    timings.get("t_reward", 0), timings.get("t_rl_total", 0),
                    f"  Q={rl_stats.get('q_xk', 0):.4f} A={rl_stats.get('A_k', 0):.4f}"
                    if rl_stats else "",
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
                "[Outer %d] ppl=%.2f eval_lp=%.4f (ex=%d, %.0fs) outer=%.1fs (data=%.2f emb=%.2f held=%.2f eval=%.2f)",
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
                **_weight_norms(model),
                **_gpu_stats(),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(eval_entry) + "\n")

        if cfg.save_every_outer > 0 and (t + 1) % cfg.save_every_outer == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"model_outer_{t:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logging.info("[Outer %d] Checkpoint saved to %s", t, ckpt_path)

    logging.info("Training complete.  Total examples seen: %d", total_examples_seen)


# ---------------------------------------------------------------------------
# RL-specific helpers (kept separate for clarity)
# ---------------------------------------------------------------------------

def _rl_select(
    model: CurriculumGPT2,
    E_S: torch.Tensor,
    device: torch.device,
    fix_sigma: bool = False,
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Step 1 of the algorithm: compute policy and sample an example.
    Returns (idx_k, log_pi_k, z_k, mu_k, log_sigma2_k).
    """
    z_k = model.get_policy_state(device)              # (d,)
    mu_k, log_sigma2_k = model.policy_params(z_k)     # (d_phi,) each
    log_pi_k = compute_log_pi(mu_k, log_sigma2_k, E_S, fix_sigma=fix_sigma,
                              temperature=temperature)
    idx_k = sample_from_pi(log_pi_k)
    return idx_k, log_pi_k, z_k, mu_k, log_sigma2_k


def _rl_update(
    *,
    model: CurriculumGPT2,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    S_t_tokens: list[list[int]],
    E_S: torch.Tensor,
    x_k_tokens: list[int],
    idx_k: int,
    r_k: float,
    V_k: float,
    fix_sigma: bool = False,
    temperature: float = 1.0,
) -> dict:
    """
    Steps 4, 5, 7 of the algorithm (Step 6 / V_k is pre-computed).

    At entry, model holds theta_{k+1} (after the LM step).
    At exit, model holds theta_{k+1} with the RL gradient applied on top
    (i.e. the combined update from both the LM transition and RL loss).

    V_k (the baseline value) is computed in the main loop *before* the LM
    step, while the model is still at theta_k.  This ensures V_k reflects
    the state in which the action was selected, as the paper specifies.

    The paper specifies:
      - Q_{theta_k}(x_k) = W_Q * u_k  where u_k = f_{theta_{k+1}}^{(L-1)}(x_k^last)
        (eq 6: Q uses W_Q but u_k comes from theta_{k+1} forward pass)
      - The Bellman loss LHS is Q_{theta_k}(x_k), target is r_k + gamma * sg(V_{k+1})
      - The policy loss uses advantage A_k = sg(Q_{theta_k}(x_k) - V_k)
      - Gradients of L_k flow into W_mu, W_gamma, W_Q, and the backbone

    Implementation notes:
      We are at theta_{k+1}.  We compute u_k (hidden for x_k) under theta_{k+1}
      and Q(x_k) = W_Q @ u_k.  V_{k+1} is computed under theta_{k+1} (correct).
      V_k was computed under theta_k before the LM step (correct).
    """
    # ---- Step 4: Q-value of selected example under theta_{k+1} ----
    t0 = time.time()
    input_ids_xk = torch.tensor([x_k_tokens], device=device)
    _, h_last_k = model.forward_lm(input_ids_xk)  # u_k under theta_{k+1}
    q_xk = model.q_value(h_last_k.squeeze(0))      # scalar, has grad
    dt_q_xk = time.time() - t0

    # ---- Step 5: Next-state value V_{k+1} ----
    t0 = time.time()
    z_kp1 = model.get_policy_state(device)
    mu_kp1, log_sigma2_kp1 = model.policy_params(z_kp1)
    log_pi_kp1 = compute_log_pi(mu_kp1, log_sigma2_kp1, E_S, fix_sigma=fix_sigma,
                                temperature=temperature)

    sample_indices_kp1 = sample_M_from_pi(log_pi_kp1.detach(), cfg.M)
    sampled_tokens_kp1 = [S_t_tokens[i] for i in sample_indices_kp1]
    with torch.no_grad():
        q_samples_kp1 = compute_q_values(model, sampled_tokens_kp1, device, batch_size=cfg.M)
        V_kp1 = q_samples_kp1.mean()
    dt_V_kp1 = time.time() - t0

    # ---- Step 6: Baseline value V_k (pre-computed under theta_k) ----
    # V_k was computed in the main loop before the LM step, passed in as a float.

    # ---- Step 7: Combined loss ----
    t0 = time.time()
    # Bellman target (stop-gradient)
    y_k = r_k + cfg.gamma * V_kp1.item()
    L_Q = (q_xk - y_k) ** 2

    # Policy log-prob under current params (has grad through W_mu, W_gamma, backbone)
    z_k_pg = model.get_policy_state(device)
    mu_k_pg, log_sigma2_k_pg = model.policy_params(z_k_pg)
    log_pi_k_pg = compute_log_pi(mu_k_pg, log_sigma2_k_pg, E_S, fix_sigma=fix_sigma,
                                temperature=temperature)

    # Advantage (stop-gradient)
    A_k = q_xk.detach().item() - V_k

    L_pi = -A_k * log_pi_k_pg[idx_k]

    L_k = L_Q + L_pi

    optimizer.zero_grad()
    L_k.backward()
    rl_grad_stats = _grad_norm(model)
    rl_grad_total = math.sqrt(sum(
        p.grad.detach().norm().item()**2
        for p in model.parameters() if p.grad is not None
    ))
    if cfg.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    dt_pg_step = time.time() - t0

    # Policy distribution diagnostics
    pol_stats = _policy_stats(log_pi_k_pg, log_sigma2_k_pg, fix_sigma=fix_sigma)

    # Bellman TD error
    td_error = q_xk.item() - y_k

    return {
        # -- Q / value / advantage --
        "q_xk": q_xk.item(),
        "V_k": V_k,
        "V_kp1": V_kp1.item(),
        "A_k": A_k,
        "td_error": td_error,
        "bellman_target": y_k,
        # -- Losses --
        "L_Q": L_Q.item(),
        "L_pi": L_pi.item(),
        "L_k": L_k.item(),
        # -- RL gradient norms --
        "rl_grad_total": rl_grad_total,
        **{f"rl_{k2}": v for k2, v in rl_grad_stats.items()},
        # -- Policy distribution --
        **pol_stats,
        # -- RL sub-timings --
        "t_q_xk": round(dt_q_xk, 4),
        "t_V_kp1": round(dt_V_kp1, 4),
        "t_pg_step": round(dt_pg_step, 4),
    }


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
