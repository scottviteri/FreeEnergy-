"""
Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models.

Implementation of the algorithm described in project.tex.
Uses GPT-2 (small) as the transformer and OpenAI text-embedding-3-small as phi.
"""

import os
import copy
import math
import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import openai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Model
    model_name: str = "gpt2"
    context_length: int = 256  # tokens per context window

    # Embedding model (OpenAI)
    embedding_model: str = "text-embedding-3-small"
    d_phi: int = 1536  # dimension of text-embedding-3-small

    # Data streaming
    num_candidates: int = 90   # |S_t|  (scaled down for laptop GPU)
    num_heldout: int = 10      # |D_t|
    outer_steps: int = 50      # number of outer-loop stream steps
    inner_steps: int = 20      # K inner-loop steps per outer step

    # RL
    gamma: float = 0.99        # discount factor
    lr: float = 1e-4           # shared learning rate alpha
    M: int = 8                 # number of samples for V estimation

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_every: int = 1         # log every N inner steps
    data_name: str = "openwebtext"
    data_split: str = "train"
    embedding_cache_path: str = "embedding_cache.json"
    checkpoint_dir: str = "checkpoints"


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
            cfg.data_name, split=cfg.data_split, streaming=True, trust_remote_code=True
        )
        self.iterator = iter(self.dataset)
        self.token_buffer: list[int] = []

    def _refill_buffer(self, need: int):
        """Tokenise more documents until we have at least `need` tokens."""
        while len(self.token_buffer) < need:
            try:
                doc = next(self.iterator)
            except StopIteration:
                # restart stream
                self.iterator = iter(self.dataset)
                doc = next(self.iterator)
            tokens = self.tokenizer.encode(doc["text"])
            self.token_buffer.extend(tokens)

    def get_batch(self, n: int) -> list[list[int]]:
        """Return n context windows, each of length cfg.context_length tokens."""
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
# Embedding via OpenAI API (with disk cache)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Caches OpenAI embeddings keyed by text to avoid redundant API calls."""

    def __init__(self, cfg: Config, tokenizer: GPT2Tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.client = openai.OpenAI()  # uses OPENAI_API_KEY env var
        self.cache: dict[str, list[float]] = {}
        self._load_cache()

    def _load_cache(self):
        p = Path(self.cfg.embedding_cache_path)
        if p.exists():
            with open(p) as f:
                self.cache = json.load(f)
            logging.info("Loaded %d cached embeddings.", len(self.cache))

    def _save_cache(self):
        with open(self.cfg.embedding_cache_path, "w") as f:
            json.dump(self.cache, f)

    def _text_key(self, token_ids: list[int]) -> str:
        """Deterministic short key for a token list (first 128 chars of decoded text)."""
        text = self.tokenizer.decode(token_ids[:64])
        return text[:128]

    def embed_batch(self, token_id_lists: list[list[int]]) -> torch.Tensor:
        """
        Return (N, d_phi) tensor of embeddings for N context windows.
        Uses cache; calls OpenAI API only for misses.
        """
        texts = [self.tokenizer.decode(ids) for ids in token_id_lists]
        keys = [t[:128] for t in texts]

        # find misses
        miss_indices = [i for i, k in enumerate(keys) if k not in self.cache]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            # OpenAI batch embed (max 2048 per call, we're well under)
            response = self.client.embeddings.create(
                model=self.cfg.embedding_model,
                input=miss_texts,
            )
            for idx, emb_obj in zip(miss_indices, response.data):
                self.cache[keys[idx]] = emb_obj.embedding
            self._save_cache()

        embeddings = [self.cache[k] for k in keys]
        return torch.tensor(embeddings, dtype=torch.float32)


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

        # The second-to-last layer index
        self.hook_layer = n_layers - 2  # layer L-2 (0-indexed), i.e. layer index 10

        # Projection heads
        self.W_mu = nn.Linear(d_model, cfg.d_phi, bias=False)
        self.W_gamma = nn.Linear(d_model, cfg.d_phi, bias=False)
        self.W_Q = nn.Linear(d_model, 1, bias=False)

        # Init heads small
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
        # index 0 = embedding, index i = after layer i-1
        # second-to-last layer = index (n_layer - 1) = hook_layer + 1
        hidden = outputs.hidden_states[self.hook_layer + 1]  # (B, T, d)
        return outputs.logits, hidden

    def forward_lm(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (mean_log_prob, hidden_last_token).
        mean_log_prob: scalar, mean log p(x) over tokens (teacher-forced).
        hidden_last_token: (d,) activation at last real token from layer L-1.
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
            # find last real token position per batch element
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
        """
        Given z (d,), return mu (d_phi,) and log_sigma2 (d_phi,).
        """
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
) -> torch.Tensor:
    """
    Compute log pi(x) for all x in S_t.

    mu:         (d_phi,)
    log_sigma2: (d_phi,)
    E_S:        (|S_t|, d_phi)

    Returns log_pi: (|S_t|,)  (log-normalised over S_t)

    pi(x) = N(phi(x); mu, diag(sigma^2)) / Z
    log pi(x) = log N(phi(x); mu, diag(sigma^2)) - log Z
    log N(phi(x); mu, diag(sigma^2)) = -0.5 * sum_d [ (phi(x)_d - mu_d)^2 / sigma^2_d + log sigma^2_d + log(2pi) ]

    Since the constant terms (-0.5 * sum(log sigma^2 + log 2pi)) are the same for all x,
    they cancel in the normalisation. So we only need the quadratic part for the logits:
    logit(x) = -0.5 * sum_d (phi(x)_d - mu_d)^2 / sigma^2_d
    """
    sigma2 = torch.exp(log_sigma2)  # (d_phi,)
    diff = E_S - mu.unsqueeze(0)     # (|S_t|, d_phi)
    logits = -0.5 * (diff ** 2 / sigma2.unsqueeze(0)).sum(dim=-1)  # (|S_t|,)
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
# Utility: compute mean log-prob of a set of context windows
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_log_prob_dataset(
    model: CurriculumGPT2,
    token_id_lists: list[list[int]],
    device: torch.device,
) -> float:
    """Compute (1/|D|) sum_x log p_theta(x) over a list of context windows."""
    total_lp = 0.0
    for ids in token_id_lists:
        input_ids = torch.tensor([ids], device=device)
        lp, _ = model.forward_lm(input_ids)
        total_lp += lp.item()
    return total_lp / len(token_id_lists)


# ---------------------------------------------------------------------------
# Utility: compute Q-values for a batch of context windows
# ---------------------------------------------------------------------------

def compute_q_values(
    model: CurriculumGPT2,
    token_id_lists: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute Q_theta(x) for each x in the list.
    Returns (N,) tensor of Q-values (with grad through W_Q and backbone).
    """
    q_vals = []
    for ids in token_id_lists:
        input_ids = torch.tensor([ids], device=device)
        _, h_last = model.forward_lm(input_ids)
        q = model.q_value(h_last)
        q_vals.append(q)
    return torch.stack(q_vals)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    logging.info("Device: %s", device)

    # --- Model ---
    model = CurriculumGPT2(cfg).to(device)
    tokenizer = model.tokenizer

    # Single optimizer for all parameters (shared lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # --- Data ---
    stream = DataStream(cfg, tokenizer)
    embedder = EmbeddingCache(cfg, tokenizer)

    # --- Logging ---
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = Path(cfg.checkpoint_dir) / "log.jsonl"

    logging.info(
        "Starting training: %d outer steps x %d inner steps",
        cfg.outer_steps,
        cfg.inner_steps,
    )

    for t in range(cfg.outer_steps):
        # === Outer loop: refresh S_t, D_t ===
        S_t_tokens, D_t_tokens = stream.get_stream_step()

        # Pre-compute embeddings for S_t (phi(x) for all x in S_t)
        logging.info("[Outer %d] Embedding %d candidates ...", t, len(S_t_tokens))
        E_S = embedder.embed_batch(S_t_tokens).to(device)  # (|S_t|, d_phi)

        # Cache held-out log-prob before inner loop
        prev_heldout_lp = mean_log_prob_dataset(model, D_t_tokens, device)

        for k in range(cfg.inner_steps):
            step_t0 = time.time()

            # ---- Step 1: Select a training example ----
            z_k = model.get_policy_state(device)          # (d,)
            mu_k, log_sigma2_k = model.policy_params(z_k) # (d_phi,) each
            log_pi_k = compute_log_pi(mu_k, log_sigma2_k, E_S)  # (|S_t|,)

            idx_k = sample_from_pi(log_pi_k)
            x_k_tokens = S_t_tokens[idx_k]

            # ---- Step 2: Language-modeling update (state transition) ----
            # We need theta_k and theta_{k+1} in memory.
            # Save theta_k
            theta_k_state = copy.deepcopy(model.state_dict())

            # Compute LM loss and do gradient step
            input_ids_xk = torch.tensor([x_k_tokens], device=device)
            lm_logp, _ = model.forward_lm(input_ids_xk)
            lm_loss = -lm_logp  # maximise log p => minimise -log p
            optimizer.zero_grad()
            lm_loss.backward()
            optimizer.step()
            # Now model has theta_{k+1}

            # ---- Step 3: Reward ----
            new_heldout_lp = mean_log_prob_dataset(model, D_t_tokens, device)
            r_k = new_heldout_lp - prev_heldout_lp

            # ---- Step 4: Q-value of selected example ----
            # u_k from theta_{k+1} forward pass on x_k (byproduct of reward computation)
            # We re-run on x_k under theta_{k+1} to get u_k with grad
            input_ids_xk = torch.tensor([x_k_tokens], device=device)
            _, h_last_k = model.forward_lm(input_ids_xk)  # under theta_{k+1}
            q_xk = model.q_value(h_last_k)  # scalar, has grad through W_Q

            # ---- Step 5: Next-state value V_{k+1} ----
            # Policy at theta_{k+1}
            z_kp1 = model.get_policy_state(device)
            mu_kp1, log_sigma2_kp1 = model.policy_params(z_kp1)
            log_pi_kp1 = compute_log_pi(mu_kp1, log_sigma2_kp1, E_S)

            sample_indices_kp1 = sample_M_from_pi(log_pi_kp1.detach(), cfg.M)
            sampled_tokens_kp1 = [S_t_tokens[i] for i in sample_indices_kp1]
            with torch.no_grad():
                q_samples_kp1 = compute_q_values(model, sampled_tokens_kp1, device)
                V_kp1 = q_samples_kp1.mean()

            # ---- Step 6: Baseline value V_k ----
            # We need pi at theta_k, so temporarily load theta_k
            theta_kp1_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(theta_k_state)

            z_k_for_baseline = model.get_policy_state(device)
            mu_k_bl, log_sigma2_k_bl = model.policy_params(z_k_for_baseline)
            log_pi_k_bl = compute_log_pi(mu_k_bl, log_sigma2_k_bl, E_S)

            sample_indices_k = sample_M_from_pi(log_pi_k_bl.detach(), cfg.M)
            sampled_tokens_k = [S_t_tokens[i] for i in sample_indices_k]
            with torch.no_grad():
                q_samples_k = compute_q_values(model, sampled_tokens_k, device)
                V_k = q_samples_k.mean()

            # ---- Step 7: Combined loss L_k = L_Q + L_pi ----
            # We need to compute gradients w.r.t. theta_k.
            # Recompute Q(x_k) and log_pi(x_k) under theta_k with grad.

            # Q(x_k) under theta_k: forward pass on x_k
            input_ids_xk = torch.tensor([x_k_tokens], device=device)
            _, h_last_k_old = model.forward_lm(input_ids_xk)
            q_xk_old = model.q_value(h_last_k_old)  # has grad

            # Bellman target (stop-gradient)
            y_k = r_k + cfg.gamma * V_kp1  # both are plain floats/detached
            L_Q = (q_xk_old - y_k) ** 2

            # Policy log-prob under theta_k (has grad through W_mu, W_gamma, backbone)
            z_k_pg = model.get_policy_state(device)
            mu_k_pg, log_sigma2_k_pg = model.policy_params(z_k_pg)
            log_pi_k_pg = compute_log_pi(mu_k_pg, log_sigma2_k_pg, E_S)

            # Advantage (stop-gradient)
            A_k = (q_xk_old.detach() - V_k).detach()

            L_pi = -A_k * log_pi_k_pg[idx_k]

            L_k = L_Q + L_pi

            optimizer.zero_grad()
            L_k.backward()
            optimizer.step()

            # Restore theta_{k+1} and apply the RL gradient on top
            # Actually: the RL gradient was applied to theta_k.
            # The final state after this inner step should be:
            #   theta_{k+1} (from LM step) + RL gradient step.
            # We applied LM step first (model had theta_{k+1}), then loaded theta_k
            # for the RL loss. Now we need to combine them.
            #
            # Correct approach: the LM update is the state transition, and the RL
            # loss updates the same parameters. So after one inner step:
            #   theta_{k+1}' = theta_k + alpha * grad_LM + alpha * grad_RL
            #
            # Since we did LM step (theta_k -> theta_{k+1}), then loaded theta_k
            # and did RL step (theta_k -> theta_k + RL_grad), we need to add the
            # LM gradient delta back.
            #
            # current model = theta_k + RL_grad_step
            # we want = theta_{k+1} + RL_grad_step = (theta_k + LM_grad_step) + RL_grad_step
            # delta_LM = theta_{k+1} - theta_k
            # So: add delta_LM to current model

            current_state = model.state_dict()
            for key in current_state:
                lm_delta = theta_kp1_state[key] - theta_k_state[key]
                current_state[key] = current_state[key] + lm_delta
            model.load_state_dict(current_state)

            # Update cached held-out log-prob for next step
            prev_heldout_lp = new_heldout_lp

            step_time = time.time() - step_t0
            if k % cfg.log_every == 0:
                log_entry = {
                    "outer_step": t,
                    "inner_step": k,
                    "r_k": r_k,
                    "q_xk": q_xk_old.item(),
                    "V_k": V_k.item(),
                    "V_kp1": V_kp1.item(),
                    "A_k": A_k.item(),
                    "L_Q": L_Q.item(),
                    "L_pi": L_pi.item(),
                    "L_k": L_k.item(),
                    "heldout_lp": new_heldout_lp,
                    "lm_loss": lm_loss.item(),
                    "step_time": round(step_time, 2),
                }
                logging.info(
                    "[t=%d k=%d] r=%.4f Q=%.4f A=%.4f L=%.4f heldout=%.4f (%.1fs)",
                    t, k, r_k, q_xk_old.item(), A_k.item(),
                    L_k.item(), new_heldout_lp, step_time,
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        # Save checkpoint after each outer step
        ckpt_path = Path(cfg.checkpoint_dir) / f"model_outer_{t:04d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        logging.info("[Outer %d] Checkpoint saved to %s", t, ckpt_path)

    logging.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config()
    train(cfg)

