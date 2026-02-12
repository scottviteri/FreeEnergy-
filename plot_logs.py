#!/usr/bin/env python3
"""
Asynchronous log plotter for curriculum RL training.

Reads checkpoints/log.jsonl periodically and generates plots.
Can be edited and re-run mid-training -- it reads the log file fresh each time.

Usage:
    python plot_logs.py [--log_path checkpoints/log.jsonl] [--out_dir plots] [--interval 30] [--once]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(path: str) -> tuple[list[dict], list[dict]]:
    """Load log.jsonl, return (step_entries, eval_entries)."""
    steps, evals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "eval":
                evals.append(entry)
            else:
                steps.append(entry)
    return steps, evals


def get(entries: list[dict], key: str) -> tuple[list, list]:
    """Extract (x=global_step or total_examples, y=key) pairs, skipping missing."""
    xs, ys = [], []
    for e in entries:
        if key in e:
            x = e.get("global_step", e.get("total_examples", len(xs)))
            xs.append(x)
            ys.append(e[key])
    return xs, ys


def get_eval(entries: list[dict], key: str) -> tuple[list, list]:
    """Extract (x=total_examples, y=key) from eval entries."""
    xs, ys = [], []
    for e in entries:
        if key in e:
            xs.append(e.get("total_examples", len(xs)))
            ys.append(e[key])
    return xs, ys


def smooth(ys, window=10):
    """Simple moving average."""
    if len(ys) < window:
        return ys
    kernel = np.ones(window) / window
    return np.convolve(ys, kernel, mode="valid").tolist()


def smooth_xy(xs, ys, window=10):
    """Smooth ys and trim xs to match."""
    sy = smooth(ys, window)
    offset = len(ys) - len(sy)
    return xs[offset:], sy


# ---------------------------------------------------------------------------
# Plotting functions -- each creates one figure
# ---------------------------------------------------------------------------

def plot_core_metrics(steps, evals, out_dir):
    """Reward, heldout log-prob, LM loss, cumulative reward."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Core Training Metrics", fontsize=14, fontweight="bold")

    # Reward per step
    ax = axes[0, 0]
    xs, ys = get(steps, "r_k")
    if xs:
        ax.plot(xs, ys, alpha=0.3, color="C0", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
        ax.plot(sx, sy, color="C0", linewidth=1.5, label="smoothed")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Reward (r_k)")
    ax.set_xlabel("global step")
    ax.set_ylabel("r_k")
    ax.legend(fontsize=8)

    # Heldout log-prob
    ax = axes[0, 1]
    xs, ys = get(steps, "heldout_lp")
    if xs:
        ax.plot(xs, ys, color="C1", linewidth=1)
    ax.set_title("Held-out Log-Prob")
    ax.set_xlabel("global step")
    ax.set_ylabel("mean log p(D)")

    # LM loss
    ax = axes[1, 0]
    xs, ys = get(steps, "lm_loss")
    if xs:
        ax.plot(xs, ys, alpha=0.3, color="C2", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
        ax.plot(sx, sy, color="C2", linewidth=1.5, label="smoothed")
    ax.set_title("LM Loss (selected example)")
    ax.set_xlabel("global step")
    ax.set_ylabel("-log p(x_k)")
    ax.legend(fontsize=8)

    # Cumulative reward
    ax = axes[1, 1]
    xs, ys = get(steps, "cumulative_reward")
    if xs:
        ax.plot(xs, ys, color="C3", linewidth=1.5)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel("global step")
    ax.set_ylabel("sum(r_k)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "01_core_metrics.png", dpi=150)
    plt.close(fig)


def plot_eval_perplexity(steps, evals, out_dir):
    """Eval perplexity over training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Evaluation Metrics", fontsize=14, fontweight="bold")

    # Perplexity vs examples seen
    ax = axes[0]
    xs, ys = get_eval(evals, "perplexity")
    if xs:
        ax.plot(xs, ys, "o-", color="C4", markersize=4, linewidth=1.5)
    ax.set_title("Eval Perplexity vs Examples Seen")
    ax.set_xlabel("total examples")
    ax.set_ylabel("perplexity")

    # Eval log-prob vs examples seen
    ax = axes[1]
    xs, ys = get_eval(evals, "eval_lp")
    if xs:
        ax.plot(xs, ys, "s-", color="C5", markersize=4, linewidth=1.5)
    ax.set_title("Eval Log-Prob vs Examples Seen")
    ax.set_xlabel("total examples")
    ax.set_ylabel("mean log p(eval)")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "02_eval_perplexity.png", dpi=150)
    plt.close(fig)


def plot_rl_losses(steps, evals, out_dir):
    """Q-loss, policy loss, total RL loss, TD error."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL Losses & Values", fontsize=14, fontweight="bold")

    for ax, key, title, color in [
        (axes[0, 0], "L_Q", "Bellman Loss (L_Q)", "C0"),
        (axes[0, 1], "L_pi", "Policy Loss (L_pi)", "C1"),
        (axes[1, 0], "L_k", "Total RL Loss (L_k)", "C2"),
        (axes[1, 1], "td_error", "TD Error (Q - y)", "C3"),
    ]:
        xs, ys = get(steps, key)
        if xs:
            ax.plot(xs, ys, alpha=0.3, color=color, linewidth=0.5)
            sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
            ax.plot(sx, sy, color=color, linewidth=1.5)
            if key == "td_error":
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("global step")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "03_rl_losses.png", dpi=150)
    plt.close(fig)


def plot_q_values(steps, evals, out_dir):
    """Q(x_k), V_k, V_{k+1}, Advantage."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Q-Values & Advantage", fontsize=14, fontweight="bold")

    for ax, key, title, color in [
        (axes[0, 0], "q_xk", "Q(x_k)", "C0"),
        (axes[0, 1], "V_k", "Baseline V_k", "C1"),
        (axes[1, 0], "V_kp1", "Next-State V_{k+1}", "C2"),
        (axes[1, 1], "A_k", "Advantage A_k", "C3"),
    ]:
        xs, ys = get(steps, key)
        if xs:
            ax.plot(xs, ys, alpha=0.4, color=color, linewidth=0.5)
            sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
            ax.plot(sx, sy, color=color, linewidth=1.5)
            if key == "A_k":
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("global step")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "04_q_values.png", dpi=150)
    plt.close(fig)


def plot_gradient_norms(steps, evals, out_dir):
    """LM and RL gradient norms (total and per-component)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gradient Norms", fontsize=14, fontweight="bold")

    # LM total grad norm
    ax = axes[0, 0]
    xs, ys = get(steps, "lm_grad_total")
    if xs:
        ax.plot(xs, ys, alpha=0.4, color="C0", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
        ax.plot(sx, sy, color="C0", linewidth=1.5)
    ax.set_title("LM Gradient Norm (total)")
    ax.set_xlabel("global step")
    ax.set_yscale("log")

    # LM per-component
    ax = axes[0, 1]
    for key, label, color in [
        ("lm_grad_norm_backbone", "backbone", "C0"),
        ("lm_grad_norm_W_mu", "W_mu", "C1"),
        ("lm_grad_norm_W_gamma", "W_gamma", "C2"),
        ("lm_grad_norm_W_Q", "W_Q", "C3"),
    ]:
        xs, ys = get(steps, key)
        if xs and any(y > 0 for y in ys):
            ax.plot(xs, ys, alpha=0.6, linewidth=1, color=color, label=label)
    ax.set_title("LM Gradient Norms (per head)")
    ax.set_xlabel("global step")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # RL total grad norm
    ax = axes[1, 0]
    xs, ys = get(steps, "rl_grad_total")
    if xs:
        ax.plot(xs, ys, alpha=0.4, color="C4", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
        ax.plot(sx, sy, color="C4", linewidth=1.5)
    ax.set_title("RL Gradient Norm (total)")
    ax.set_xlabel("global step")
    ax.set_yscale("log")

    # RL per-component
    ax = axes[1, 1]
    for key, label, color in [
        ("rl_grad_norm_backbone", "backbone", "C0"),
        ("rl_grad_norm_W_mu", "W_mu", "C1"),
        ("rl_grad_norm_W_gamma", "W_gamma", "C2"),
        ("rl_grad_norm_W_Q", "W_Q", "C3"),
    ]:
        xs, ys = get(steps, key)
        if xs and any(y > 0 for y in ys):
            ax.plot(xs, ys, alpha=0.6, linewidth=1, color=color, label=label)
    ax.set_title("RL Gradient Norms (per head)")
    ax.set_xlabel("global step")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "05_gradient_norms.png", dpi=150)
    plt.close(fig)


def plot_weight_norms(steps, evals, out_dir):
    """Weight norms of projection heads over time."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle("Projection Head Weight Norms", fontsize=14, fontweight="bold")

    # Use step entries (more frequent)
    for key, label, color in [
        ("wnorm_W_mu", "W_mu", "C0"),
        ("wnorm_W_gamma", "W_gamma", "C1"),
        ("wnorm_W_Q", "W_Q", "C2"),
    ]:
        xs, ys = get(steps, key)
        if xs:
            ax.plot(xs, ys, linewidth=1.5, color=color, label=label)
    ax.set_xlabel("global step")
    ax.set_ylabel("Frobenius norm")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "06_weight_norms.png", dpi=150)
    plt.close(fig)


def plot_policy_stats(steps, evals, out_dir):
    """Policy entropy, effective N, sigma2 statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Policy Distribution Diagnostics", fontsize=14, fontweight="bold")

    # Entropy
    ax = axes[0, 0]
    xs, ys = get(steps, "pi_entropy")
    if xs:
        ax.plot(xs, ys, color="C0", linewidth=1)
    ax.set_title("Policy Entropy H(pi)")
    ax.set_xlabel("global step")

    # Effective N
    ax = axes[0, 1]
    xs, ys = get(steps, "pi_effective_n")
    if xs:
        ax.plot(xs, ys, color="C1", linewidth=1)
    ax.set_title("Effective N = exp(H)")
    ax.set_xlabel("global step")

    # Max/min prob
    ax = axes[1, 0]
    xs_max, ys_max = get(steps, "pi_max_prob")
    xs_min, ys_min = get(steps, "pi_min_prob")
    if xs_max:
        ax.plot(xs_max, ys_max, color="C3", linewidth=1, label="max")
    if xs_min:
        ax.plot(xs_min, ys_min, color="C4", linewidth=1, label="min")
    ax.set_title("Policy Prob Range")
    ax.set_xlabel("global step")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # Sigma2 stats (only present when fix_sigma=False)
    ax = axes[1, 1]
    xs_mean, ys_mean = get(steps, "sigma2_mean")
    if xs_mean:
        xs_min2, ys_min2 = get(steps, "sigma2_min")
        xs_max2, ys_max2 = get(steps, "sigma2_max")
        ax.plot(xs_mean, ys_mean, color="C0", linewidth=1.5, label="mean")
        if xs_min2:
            ax.plot(xs_min2, ys_min2, color="C1", linewidth=0.8, label="min", alpha=0.7)
        if xs_max2:
            ax.plot(xs_max2, ys_max2, color="C2", linewidth=0.8, label="max", alpha=0.7)
        ax.set_title("sigma^2 Statistics")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "sigma^2 fixed at 1.0\n(--fix_sigma)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("sigma^2 (fixed)")
    ax.set_xlabel("global step")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "07_policy_stats.png", dpi=150)
    plt.close(fig)


def plot_timing(steps, evals, out_dir):
    """Step time breakdown and GPU memory."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Timing Breakdown", fontsize=14, fontweight="bold")

    # Total step time
    ax = axes[0, 0]
    xs, ys = get(steps, "step_time")
    if xs:
        ax.plot(xs, ys, alpha=0.3, color="C0", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
        ax.plot(sx, sy, color="C0", linewidth=1.5)
    ax.set_title("Total Step Time")
    ax.set_xlabel("global step")
    ax.set_ylabel("seconds")

    # Sub-step breakdown (stacked area)
    ax = axes[0, 1]
    timing_keys = [
        ("t_select", "Selection", "C0"),
        ("t_lm_update", "LM Update", "C1"),
        ("t_reward", "Reward (heldout)", "C2"),
        ("t_rl_total", "RL Update", "C3"),
    ]
    for key, label, color in timing_keys:
        xs, ys = get(steps, key)
        if xs:
            sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
            ax.plot(sx, sy, linewidth=1.5, color=color, label=label)
    ax.set_title("Step Time Breakdown")
    ax.set_xlabel("global step")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)

    # RL sub-timings
    ax = axes[1, 0]
    rl_timing_keys = [
        ("t_q_xk", "Q(x_k) fwd", "C0"),
        ("t_V_kp1", "V_{k+1} est.", "C1"),
        ("t_V_k", "V_k est.", "C2"),
        ("t_pg_step", "PG + backward", "C3"),
    ]
    for key, label, color in rl_timing_keys:
        xs, ys = get(steps, key)
        if xs:
            sx, sy = smooth_xy(xs, ys, min(20, max(1, len(ys)//5)))
            ax.plot(sx, sy, linewidth=1.5, color=color, label=label)
    ax.set_title("RL Update Sub-timings")
    ax.set_xlabel("global step")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)

    # GPU memory
    ax = axes[1, 1]
    xs, ys = get(steps, "gpu_mem_allocated_mb")
    if xs:
        ax.plot(xs, ys, color="C1", linewidth=1, label="allocated")
    xs2, ys2 = get(steps, "gpu_mem_peak_mb")
    if xs2:
        ax.plot(xs2, ys2, color="C3", linewidth=1, label="peak", alpha=0.7)
    ax.set_title("GPU Memory (MB)")
    ax.set_xlabel("global step")
    ax.set_ylabel("MB")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "08_timing.png", dpi=150)
    plt.close(fig)


def plot_reward_distribution(steps, evals, out_dir):
    """Histogram of rewards and reward over outer steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reward Analysis", fontsize=14, fontweight="bold")

    # Histogram
    ax = axes[0]
    _, ys = get(steps, "r_k")
    if ys:
        ax.hist(ys, bins=min(50, max(10, len(ys)//5)), color="C0", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        frac_pos = sum(1 for y in ys if y > 0) / len(ys)
        ax.set_title(f"Reward Distribution ({frac_pos:.0%} positive)")
    ax.set_xlabel("r_k")
    ax.set_ylabel("count")

    # Mean reward per outer step
    ax = axes[1]
    outer_rewards = {}
    for e in steps:
        if "r_k" in e and "outer_step" in e:
            t = e["outer_step"]
            outer_rewards.setdefault(t, []).append(e["r_k"])
    if outer_rewards:
        ts = sorted(outer_rewards.keys())
        means = [np.mean(outer_rewards[t]) for t in ts]
        ax.bar(ts, means, color="C2", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="red", linestyle="--", linewidth=0.5)
    ax.set_title("Mean Reward per Outer Step")
    ax.set_xlabel("outer step")
    ax.set_ylabel("mean r_k")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "09_reward_analysis.png", dpi=150)
    plt.close(fig)


def plot_grad_norms_detail(steps, evals, out_dir):
    """Detailed gradient norms: LM backbone vs RL per-component."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Gradient Norms Detail", fontsize=14, fontweight="bold")

    win = lambda ys: min(30, max(1, len(ys)//10))

    # --- Row 0: LM vs RL total, and LM backbone ---
    # LM total
    ax = axes[0, 0]
    xs, ys = get(steps, "lm_grad_total")
    if xs:
        ax.plot(xs, ys, alpha=0.2, color="C0", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, win(ys))
        ax.plot(sx, sy, color="C0", linewidth=1.5)
    ax.set_title("LM Grad Total")
    ax.set_xlabel("global step")
    ax.set_yscale("log")

    # RL total
    ax = axes[0, 1]
    xs, ys = get(steps, "rl_grad_total")
    if xs:
        ax.plot(xs, ys, alpha=0.2, color="C3", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, win(ys))
        ax.plot(sx, sy, color="C3", linewidth=1.5)
    ax.set_title("RL Grad Total")
    ax.set_xlabel("global step")
    ax.set_yscale("log")

    # LM backbone (only component with nonzero LM grads)
    ax = axes[0, 2]
    xs, ys = get(steps, "lm_grad_norm_backbone")
    if xs and any(y > 0 for y in ys):
        ax.plot(xs, ys, alpha=0.2, color="C0", linewidth=0.5)
        sx, sy = smooth_xy(xs, ys, win(ys))
        ax.plot(sx, sy, color="C0", linewidth=1.5)
    ax.set_title("LM Grad: Backbone\n(W_mu, W_gamma, W_Q get zero LM grad)")
    ax.set_xlabel("global step")
    ax.set_yscale("log")

    # --- Row 1: RL grads per component ---
    rl_components = [
        ("rl_grad_norm_backbone", "RL Grad: Backbone", "C4"),
        ("rl_grad_norm_W_mu", "RL Grad: W_mu (policy mean)", "C1"),
        ("rl_grad_norm_W_Q", "RL Grad: W_Q (Q-head)", "C2"),
    ]
    for i, (key, title, color) in enumerate(rl_components):
        ax = axes[1, i]
        xs, ys = get(steps, key)
        if xs and any(y > 0 for y in ys):
            ax.plot(xs, ys, alpha=0.2, color=color, linewidth=0.5)
            sx, sy = smooth_xy(xs, ys, win(ys))
            ax.plot(sx, sy, color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("global step")
        ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "12_grad_norms_detail.png", dpi=150)
    plt.close(fig)


def plot_outer_step_timing(steps, evals, out_dir):
    """Outer-step level timing: embedding, data, eval."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Outer Step Timing", fontsize=14, fontweight="bold")

    for ax, key, title, color in [
        (axes[0], "t_embed", "Embedding Time", "C0"),
        (axes[1], "t_eval", "Eval Time", "C1"),
        (axes[2], "outer_time", "Total Outer Step Time", "C2"),
    ]:
        xs, ys = get_eval(evals, key)
        if xs:
            ax.plot(xs, ys, "o-", color=color, markersize=4, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("total examples")
        ax.set_ylabel("seconds")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "13_outer_timing.png", dpi=150)
    plt.close(fig)


def plot_selected_data(steps, evals, out_dir):
    """Visualize which candidates were selected and their text snippets."""
    # Only steps with selected_text
    sel = [e for e in steps if "selected_text" in e and "selected_idx" in e]
    if not sel:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Selected Data Analysis", fontsize=14, fontweight="bold")

    # --- Panel 1: Selected index over time ---
    ax = axes[0, 0]
    xs = [e.get("global_step", i) for i, e in enumerate(sel)]
    idxs = [e["selected_idx"] for e in sel]
    ax.scatter(xs, idxs, s=6, alpha=0.6, c="C0")
    ax.set_title("Selected Candidate Index")
    ax.set_xlabel("global step")
    ax.set_ylabel("index in S_t")

    # --- Panel 2: Index histogram (are we picking diverse candidates?) ---
    ax = axes[0, 1]
    ax.hist(idxs, bins=min(50, max(10, max(idxs)+1)), color="C1", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    n_unique = len(set(idxs))
    ax.set_title(f"Selection Frequency ({n_unique} unique indices)")
    ax.set_xlabel("candidate index")
    ax.set_ylabel("times selected")

    # --- Panel 3: LM loss of selected example vs reward ---
    ax = axes[1, 0]
    lm_losses = [e.get("lm_loss", None) for e in sel]
    rewards = [e.get("r_k", None) for e in sel]
    valid = [(l, r) for l, r in zip(lm_losses, rewards) if l is not None and r is not None]
    if valid:
        ls, rs = zip(*valid)
        ax.scatter(ls, rs, s=8, alpha=0.5, c="C2")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Reward vs LM Loss of Selected Example")
    ax.set_xlabel("lm_loss (selected x_k)")
    ax.set_ylabel("reward r_k")

    # --- Panel 4: Text snippets table (last N selections) ---
    ax = axes[1, 1]
    ax.axis("off")
    n_show = 15
    recent = sel[-n_show:]
    lines = []
    for e in recent:
        gs = e.get("global_step", "?")
        idx = e["selected_idx"]
        rk = e.get("r_k", 0)
        txt = e.get("selected_text", "")[:80]
        # Strip non-ASCII chars (e.g. CJK/Hangul) that monospace fonts can't render
        txt = txt.encode("ascii", errors="replace").decode("ascii")
        # Escape any problematic chars for matplotlib
        txt = txt.replace("$", "\\$").replace("_", "\\_")
        rk_str = f"{rk:+.4f}"
        lines.append(f"[{gs:>4d}] idx={idx:<3d} r={rk_str}  {txt}")
    text_block = "\n".join(lines)
    ax.text(0.02, 0.98, text_block, transform=ax.transAxes,
            fontsize=7, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.set_title(f"Recent Selections (last {n_show})", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "10_selected_data.png", dpi=150)
    plt.close(fig)


def plot_selection_per_outer(steps, evals, out_dir):
    """Per-outer-step selection diversity and reward breakdown."""
    sel = [e for e in steps if "selected_idx" in e and "outer_step" in e]
    if not sel:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Selection Diversity per Outer Step", fontsize=14, fontweight="bold")

    # Group by outer step
    by_outer = {}
    for e in sel:
        t = e["outer_step"]
        by_outer.setdefault(t, []).append(e)

    ts = sorted(by_outer.keys())

    # --- Unique indices per outer step ---
    ax = axes[0]
    n_unique = [len(set(e["selected_idx"] for e in by_outer[t])) for t in ts]
    n_total = [len(by_outer[t]) for t in ts]
    ax.bar(ts, n_unique, color="C0", alpha=0.7, label="unique")
    ax.plot(ts, n_total, "r--", linewidth=1, label="total")
    ax.set_title("Unique Candidates Selected")
    ax.set_xlabel("outer step")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    # --- Fraction of positive rewards ---
    ax = axes[1]
    frac_pos = []
    for t in ts:
        rs = [e.get("r_k", 0) for e in by_outer[t]]
        frac_pos.append(sum(1 for r in rs if r > 0) / max(len(rs), 1))
    ax.bar(ts, frac_pos, color="C2", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Fraction of Positive Rewards")
    ax.set_xlabel("outer step")
    ax.set_ylabel("fraction r_k > 0")
    ax.set_ylim(0, 1)

    # --- Mean LM loss of selected examples ---
    ax = axes[2]
    mean_lm = [np.mean([e.get("lm_loss", 0) for e in by_outer[t]]) for t in ts]
    ax.bar(ts, mean_lm, color="C4", alpha=0.7)
    ax.set_title("Mean LM Loss of Selected Examples")
    ax.set_xlabel("outer step")
    ax.set_ylabel("mean lm_loss")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "11_selection_diversity.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

ALL_PLOTTERS = [
    plot_core_metrics,
    plot_eval_perplexity,
    plot_rl_losses,
    plot_q_values,
    plot_gradient_norms,
    plot_weight_norms,
    plot_policy_stats,
    plot_timing,
    plot_reward_distribution,
    plot_selected_data,
    plot_selection_per_outer,
    plot_grad_norms_detail,
    plot_outer_step_timing,
]


def generate_all_plots(log_path: str, out_dir: Path):
    """Read log and generate all plots."""
    steps, evals = load_log(log_path)
    if not steps and not evals:
        print(f"  No data yet in {log_path}")
        return
    print(f"  Loaded {len(steps)} step entries, {len(evals)} eval entries")
    out_dir.mkdir(parents=True, exist_ok=True)
    for plotter in ALL_PLOTTERS:
        try:
            plotter(steps, evals, out_dir)
        except Exception as e:
            print(f"  Warning: {plotter.__name__} failed: {e}")
    print(f"  Plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument("--log_path", type=str, default="checkpoints/log.jsonl")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between plot updates")
    parser.add_argument("--once", action="store_true", help="Generate plots once and exit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_path = args.log_path

    if args.once:
        if not Path(log_path).exists():
            print(f"Log file not found: {log_path}")
            sys.exit(1)
        print(f"Generating plots from {log_path} ...")
        generate_all_plots(log_path, out_dir)
        return

    print(f"Watching {log_path} every {args.interval}s  (plots -> {out_dir}/)")
    print("Edit this script and re-run to change plots mid-training.")
    print("Press Ctrl+C to stop.\n")

    while True:
        if Path(log_path).exists():
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Updating plots ...")
            try:
                generate_all_plots(log_path, out_dir)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Waiting for {log_path} to appear ...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

