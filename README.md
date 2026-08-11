# Curriculum Selection via Reinforcement Learning

PyTorch implementation of the algorithm described in `project.tex`: a language model that learns to select its own training data via reinforcement learning.

## Overview

Instead of training on randomly sampled data, the model:
1. Uses a learned policy (Gaussian in embedding space) to select training examples
2. Receives reward based on how much each selection improves held-out performance
3. Updates via actor-critic RL (Q-learning + policy gradient)

## Architecture

- **Backbone**: GPT-2 (small, 124M parameters)
- **Embedding model**: OpenAI `text-embedding-3-small` (1536-dim) for encoding candidates
- **Policy heads**: `W_μ`, `W_γ` project from transformer hidden states to embedding space
- **Q-head**: `W_Q` projects last-token activations to scalar Q-values

## Requirements

```bash
pip install -r requirements.txt
```

You need:
- `OPENAI_API_KEY` environment variable (for embeddings)
- CUDA-capable GPU (tested on RTX 3070, 8GB VRAM)

## Usage

```bash
python train.py
```

Configuration is in the `Config` dataclass at the top of `train.py`. Key hyperparameters:
- `num_candidates`: 90 (|S_t|, training candidate set size)
- `num_heldout`: 10 (|D_t|, held-out evaluation set size)
- `outer_steps`: 50 (number of data refresh cycles)
- `inner_steps`: 20 (RL updates per outer step)
- `gamma`: 0.99 (discount factor)
- `lr`: 1e-4 (shared learning rate)
- `M`: 8 (samples for V estimation)

## Output

- **Logs**: `checkpoints/log.jsonl` (one JSON object per inner step)
- **Checkpoints**: `checkpoints/model_outer_XXXX.pt` (saved after each outer step)
- **Embedding cache**: `embedding_cache.json` (avoids redundant OpenAI API calls)

## Algorithm Details

See `project.tex` for the full specification. Key points:

1. **Outer loop** (every ~1000 data points): refresh S_t (candidates) and D_t (held-out)
2. **Inner loop** (K steps per outer):
   - Select x_k from S_t via policy π_θ(x) ∝ N(φ(x); μ_k, σ_k²)
   - LM update: θ_{k+1} = θ_k + α ∇ log p_θ(x_k)
   - Reward: r_k = mean log p_{θ_{k+1}}(D_t) - mean log p_θ_k(D_t)
   - Q-update: minimize (Q_θ(x_k) - [r_k + γ V_{k+1}])²
   - Policy update: advantage-weighted gradient A_k ∇ log π_θ(x_k)

## Notes

- The policy is a Gaussian in the embedding space of φ (OpenAI embeddings), normalized over S_t
- We keep both θ_k and θ_{k+1} in memory to compute the RL gradients correctly
- Numerical stability: log σ² is clamped to [-10, 10] to avoid NaN in the policy distribution
- Data source: OpenWebText (streaming from HuggingFace)

## Citation

```
@misc{viteri2026curriculum,
  title={Learning to Learn: Curriculum Selection via Reinforcement Learning for Sample-Efficient Language Models},
  author={Viteri, Scott},
  year={2026}
}
```

