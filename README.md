# Auto Researcher — MacBook Air M2 Fork

Personal fork of [autoresearch-macos](https://github.com/miolini/autoresearch-macos) adapted to run on my MacBook Air M2 with 8 GB unified memory.

## My Hardware

| Component | Spec |
|-----------|------|
| Device | MacBook Air (Mac14,15) |
| Chip | Apple M2 |
| Cores | 8 (4 performance + 4 efficiency) |
| Memory | 8 GB unified |
| Acceleration | Metal (MPS) |

## Changes I Made

### Dataset
Switched from the large multi-shard `climbmix-400b-shuffle` dataset to the single-file [TinyStories GPT-4 clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) dataset (~641 MB). Much more manageable on 8 GB RAM and trains a coherent model at small scale.

### Hyperparameters tuned for M2
- `MAX_SEQ_LEN`: 2048 → **256** (fits in MPS memory)
- `VOCAB_SIZE`: 8192 → **4096**
- `TOTAL_BATCH_SIZE`: 2^16 → **2^14** (~16K tokens/step)
- `DEVICE_BATCH_SIZE`: 16 → **1** (prevents MPS OOM)
- `EVAL_TOKENS`: 40 × 524288 → **2^17** (~131K tokens)

### Download simplification
Replaced the parallel multi-shard downloader (argparse + multiprocessing Pool) with a simple single-file download with progress output. No CLI flags needed — just run `uv run prepare.py`.

## Baseline Results

**Run 1 — Baseline (5 min, TinyStories, 7.3M params)**

| Metric | Value |
|--------|-------|
| val_bpb | 0.7981 |
| num_steps | 148 |
| total_tokens | 2.4M |
| peak tok/sec | ~10,800 |
| training_seconds | 301.5 |
| depth | 4 |
| n_embd | 256 |
| n_head | 2 |

Loss curve: 8.32 → 2.09 over 148 steps.

## Planned Experiments

Each experiment will be run for the standard 5-minute budget and val_bpb recorded.

- [ ] **SwiGLU** — replace standard FFN activation with SwiGLU (used in LLaMA/PaLM). Hypothesis: better loss per parameter.
- [ ] **RMSNorm** — replace LayerNorm with RMSNorm. Simpler, faster, and standard in modern LLMs.
- [ ] **Tied embeddings** — share weights between `wte` and `lm_head`. Reduces parameter count by ~1M, may regularize.
- [ ] **RoPE** — replace learned positional embeddings with Rotary Position Embeddings. Better length generalization.
- [ ] **Cosine schedule** — replace current LR schedule with a full cosine decay. Hypothesis: smoother convergence.

Results table will be updated as experiments run:

| Experiment | val_bpb | Δ vs baseline | Notes |
|------------|---------|---------------|-------|
| Baseline | 0.7981 | — | TinyStories, 4-layer, 256 embd |
| SwiGLU | | | |
| RMSNorm | | | |
| Tied embeddings | | | |
| RoPE | | | |
| Cosine schedule | | | |

## License

MIT
