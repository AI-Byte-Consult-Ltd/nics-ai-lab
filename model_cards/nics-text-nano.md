# Model Card: `nics-text-nano`

## Summary

A tiny decoder-only Transformer language model, trained entirely from
random initialisation on a single public-domain text corpus. This is a
**pipeline-validation run**, not a production model: its purpose was to
prove the from-scratch training pipeline (tokenizer, model, checkpointing,
telemetry, export) works end to end on real data, with real, verifiable
provenance. It is not intended for deployment, generation quality is not
evaluated beyond loss, and it should not be presented anywhere as a
capable assistant model.

## Provenance

| Field | Value |
|---|---|
| Model ID | `nics-text-nano` |
| Experiment ID | `exp-0001` |
| Architecture | Decoder-only Transformer (RMSNorm, causal self-attention via `F.scaled_dot_product_attention`, SwiGLU FFN, tied input/output embeddings) -- `src/nics/models/text/model.py` |
| Parameters | 868,992 |
| Seed | 42 |
| Pretrained weights used | **None** -- random initialisation only (see `docs/architecture.md` Section 2) |
| Dataset | `tinyshakespeare-v1` (see `docs/dataset-register.md`), public domain |
| Tokenizer | Byte-level BPE, trained from scratch on the same corpus, vocab size 512 (`scripts/train_tokenizer.py`) |
| Config | `configs/text/nano.yaml` |
| Hardware | CPU only |

## Training

- 519,486 train tokens / 57,721 val tokens (90/10 split of the tokenized
  corpus).
- `d_model=128`, `n_layers=4`, `n_heads=4`, `max_seq_len=128`.
- AdamW, `lr=3e-4` with 50-step linear warmup + cosine decay, weight decay
  0.01, grad clip 1.0, batch size 32, 500 steps.

### Loss

| Step | Train loss | Val loss |
|---|---|---|
| 0 (random init, first eval) | 6.2509 | 6.2346 |
| 499 (final) | 3.9141 | 3.9879 |

Random-init sanity check: for a 512-token vocabulary with no training,
expected loss is `ln(512) ≈ 6.24` -- the observed step-0 loss (6.2509 /
6.2346) matches this closely, which is the evidence that the model truly
started from random weights and not from any prior checkpoint.

## Artifacts (not tracked in git -- see `docs/model-ownership.md`)

- `checkpoint-step-000000/trainer-state.pt` -- proof-of-random-init
  checkpoint.
- `checkpoints/step-{100,200,300,400,499}/trainer-state.pt` -- periodic
  resume checkpoints.
- `release/model.safetensors` -- final exported weights.
  SHA-256: `c651d4233eeb5d4c45ac583ce237ff5bf6d353e35d396679de53ae9e4d556fa8`.
- `release/training-manifest.json`, `release/sha256sums.txt`.
- `events.jsonl`, `manifest.json` -- tracked in git, at
  `experiments/text/nics-text-nano/exp-0001/`.

## Intended use

Internal pipeline validation and as a reference point for scaling up to
`nics-text-alpha`. Not fit for: production text generation, any
customer-facing feature, benchmarking against real language models.

## Known limitations

- Tiny scale (869K params) trained for only 500 steps on ~520K tokens of a
  single author's work -- expected output is repetitive, mostly
  incoherent English-like text, not fluent language.
- No held-out benchmark evaluation was run (only train/val loss on splits
  of the same corpus).
- No safety, bias, or alignment evaluation has been performed -- not
  applicable at this scale/purpose, but must not be skipped for
  `nics-text-alpha` or any customer-facing model.
- Single dataset, single domain (Early Modern English drama/poetry) --
  vocabulary and style are extremely narrow.
