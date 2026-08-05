# Changelog

All notable changes to this repository are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

- **MAJOR** -- breaking changes to checkpoint format, telemetry schema, or
  repository layout that downstream consumers (the website API, R2 sync)
  depend on.
- **MINOR** -- a new capability: a new model, a new dataset, a new pipeline
  stage.
- **PATCH** -- fixes, docs, tooling that don't add or break a capability.

Every merged change bumps [`VERSION`](VERSION) and gets an entry here --
no silent, unversioned changes.

## [0.1.0] - 2026-08-05

### Added
- Repository scaffold: proprietary `LICENSE`, pinned `requirements/base.txt`
  and `requirements/cpu.txt`, `.gitignore` policy for weights/datasets.
- `nics-text-nano`: decoder-only Transformer architecture from scratch
  (RMSNorm, causal self-attention, SwiGLU, tied embeddings) --
  `src/nics/models/text/model.py`.
- `nics-vision-nano`: compact CNN classifier architecture from scratch --
  `src/nics/models/vision/model.py` (not yet trained).
- From-scratch byte-level BPE tokenizer training script
  (`scripts/train_tokenizer.py`).
- Dataset provenance registry (`data_registry/datasets.json`) with the
  first registered dataset, `tinyshakespeare-v1` (public domain).
- Full training pipeline (`scripts/train_text.py`): step-0 checkpoint
  (proof of random init), periodic resume checkpoints, JSONL telemetry,
  final `safetensors` export with checksums.
- Checkpoint/telemetry/export utilities (`src/nics/training/checkpoint.py`,
  `src/nics/telemetry/events.py`).
- First verified, from-scratch, end-to-end training run:
  `nics-text-nano` / `exp-0001` -- see
  [`model_cards/nics-text-nano.md`](model_cards/nics-text-nano.md).
- Governance docs: `docs/architecture.md`, `docs/model-ownership.md`,
  `docs/dataset-register.md`.
