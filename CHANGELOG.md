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

## [0.2.0] - 2026-08-06

### Added
- `nics-vision-nano`: first trained checkpoint of the compact CNN
  classifier, trained from random initialisation on `fashion-mnist-v1` --
  see [`model_cards/nics-vision-nano.md`](model_cards/nics-vision-nano.md).
  Step-0 val loss (2.3038) and val accuracy (0.1313) match the `ln(10) ≈
  2.30` / chance-level sanity check for a 10-class classifier at random
  init; final val accuracy 0.8711 after 500 steps.
- `fashion-mnist-v1` registered in `data_registry/datasets.json` and
  `docs/dataset-register.md` -- selected over Imagenette/Imagewoof after
  checking license text directly (Imagenette/Imagewoof's Apache-2.0 covers
  code only, not an explicit commercial-use grant for the underlying
  ImageNet-derived images).
- Vision data/training scripts: `scripts/prepare_vision_data.py`,
  `scripts/train_vision.py`, `configs/vision/nano.yaml`.

## [0.1.1] - 2026-08-06

### Added
- Independent reproduction of `nics-text-nano` on the laptop (Windows,
  Intel Core Ultra 5 125H, CPU-only): `exp-0002-laptop`. Same seed (42),
  config (`configs/text/nano.yaml`), and dataset as `exp-0001`; confirms
  the from-scratch pipeline (tokenizer, model, checkpointing, telemetry,
  export) is reproducible on different hardware, not just the original
  sandbox. Step-0 loss (6.2509) again matches the `ln(512) ≈ 6.24`
  random-init sanity check; final loss matches `exp-0001` within
  floating-point noise.
- `experiments/text/nics-text-nano/exp-0002-laptop/manifest.json` and
  `events.jsonl` added as provenance (checkpoints and exported weights
  excluded per `docs/architecture.md` Section 3).

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
