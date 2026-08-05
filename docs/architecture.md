# Architecture &amp; Policy

This document is the source of truth for the rules every model, script and
integration in this repository must follow. If code and this document
disagree, the code is wrong.

## 1. No mock data, anywhere

Nothing in this repository, or any downstream system it feeds (the public
website's Live Training panel, the telemetry API, model cards), may display
invented, simulated, hardcoded-as-if-real, or "placeholder" training
metrics. Concretely:

- If a training run is not currently active, downstream consumers must show
  an honest "offline" / "training not started" state -- never a fabricated
  loss curve or progress bar.
- Every number that reaches the website comes from a real `events.jsonl`
  emitted by an actual training process (see Section 4).
- This rule replaces the previous website panels (`LiveControlPanel`,
  `NicsLiveDashboard`) that showed fabricated "live model training" data --
  those are to be treated as a known defect until wired to real telemetry
  (Stage 6-7), not as a template to imitate.

## 2. No pretrained foundation weights

`nics-text-*` and `nics-vision-*` models are initialised with random
weights (see each model's `_init_weights`) and trained from that point
only on data listed in [`docs/dataset-register.md`](dataset-register.md).
No GPT, Llama, Qwen, Mistral, BERT, CLIP, ViT, ResNet or any other
pretrained checkpoint is loaded into these models, at any point, by any
script in this repository.

Provenance evidence required for every training run:
- A `checkpoint-step-000000/` saved immediately after model construction,
  before any optimiser step -- proof of the exact random init.
- `manifest.json` recording the seed, git commit, and full config used.
- Loss at step 0 should be consistent with random-init entropy (roughly
  `ln(vocab_size)` for a from-scratch language model with an untrained
  softmax) -- a sanity check documented in each model card.

## 3. Repository vs. weight/dataset storage

This repository (`nics-ai-lab`) holds **code only**: architectures,
training/eval scripts, configs, telemetry plumbing, and documentation.

It does **not** hold:
- Raw or processed datasets (`.gitignore`: `data/`, `datasets/`).
- Trainer-state checkpoints or exported weights (`.gitignore`: `checkpoints/`,
  `weights/`, `artifacts/`, `*.pt`, `*.pth`, `*.ckpt`, `*.bin`,
  `*.safetensors`, `*.onnx`, `*.npz`).
- Run logs (`.gitignore`: `runs/`, `logs/`, `wandb/`, `tensorboard/`).

The `experiments/<domain>/<model_id>/<experiment_id>/` directory itself
**is** tracked (its `manifest.json` and `events.jsonl` are small, textual,
and are exactly the provenance record this document requires) -- only the
binary weight files inside `checkpoints/` and `release/` are excluded.

See [`docs/model-ownership.md`](model-ownership.md) for where the actual
weight files live and who is accountable for them, and
[`docs/dataset-register.md`](dataset-register.md) for dataset governance.

## 4. Checkpoint and telemetry formats

Two distinct artifact types (`src/nics/training/checkpoint.py`):

| File | Contents | Tracked in git? |
|---|---|---|
| `checkpoint-step-000000/trainer-state.pt` | Full resumable state at random init: model + optimizer + RNG | No (binary) -- directory yes, file no |
| `checkpoints/step-XXXXXX/trainer-state.pt` | Full resumable state at a given step | No |
| `release/model.safetensors` | Distribution weights only, safetensors format | No -- upload target is Cloudflare R2 |
| `release/training-manifest.json` | Copy of the run manifest at completion | No (lives alongside the weights it describes) |
| `release/sha256sums.txt` | Checksums for every file in `release/` | No |
| `manifest.json` | Seed, git commit, config, dataset, parameter count, timestamps | **Yes** |
| `events.jsonl` | Structured telemetry, one JSON object per line | **Yes** |

`events.jsonl` event types emitted today (`src/nics/telemetry/events.py`),
matching the schema the future website telemetry API will read:

- `training_metric` -- `model_id`, `experiment_id`, `step`, `train_loss`,
  `validation_loss`, `learning_rate`, `tokens_processed`,
  `tokens_per_second`, `gpu_utilisation_percent` (`null` on CPU-only
  hardware), `checkpoint_status`.
- `checkpoint_saved` -- `model_id`, `experiment_id`, `step`, `path`.
- `training_completed` -- `model_id`, `experiment_id`, `final_step`,
  `best_val_loss`.

Every event also carries `event_type` and a UTC ISO-8601 `timestamp`,
appended by `TelemetryWriter`.

## 5. Versioning policy

The repository has a single `VERSION` file (semantic versioning) and a
`CHANGELOG.md`. Every merged change bumps `VERSION` and adds a
`CHANGELOG.md` entry -- there is no unversioned or silent change. See
`CHANGELOG.md` for the MAJOR/MINOR/PATCH rules.

## 6. Hardware assumption

Training runs on CPU only today (no discrete GPU on either the sandbox or
the current Windows laptop -- confirmed via hardware audit: Intel Core
Ultra 5 125H, integrated graphics only). This is why:
- `nics-vision-*` uses a small CNN rather than a Vision Transformer.
- `nano` configs are deliberately tiny (hundreds of thousands of
  parameters, not millions+) -- proof-of-pipeline scale, not
  capability-scale.
- `alpha`-scale models (Section in `README.md`) will need either much
  longer CPU training time or a GPU upgrade; that tradeoff is not yet
  decided.
