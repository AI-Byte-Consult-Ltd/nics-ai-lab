# Model Card: `nics-vision-nano`

## Summary

A tiny convolutional image classifier, trained entirely from random
initialisation on a single, unambiguously-licensed image corpus. This is a
**pipeline-validation run**, not a production model: its purpose was to
prove the from-scratch vision training pipeline (data loading, model,
checkpointing, telemetry, export) works end to end on real data, with real,
verifiable provenance. It is not intended for deployment, and its accuracy
should not be presented anywhere as competitive with production image
classifiers.

## Provenance

| Field | Value |
|---|---|
| Model ID | `nics-vision-nano` |
| Experiment ID | `exp-0001` |
| Architecture | Compact CNN (3x `Conv-BatchNorm-ReLU-MaxPool` blocks, dropout, linear head) -- `src/nics/models/vision/model.py` |
| Parameters | 33,658 |
| Seed | 42 |
| Pretrained weights used | **None** -- random initialisation only (see `docs/architecture.md` Section 2) |
| Dataset | `fashion-mnist-v1` (see `docs/dataset-register.md`), MIT license |
| Input | 32x32, 1 channel (28x28 source images, zero-padded to 32x32) |
| Config | `configs/vision/nano.yaml` |
| Hardware | CPU only |

## Training

- 60,000 train images / 10,000 val images (Fashion-MNIST's own train/test
  split; no re-splitting of the training set).
- `base_channels=16`, `image_size=32`, `dropout=0.1`.
- AdamW, `lr=1e-3` with 50-step linear warmup + cosine decay, weight decay
  0.01, grad clip 1.0, batch size 64, 500 steps.

### Loss and accuracy

| Step | Train loss | Val loss | Val accuracy |
|---|---|---|---|
| 0 (random init, first eval) | 2.4690 | 2.3038 | 0.1313 |
| 499 (final) | 0.3157 | 0.3952 | 0.8711 |

Random-init sanity check: for a 10-class classifier with no training,
expected loss is `ln(10) ≈ 2.3026` and expected accuracy is `1/10 = 0.10`
-- the observed step-0 val loss (2.3038) and val accuracy (0.1313, close to
chance) match this closely, which is the evidence that the model truly
started from random weights and not from any prior checkpoint.

## Artifacts (not tracked in git -- see `docs/model-ownership.md`)

- `checkpoint-step-000000/trainer-state.pt` -- proof-of-random-init
  checkpoint.
- `checkpoints/step-{100,200,300,400,499}/trainer-state.pt` -- periodic
  resume checkpoints.
- `release/model.safetensors` -- final exported weights.
  SHA-256: `e4a12a6db147e9d21e7be299d0316d3b0f7889a20a989111f664acea741de8e3`.
- `release/training-manifest.json`, `release/sha256sums.txt`.
- `events.jsonl`, `manifest.json` -- tracked in git, at
  `experiments/vision/nics-vision-nano/exp-0001/`.

## Intended use

Internal pipeline validation and as a reference point for scaling up to
`nics-vision-alpha`. Not fit for: production image classification, any
customer-facing feature, benchmarking against real vision models.

## Known limitations

- Tiny scale (34K params) trained for only 500 steps -- 87.1% validation
  accuracy is well below what larger CNNs or ViTs reach on Fashion-MNIST
  (typically 92-95%+); this run optimises for pipeline correctness, not
  peak accuracy.
- Single dataset, single domain (Zalando product photography, 10 clothing
  categories, grayscale, low resolution) -- does not generalise to
  photographs taken in uncontrolled conditions, other object categories,
  or color images.
- No held-out benchmark beyond Fashion-MNIST's own test split.
- No safety, bias, or fairness evaluation has been performed -- not
  applicable at this scale/purpose, but must not be skipped for
  `nics-vision-alpha` or any customer-facing model.
