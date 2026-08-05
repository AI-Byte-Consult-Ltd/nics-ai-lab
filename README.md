# NICS AI Lab

**Version 0.1.0** -- see [`CHANGELOG.md`](CHANGELOG.md) for release history.
Every merged change bumps this version; there are no unversioned changes.

Independent model training infrastructure for **AI Byte Consult Ltd** / the
**NICS AI Ecosystem**.

This repository holds code only: model architectures, training pipelines,
configuration, telemetry and documentation. It does **not** hold model
weights or datasets -- see [`docs/model-ownership.md`](docs/model-ownership.md)
and [`docs/dataset-register.md`](docs/dataset-register.md) for where those
live and how they're licensed.

## Hard rule: no mock data, anywhere

Nothing in this repository or any system it feeds (including the public
website) may display invented or simulated training metrics. If a training
process is not currently running, the honest state is "offline" /
"training not started" -- never a fabricated number. See
[`docs/architecture.md`](docs/architecture.md) for the full policy.

## Hard rule: no pretrained foundation weights

`nics-text-*` and `nics-vision-*` models start from random weight
initialisation and are trained on licensed data from scratch. No GPT,
Llama, Qwen, BERT, CLIP, ViT or other pretrained checkpoint is loaded into
these models. See [`model_cards/`](model_cards/) for per-model provenance.

## Current status

| Model | Status | Parameters | Notes |
|---|---|---|---|
| `nics-text-nano` | First run complete | ~869K | CPU-trained, `tinyshakespeare-v1` dataset, proof-of-pipeline scale |
| `nics-vision-nano` | Architecture ready, not yet trained | ~34K (10-class config) | |
| `nics-text-alpha` | Planned | -- | |
| `nics-vision-alpha` | Planned | -- | |

Hardware: CPU only (no discrete GPU on the current training machine) --
see the hardware report referenced in `docs/architecture.md`.

## Repository layout

```
configs/          per-model training configs (YAML)
data_registry/     dataset provenance (NOT the raw data itself)
docs/              architecture, ownership, dataset governance
model_cards/       per-model capability/limitation summaries
requirements/      pinned dependencies (base + CPU torch)
scripts/           entry points: prepare data, train tokenizer, train models
src/nics/          library code: models, training loop, checkpointing, telemetry
```

## Quickstart (CPU)

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements/cpu.txt

python scripts/prepare_text_data.py
python scripts/train_tokenizer.py --vocab-size 512
python scripts/train_text.py --config configs/text/nano.yaml --experiment-id exp-0001
```

Every run produces, under `experiments/text/nics-text-nano/<experiment-id>/`:
- `checkpoint-step-000000/` -- proof of random initialisation
- `checkpoints/step-XXXXXX/` -- periodic resume checkpoints (not tracked in git)
- `events.jsonl` -- structured telemetry (same schema the website API reads)
- `manifest.json` -- seed, git commit, config, dataset, parameter count
- `release/` -- final `model.safetensors` + checksums (not tracked in git; upload to R2)

## Versioning

This repository follows [Semantic Versioning](https://semver.org/); the
current version lives in [`VERSION`](VERSION) and every change is logged in
[`CHANGELOG.md`](CHANGELOG.md).

## License

Proprietary, all rights reserved -- see [`LICENSE`](LICENSE). Not open source.
Model weights, checkpoints and datasets are governed separately -- see
[`docs/model-ownership.md`](docs/model-ownership.md) and
[`docs/dataset-register.md`](docs/dataset-register.md).
