# NICS AI Lab

**Version 0.11.0** -- see [`CHANGELOG.md`](CHANGELOG.md) for release history.
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
| `nics-text-nano` | Archived pipeline check (Aug 2026) | ~869K | Not a NICS Language model; no longer trainable |
| `nics-text-micro` | Prototype: code and tests ready, not trained yet | ~23.6M | NICS Language, first working size; own tokenizer for all 12 site languages |
| `nics-text-alpha` | Plan: config ready, starts after micro passes its checks | ~50.3M | NICS Language, largest size realistic on the laptop CPU |
| `nics-vision-nano` | First run complete | ~34K | CPU-trained, `fashion-mnist-v1`, 87.1% val accuracy, pipeline scale |
| `nics-trading-nano` | Prototype: self-learning pipeline ready, not trained on real data yet | ~167K | NICS Trading; trains on all our trading data (`nics-market-v2`, own trades) plus public D1/W1 history, on the laptop |
| `nics-vision-alpha` | Planned | -- | |

Hardware: CPU only (no discrete GPU on the current training machine) --
see the hardware report referenced in `docs/architecture.md`.

Model families (names decided 2026-11-18): **NICS Language** (`nics-text-*`),
**NICS Vision** (`nics-vision-*`), **NICS Trading** (`nics-trading-*`).
Status words: implemented / prototype / experiment / plan -- nothing is
called ready before it has been checked.

## NICS Trading quickstart (laptop, CPU)

Step-by-step guide in Russian: [`docs/laptop-guide-ru.md`](docs/laptop-guide-ru.md).

```bat
scripts\windows\setup.bat                      :: venv + dependencies + self-check
setx NICS_DATA_URL https://n8n.aibyteconsult.com/webhook/nics-lab-market-export
setx NICS_DATA_KEY <key from n8n "[41 LAB] Market Data Export">
.\venv\Scripts\python.exe scripts\self_learn.py --from-scratch :: first cycle: download everything, train, crown a champion
scripts\windows\install_tasks.bat              :: nightly self-learning 03:30 + hourly shadow scoring
```

The self-learning cycle (`scripts/self_learn.py`) downloads fresh data, re-scores the
champion on it, trains a warm-started challenger (and weekly one from random weights),
and promotes a challenger only if it beats the champion on the same new validation data
(the test split is never used for the choice, only reported).
`scripts/score_live.py` scores new scanner signals in the shadow; the next cycle checks
those scores against the outcomes that followed (forward test). Nothing trades.

Building blocks: `fetch_market_data.py` (all our data), `fetch_public_history.py`
(public D1/W1), `fetch_mt5_history.py` (our MT5 trades), `train_trading.py`,
`eval_trading.py`; methodologies in `src/nics/market/methods.py`, indicators and
patterns in `src/nics/market/indicators.py`. Self-checks on synthetic data:
`tests/test_trading_pipeline.py`, `tests/test_self_learning.py`.

## Repository layout

```
configs/          per-model training configs (YAML)
data_registry/     dataset provenance (NOT the raw data itself)
docs/              architecture, ownership, dataset governance
model_cards/       per-model capability/limitation summaries
requirements/      pinned dependencies (base + CPU torch)
scripts/           entry points: prepare data, train tokenizer, train models, benchmark
src/nics/          library code: models, training loop, checkpointing, telemetry
```

## NICS Language quickstart (laptop, CPU)

Everything is ours: the tokenizer algorithm (`src/nics/text/tokenizer.py`, standard
library only), the architecture, the training loop. No pretrained weights, no pretrained
tokenizer, no third-party tokenization library.

```bat
scripts\windows\bench_language.bat                 :: how fast this laptop trains and answers (2 minutes, no data)
venv\Scripts\python.exe scripts\prepare_language_corpus.py --list   :: which sources passed the licence check
venv\Scripts\python.exe scripts\prepare_language_corpus.py          :: build the cleaned corpus
venv\Scripts\python.exe scripts\train_tokenizer.py --vocab-size 24576
venv\Scripts\python.exe scripts\encode_corpus.py --tokenizer data\processed\language\tokenizer-24k.json
venv\Scripts\python.exe scripts\train_text.py --config configs\text\micro.yaml --max-minutes 290
```

A corpus source is used only when `data_registry/language_sources.json` states its
licence, allows commercial use and carries the date the licence was read; otherwise the
pipeline refuses it. Validation documents are split off by document and never trained on.

Every run produces, under `experiments/text/<model_id>/<experiment-id>/`:
- `checkpoint-step-000000/` -- proof of random initialisation
- `checkpoints/step-XXXXXX/` -- periodic resume checkpoints (not tracked in git)
- `events.jsonl` -- structured telemetry (same schema the website API reads)
- `manifest.json` -- seed, git commit, config, corpus and tokenizer checksums, parameter count
- `release/` -- final `model.safetensors` + checksums (not tracked in git)

Token metering and credit prices (proposed, nothing is charged): `src/nics/billing/metering.py`.

## Versioning

This repository follows [Semantic Versioning](https://semver.org/); the
current version lives in [`VERSION`](VERSION) and every change is logged in
[`CHANGELOG.md`](CHANGELOG.md).

## License

**Proprietary commercial license of AI Byte Consult Ltd** -- see [`LICENSE`](LICENSE).
Not open source. Free use is prohibited: any use, copying, training, hosting or
distribution requires a signed paid commercial license from AI Byte Consult Ltd
(info@aibyteconsult.com). Public visibility of the code grants no rights.
Model weights, checkpoints and datasets are governed separately -- see
[`docs/model-ownership.md`](docs/model-ownership.md) and
[`docs/dataset-register.md`](docs/dataset-register.md).
