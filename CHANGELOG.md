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

## [0.5.0] - 2026-11-19

### Added
- **Self-learning cycle** (`scripts/self_learn.py`, Windows tasks in
  `scripts/windows/`): nightly download of all our data, champion re-scored on
  the new data, warm-started challenger (weekly also from random weights),
  promotion only if the challenger beats the champion on the same new
  validation data (`self_learning.min_improvement`); the test split is
  report-only, never used for selection (review finding); logs, history,
  housekeeping.
- **Shadow scoring** (`scripts/score_live.py`) of new scanner signals by the
  champion, and a forward test of those scores against later outcomes
  (`scripts/eval_trading.py --predictions`). Nothing is sent or traded.
- **Indicators and candle patterns** as bar features
  (`src/nics/market/indicators.py`): RSI, EMA 20/50/200 distance and slope,
  MACD histogram, Bollinger position, stochastic, ADX, 20/100-bar
  channel distances; engulfing, hammer / shooting star, doji, inside and
  outside bars. All causal.
- **Trading methodologies** (`src/nics/market/methods.py`): range traded from
  both sides, 20-bar breakout, EMA pullback, level bounce, run on broker
  H1/H4 history; their setups are labelled from candles and the method is a
  feature, so the model learns which method works in which market.
- **How the scanner defines a signal** as setup features: confidence,
  opportunity and technical scores, TP2/TP3 reward/risk, strategy version,
  and the daily-range "room left" (ATR rule).
- Proprietary **commercial license** of AI Byte Consult Ltd; free use
  prohibited.
- Russian laptop guide `docs/laptop-guide-ru.md`.

### Fixed
- EA manual signals without an actual fill are no longer labelled as trades
  (review finding on 0.4.0).

## [0.4.0] - 2026-11-18

### Changed
- **NICS Trading now trains on all our trading data**, not only M15 candles
  (Alexander, 2026-11-18: "абсолютно на всех торговых данных, доступных и
  публичных, которые у нас есть, на результатах"):
  - every broker timeframe (1min, 5min, 15min, 1h, 4h) for every instrument
    in one model; timeframe and source are input features;
  - public D1/W1 history (Kraken, Yahoo) for years of context the broker
    store does not have (`--no-public` trains on our data only);
  - every trade setup with entry/SL/TP: all scanner signals (scheduled,
    on-request, scalper, strong and weak), shadow strategies, EA manual
    signals and our own closed MT5 trades; outcomes re-derived the same way
    for every source; metrics split by timeframe and by setup kind;
  - candle-derived TP1/SL labels are checked against the tracker's recorded
    lifecycle events (`label_agreement_with_tracker`).
- Split boundaries come from our broker data; public history before them
  is training data. A sample is kept only if its label is known before the
  next split starts.
- Windows are built lazily, so millions of candles fit in laptop memory.

### Added
- `scripts/fetch_public_history.py` (public D1/W1, source kept per row).
- `scripts/fetch_mt5_history.py` (our closed positions from a running MT5
  terminal, account number not stored).
- n8n `[41 LAB] Market Data Export` serves all our trading tables, paged.

### Fixed
- The bridge names timeframes `1min/5min/15min/1h/4h`, not `M15`; the
  0.3.0 fetch command would have found no candles.

## [0.3.0] - 2026-11-18

### Added
- **NICS Trading** (`nics-trading-nano`), prototype, not trained yet:
  market-sequence Transformer from random initialisation, reusing the
  NICS Language blocks (`src/nics/models/trading/model.py`). Two heads:
  where price is N bars later (down / flat / up, self-supervised from
  candles) and the probability a scanner signal reaches TP1 before SL.
- `src/nics/market/data.py`: scale-free candle features (ATR units),
  forward labels, signal outcomes re-derived from candles (a bar touching
  TP1 and SL counts as a loss), signal windows use only candles closed at
  signal time, and the time split keeps a sample only if its label ends
  before the next split starts (robust to weekend and missing-bar gaps).
- `scripts/fetch_market_data.py`: downloads a snapshot of our own data
  (Vantage MT5 candles + scanner signals) from the n8n lab export into
  `data/market/<date>/` with checksums. Data stays on the training machine.
- `scripts/train_trading.py`: train / val / test by time, early stopping on
  validation (overfitting guard), best-on-val weights, `metrics.json` with
  baselines (majority class, base rate) and mean R per trade after spread;
  the signal filter threshold is chosen on val and applied once to test.
- `tests/test_trading_pipeline.py`: end-to-end smoke test on synthetic data.
- Dataset `nics-market-v1` registered (own data, snapshots versioned by date).

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
