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

## [0.13.0] - 2026-11-22

NICS Language: our own task examples for the first task (journal post about a demo trade).

### Added
- `src/nics/text/examples.py`: phrase bank written by us in ru, en, es, bg, pt and a
  composer of "facts in, text out" records. The model writes value slots, never a number,
  a ticker or the trader's note; `check()` validates a generated post by code (exact
  slots, no digits, only `<b>`/`<i>`, length) and `fill()` puts escaped real values in.
- `scripts/make_language_examples.py` and task records in `scripts/encode_corpus.py`
  (about 4 % of the training tokens), so the first model learns the task format together
  with general language. The night run writes them automatically.

## [0.12.0] - 2026-11-22

NICS Language: first training corpus and unattended nightly training on the laptop.

### Added
- Corpus sources with licences read on 2026-11-22: United Nations Parallel Corpus v1.0
  (ru, en, es; small samples of ar, zh) and Europarl v8 (bg, pt, es, en; small samples of
  de, fr, it, pl). New source kind `opus_mono` in `scripts/prepare_language_corpus.py`:
  the file is read as a stream and reading stops at the size limit, repeated and very
  short lines are dropped, nothing raw is kept on disk.
- `scripts/language_auto.py`: does whatever is missing (corpus, tokenizer, token files)
  and then trains nics-text-micro until a stop time; an interrupted night continues the
  next night. `scripts/language_night.py`: one training session with a report.
  `scripts\windows\language_train.bat` for a run by hand.
- The nightly NICS Trading cycle (`scripts/self_learn.py`) now gives the rest of the night
  to NICS Language (until 08:00 local time; only for cycles that end between 03:00 and
  07:00). `NICS_LANGUAGE_AUTO=0` turns it off, `NICS_LANGUAGE_UNTIL` moves the stop time.
- Every stage reports its status to the private dashboard (model `nics-language`).
- `scripts/encode_corpus.py --langs`: the tokenizer learns all languages that have text,
  the first model is trained on ru, en, es, bg, pt.

### Changed
- The corpus provenance file moved to `data/processed/language/datasets.json` (not
  tracked), so building the corpus on the laptop cannot block `git pull`.

### Known limits
- The first corpus is official and parliamentary language only; Project Gutenberg is
  still refused (licence page unreachable), Turkish has no source yet.

## [0.11.0] - 2026-11-22

NICS Language, stages 0 and 1 of the from-scratch plan. Nothing is trained yet and no
production system is changed.

### Added
- `src/nics/text/tokenizer.py`: our own byte-level BPE tokenizer (merge learning, encoder,
  decoder; standard library only). Lossless on any Unicode text, digits never merge,
  control tokens are produced only on request so user text cannot inject them. One
  vocabulary for all 12 site languages (ru, en, es, bg, pt, de, fr, it, pl, tr, ar, zh),
  value slots for prices and symbols so the model never writes a number itself.
- NICS Language model: rotary positions (`pos_encoding: rope`) and a key/value cache for
  generation (same logits as a full pass, checked by test), top-k / greedy sampling, stop token.
- `configs/text/micro.yaml` (23.6M parameters) and `configs/text/alpha.yaml` (50.3M).
- `scripts/prepare_language_corpus.py` + `data_registry/language_sources.json`: corpus
  builder with a licence gate (licence stated, commercial use allowed, date the licence was
  read). `scripts/encode_corpus.py`: token files with a by-document train/validation split.
- `scripts/bench_text.py` and `scripts\windows\bench_language.bat`: measured training and
  generation speed of micro and alpha on the machine it runs on.
- `src/nics/billing/metering.py`: token metering and credit pricing (1 credit = 0.01 EUR is
  the wallet unit; proposed prices per 1K tokens, dated, append-only). Computes only.
- Tests: `test_tokenizer.py`, `test_text_model.py`, `test_language_pipeline.py` (end to
  end on synthetic text, including stop and resume), `test_metering.py`.

### Changed
- `scripts/train_tokenizer.py` and `scripts/train_text.py` rewritten for the own tokenizer,
  memory-mapped token files, a time budget for nightly runs (`--max-minutes`) and resume.
  Validation loss is measured on fixed windows. Resume continues after the saved step.

### Removed
- Dependency on the third-party `tokenizers` library.
- `scripts/prepare_text_data.py` (Tiny Shakespeare download): NICS Language is not trained on
  it. `configs/text/nano.yaml` and the `nics-text-nano` card stay as an archived record.

## [0.10.1] - 2026-11-21

### Fixed
- `scripts/db_backup.py` no longer stops with "Set NICS_BACKUP_KEY ... first" when the key
  was set with `setx` in the same window or not set at all: it reads the value saved by
  `setx` from the Windows user environment, otherwise asks for it once with hidden input
  and saves it the same way for the nightly task. The server address defaults to the NICS
  n8n, so `NICS_DATA_URL` / `NICS_BACKUP_URL` are optional.

### Added
- After every real run `db_backup.py` reports the result (finished time, rows, tables, size,
  folder or the error) to the CEO desk through the existing lab metrics webhook
  (`model=nics-db`, `kind=db_backup`, key `NICS_DATA_KEY`). The desk shows it in the new
  block «Бэкап». No data and no keys are sent; a failed report never fails the backup.

## [0.10.0] - 2026-11-21

### Added
- `src/nics/market/registry.py`: machine-readable rules registry. It lists every rule the
  model learns from, every rule in shadow and the queue, with status, date and version,
  in neutral names only. After every nightly cycle it is pushed to the CEO desk as
  kind `rules` and shown there as a separate list. A test keeps it in sync with
  `methods.py` and `rules.py`.
- Queue extended with 11 neutral rule ideas from trading books. They are added one at a
  time through shadow.

## [0.9.3] - 2026-11-21

### Added
- `scripts/db_backup.py`: full backup of the NICS Trading production database to the
  laptop through the read-only n8n workflow "[00 OPS] Database Backup Export"
  (schema + every table as primary-key-ordered gzip JSONL, manifest with row counts,
  `.partial` until complete, keeps the last 14). Secret values of `app_secrets` are
  replaced by `__REDACTED__` on the server.
- `scripts/db_restore.py`: turns a backup folder into one SQL file for `psql`
  (schema -> data -> foreign keys/triggers -> sequences).
- Scheduled task "NICS DB backup" every night at 04:15 (`install_tasks.bat`).
- `tests/test_db_backup.py`: offline test against a fake server.

## [0.9.2] - 2026-11-21

### Added
- The model's opinion message now also gives its OWN plan. `scripts/opinion_live.py`
  scores a grid of variants (both directions; entry as in the signal or at market; stop
  0.75-2.5 ATR; take 1-3 R) and picks the one with the best expected R after spread.
  The message shows entry, stop, take, its chance, the expected R, and the expected R of
  the signal's own plan for comparison. Still shadow, still only to Alexander.
  `shadow/opinions_sent.csv` gains the plan columns (an older log is rewritten in place).

## [0.9.1] - 2026-11-21

### Added
- Second candidate rule in shadow: `rule_level_squeeze_breakout` (level methodology:
  squeeze into a level, breakout stop with luft, ATR reserve under 75 %, target 3 R).

### Changed
- No names of authors, schools, courses or companies anywhere: rule ids, descriptions,
  outputs, registry and changelog use neutral words only. `tests/test_rules_shadow.py`
  checks the rule outputs against a hashed list of such names.

## [0.9.0] - 2026-11-21

### Added
- **Trading-book rules, measured in shadow before the model learns them.**
  `src/nics/market/rules.py` holds candidate rules coded with our own parameters (no book
  text copied). Every night `scripts/rules_shadow.py` measures each one by itself:
  backtest on public history, our broker history before the day it was added, and forward
  from that day. Stop entries must actually fill, and a bar touching both stop and target
  counts as a loss. Results: `shadow/rules/<rule>.json`, `shadow/rules_last.txt`, and the
  CEO desk (same block as the direction test). The model does not train on a candidate
  until it is promoted after a 2-4 week forward review.
- First candidate: `rule_strong_trend_pullback` (pullback in a strong ADX trend).
- `docs/rules_registry.md`: every rule given to the model, with source idea, exact
  definition, date, status and result, so no rule is added twice.
- `tests/test_rules_shadow.py` (offline).

## [0.8.0] - 2026-11-21

### Added
- **Model opinion on every signal sent to users, for Alexander only.** New
  `scripts/opinion_live.py`, run every 2 minutes by the new scheduled task "NICS Trading
  opinion" (`scripts/windows/opinion.bat`). It fetches the last day of scanner signals,
  keeps those with `delivery_status` SENT from the last 3 hours that have no opinion yet,
  downloads 6 days of 15-minute candles for those symbols only, scores each signal and
  its mirror with the champion, and POSTs the result to the n8n webhook
  `nics-lab-opinion` (same lab key), which sends it to Alexander in Telegram. Answered
  signals are logged in `shadow/opinions_sent.csv`. Shadow only: nothing changes
  signals, users, the bot or the EA.
- `scripts/windows/hidden.vbs`: the 2-minute and hourly tasks run without a console window.
- `tests/test_opinion.py` (offline).

### Fixed
- `scripts/direction_report.py` counted every scanner record (about 1000 a day, mostly
  READY repeats of the same idea every 3 minutes), which inflated the totals. The headline
  now counts only signals actually sent to users (`delivery_status` SENT); all records are
  also reported deduplicated by symbol + direction + hour, and raw. `plan_group` is
  reported as the asset class it is (forex, gold, stocks ...), not as signal strength.

## [0.7.2] - 2026-11-20

### Fixed
- Night of 19-20.11 the laptop was on but asleep, so neither the 03:30 cycle nor the hourly
  shadow scoring ran until it woke at ~08:00. Both scheduled tasks now have `WakeToRun`
  (`scripts/windows/install_tasks.ps1`). Takes effect after `install_tasks.bat` is run once
  more. Windows must allow wake timers (Power options -> Sleep -> Allow wake timers).

## [0.7.1] - 2026-11-20

### Added
- **Push to the private dashboard.** After every nightly cycle `scripts/self_learn.py`
  POSTs the cycle report (same content as `self_learn_last.json`) and
  `shadow/direction_last.txt` to the n8n webhook `nics-lab-metrics`
  (`src/nics/telemetry/push.py`). It reuses `NICS_DATA_KEY` (header `X-NICS-Lab-Key`)
  and derives the URL from `NICS_DATA_URL`; `NICS_METRICS_URL` overrides it,
  `NICS_DASHBOARD_PUSH=0` turns it off. Metrics only: no weights, data or keys leave
  the laptop. A failed push is logged and never fails the cycle.
  Tests: `tests/test_dashboard_push.py`.

## [0.7.0] - 2026-11-19

### Added
- **Direction shadow test** (Alexander's request). Every hour `score_live.py`
  asks the champion which direction it would take for each new scanner signal:
  it scores the signal and its mirror (opposite direction, same entry, same
  stop/take distances, same reward:risk) and keeps the side with the higher
  P(TP1 before SL) in `shadow/direction.csv`. Nothing is sent anywhere; the live
  signals and the EA are untouched.
- `scripts/direction_report.py`, run by the nightly cycle: per Sofia day, scanner
  vs model in R per signal, R total and $ at a fixed risk per signal (100 $),
  agreements, who won where they disagreed, per plan group. Outputs
  `shadow/direction_reports/<day>.json`, `shadow/direction_daily.csv`,
  `shadow/direction_last.txt` (Russian summary); the last 3 days are recomputed
  until every trade has an outcome. Summary also in `self_learn_last.json`.
- `market.mirror_setup`, `build_samples(..., mirror_scanner=True)` (never used in
  training); `tzdata` in requirements (Sofia days on Windows).
- `tests/test_direction_shadow.py`.

## [0.6.0] - 2026-11-19

### Added
- **Long public H1/H4 history from Dukascopy** (`scripts/fetch_dukascopy.py`,
  dataset `nics-dukascopy-history-v1`): FX, gold, silver, Brent, indices and BTC
  since 2008 (`data.dukascopy.from_year`). Monthly files, paced and cached in
  `data/cache/dukascopy/`; high/low columns and price scale detected from the data
  and checked against our broker close. The nightly cycle runs it after the
  D1/W1 download; a refusal or failure only skips it for that night.
- **Methodologies on public history** (`methods.on_public`): the four methods run
  on every public 1h/4h series too, labelled on the same candles over the same
  48 h. These setups are training-only (never in validation or test, so the
  champion is still chosen on our own signals), and every signal batch is half
  our setups, half public ones.
- `tests/test_dukascopy.py` (offline, synthetic file).

## [0.5.2] - 2026-11-19

### Changed
- `install_tasks.bat` now registers the tasks through `install_tasks.ps1`
  (Register-ScheduledTask): a missed 03:30 run starts as soon as the laptop is
  on again (StartWhenAvailable), both tasks wait for a network connection and
  run on battery, no overlapping runs. The laptop does not have to stay on.
- Laptop guide section 5 rewritten accordingly.
- `self_learn.py` lock stores the PID: a lock left by a laptop switched off
  mid-cycle is ignored on the next run instead of blocking it for 12 hours.

## [0.5.1] - 2026-11-19

### Fixed
- Windows setup: `requirements/cpu.txt` used `--index-url` for the PyTorch CPU
  index, which hid PyPI, so `pyyaml` and then `torch` were not installed. Now
  `--extra-index-url`.
- `scripts/windows/*.bat` call `venv\Scripts\python.exe` directly (no
  `activate`, which PowerShell blocks by default) and `setup.bat` stops with
  `SETUP FAILED` on the first error instead of carrying on.
- Laptop guide: PowerShell-safe commands, open a new window after `setx`,
  three new troubleshooting rows.

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
