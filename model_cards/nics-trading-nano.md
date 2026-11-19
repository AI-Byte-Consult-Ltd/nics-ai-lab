# Model Card: `nics-trading-nano` (NICS Trading)

## Status

**Prototype: code ready, not trained yet** (2026-11-18). Nothing here is a
result. This card is updated only after a real training run on our data.

## Summary

The first NICS Trading model: a small Transformer over market candles,
trained from random initialisation on all of AI Byte Consult's trading
data (`nics-market-v2`: Vantage MT5 candles on every timeframe, every
scanner signal with its outcome, shadow strategies, EA signals;
`nics-own-trades-v1`: our closed MT5 positions) plus public D1/W1 history
(`nics-public-history-v1`, can be switched off with `--no-public`). It is meant to run in the shadow next to the existing scanner
and be compared on real outcomes before it influences anything.

## Design

| Field | Value |
|---|---|
| Architecture | Linear input projection + 3 Transformer blocks (same from-scratch blocks as NICS Language) + 2 heads -- `src/nics/models/trading/model.py` |
| Parameters | 166,532 (`configs/trading/nano.yaml`) |
| Input | last 96 bars of one series, any timeframe (1min, 5min, 15min, 1h, 4h, D1, W1): returns, ranges and wicks in ATR units, volatility regime, volume z-score, time of day / week, indicators (RSI, EMA 20/50/200, MACD, Bollinger, stochastic, ADX, channel / level distances), candle patterns, timeframe and source (broker / public) flags |
| Horizon head | price N bars later: down / flat / up (±0.5 ATR); N per timeframe in the config (1 h ahead on 1min up to 1 month ahead on W1) |
| Signal head | probability that a trade setup reaches TP1 (or its TP) before SL; setups from every source: scanner (all modes, strong and weak), shadow strategies, EA signals, our own trades, and setups our trading methodologies generate on history; source / method, scanner scores, TP2/TP3, strategy version and daily-range room are inputs |
| Pretrained weights | **None** |
| Hardware | CPU, Alexander's laptop; weights stay on the laptop |

## Self-learning

`scripts/self_learn.py` runs nightly on the laptop: fresh data, champion
re-scored on it, warm-started challenger (weekly one from random weights),
promotion only when the challenger's test_score beats the champion's on the
same new test split. The champion scores live signals in the shadow
(`scripts/score_live.py`); those scores are later checked against real
outcomes (forward test). It does not trade or send signals.

## Evaluation rules

- Train / val / test split by time with an embargo gap; test is used once.
- Signal outcomes are recomputed from candles; a bar touching TP1 and SL
  counts as a loss.
- Early stopping on validation guards against overfitting.
- Metrics are reported per timeframe, broker-only, and per setup source.
- Candle-derived outcomes are checked against the tracker's recorded
  TP1/SL events (`label_agreement_with_tracker`).
- The economic calendar is not a feature yet: it only covers 2026-10-20
  onward, so it would differ between training and test periods.
- Every metric is reported next to a baseline (majority class, base rate)
  and as mean R per trade after spread. The filter threshold is chosen on
  val only.
- Results are measurements on past data, never a forecast or a guarantee.

## Intended use

Shadow scoring of scanner signals for internal comparison. Not for
customer-facing use, not connected to any trading account.
