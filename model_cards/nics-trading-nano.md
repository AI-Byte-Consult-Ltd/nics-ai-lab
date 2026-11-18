# Model Card: `nics-trading-nano` (NICS Trading)

## Status

**Prototype: code ready, not trained yet** (2026-11-18). Nothing here is a
result. This card is updated only after a real training run on our data.

## Summary

The first NICS Trading model: a small Transformer over market candles,
trained from random initialisation only on AI Byte Consult's own data
(`nics-market-v1`: Vantage MT5 candles + NICS scanner signals with
outcomes). It is meant to run in the shadow next to the existing scanner
and be compared on real outcomes before it influences anything.

## Design

| Field | Value |
|---|---|
| Architecture | Linear input projection + 3 Transformer blocks (same from-scratch blocks as NICS Language) + 2 heads -- `src/nics/models/trading/model.py` |
| Parameters | 163,396 (`configs/trading/nano.yaml`) |
| Input | last 96 M15 bars (24 h): returns, ranges and wicks in ATR units, volatility regime, volume z-score, time of day / week |
| Horizon head | price 16 bars (4 h) later: down / flat / up (±0.5 ATR) |
| Signal head | probability that a scanner signal reaches TP1 before SL |
| Pretrained weights | **None** |
| Hardware | CPU, Alexander's laptop; weights stay on the laptop |

## Evaluation rules

- Train / val / test split by time with an embargo gap; test is used once.
- Signal outcomes are recomputed from candles; a bar touching TP1 and SL
  counts as a loss.
- Early stopping on validation guards against overfitting.
- Every metric is reported next to a baseline (majority class, base rate)
  and as mean R per trade after spread. The filter threshold is chosen on
  val only.
- Results are measurements on past data, never a forecast or a guarantee.

## Intended use

Shadow scoring of scanner signals for internal comparison. Not for
customer-facing use, not connected to any trading account.
