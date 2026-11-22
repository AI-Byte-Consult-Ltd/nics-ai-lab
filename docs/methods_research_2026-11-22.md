# Ten well-known trading methods on our data (22.11.2026)

Research only. Nothing here changes the scanner, the level method, the bot or any EA.
Code: `src/nics/market/classic_methods.py` (rules + backtester), `scripts/methods_research.py`
(reproduces the main table from a lab snapshot). Names are neutral: no authors, schools,
books or companies. Every rule is written as code with our own parameters.

## What was tested and how

**Data.** Public daily history of our 37 instruments (internal research use only), 1990–2026
(USD/CNH from 2011: before that the public series is the pegged onshore rate; SPCX has only
113 days and is too short for a daily test, it is not dropped from anything else). Broker
prices from our MT5 bridge for the spreads and for the same-window check (D1 since 2025-09,
H1 since July–September 2026).

**Split.** In-sample = entries before 2018-01-01. Out-of-sample = 2018-01-01 to 2026-11-20.
All parameters were fixed BEFORE the first run (the usual textbook values); a second run picked
parameters on in-sample only and reported out-of-sample for that choice.

**Execution (honest).** Decision at the close of a daily bar, entry at the next open. The stop
is checked inside every bar; a gap through the stop exits at the open (worse than the stop).
Stop before target on the same bar. Stops never closer than 0.5 ATR. One position per method
and instrument. A position still open at the end is marked at the last close.

**Costs.** 1.2 × median broker spread (ask − bid from our bridge) per round trip, plus swap for
every calendar day held: forex 2 %/year, gold and metals 3 %, oil and indices 4 %, stocks 5 %,
crypto 20 % (Vantage crypto financing is high). Charged on both longs and shorts (conservative).

**R.** Result in units of the initial risk (entry − stop), after costs.

## The ten methods (new) and ours (same daily data, same execution)

| id | idea |
|---|---|
| m01_ma_cross | EMA50 crosses EMA200; stop 3 ATR; exit on the opposite cross |
| m02_channel_55_20 | close beyond the 55-day high/low; exit beyond the 20-day low/high; stop 2 ATR |
| m03_rsi2_pullback | RSI(2) < 10 above SMA200 (> 90 below); exit on a close beyond SMA5; stop 3 ATR; max 10 days |
| m04_band_reversion | close outside Bollinger(20, 2); exit at the middle; stop 2 ATR; max 15 days |
| m05_macd_trend | MACD crosses its signal line on the side of zero; exit on the opposite cross; stop 2.5 ATR |
| m06_nr7_breakout | narrowest range of 7 days; stop orders both sides; stop at the other side; 2R; max 5 days |
| m07_tsmom_12m | at each month end: long if the 252-day return > 0, short if < 0; stop 3 ATR |
| m08_atr_trail | ATR(10)×3 trailing trend line, flip on a cross; the line is the stop |
| m09_cloud | 9/26 lines cross with the close beyond the 9/26/52 cloud; exit beyond the 26-line; stop 2.5 ATR |
| m10_squeeze | Bollinger width at its 120-day low, then a close outside the band; exit at the middle; stop 2 ATR |
| o1–o4 (ours) | the 4 methods already in NICS Trading training and the D20 swing shadow rule, ported to D1 |

## Results (36 instruments, R after costs)

| method | IS n | IS avg R | OOS n | OOS R | OOS avg R | t | win | PF | max DD R | OOS R per instrument-year | instruments + |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **m07 12-month momentum** | 1120 | +1.56 | 594 | +503 | **+0.85** | **2.98** | 26 % | 2.10 | 173 | +1.57 | 67 % |
| **m01 EMA50/200** | 745 | +1.34 | 476 | +460 | **+0.97** | **2.70** | 21 % | 2.20 | 147 | +1.44 | 61 % |
| **m02 channel 55/20** | 2489 | +0.20 | 1304 | +264 | **+0.20** | **2.11** | 29 % | 1.28 | 112 | +0.82 | 56 % |
| m09 cloud | 2764 | +0.10 | 1479 | +81 | +0.055 | 1.49 | 36 % | 1.14 | 48 | +0.25 | 58 % |
| m05 MACD trend | 4476 | +0.04 | 2532 | +57 | +0.023 | 1.01 | 35 % | 1.07 | 50 | +0.18 | 56 % |
| m10 squeeze | 1027 | −0.02 | 564 | +28 | +0.049 | 0.73 | 33 % | 1.10 | 38 | +0.09 | 47 % |
| m03 RSI(2) pullback | 6813 | +0.01 | 3683 | −15 | −0.004 | −0.49 | 61 % | 0.98 | 63 | −0.05 | 42 % |
| m08 ATR trail | 4772 | −0.01 | 2487 | −68 | −0.027 | −1.02 | 35 % | 0.94 | 126 | −0.21 | 31 % |
| m06 NR7 breakout | 16835 | −0.01 | 8932 | −382 | −0.043 | −2.78 | 39 % | 0.94 | 470 | −1.20 | 42 % |
| m04 band reversion | 8265 | −0.08 | 4825 | −401 | −0.083 | −5.57 | 48 % | 0.84 | 401 | −1.25 | 25 % |
| o3 ours: EMA pullback | 6847 | −0.05 | 4078 | −33 | −0.008 | −0.41 | 44 % | 0.99 | 121 | −0.10 | 42 % |
| o2 ours: breakout 20 | 7106 | −0.06 | 3520 | −121 | −0.034 | −1.66 | 44 % | 0.94 | 184 | −0.38 | 42 % |
| o4 ours: D20 swing | 10394 | −0.03 | 5195 | −136 | −0.026 | −1.82 | 46 % | 0.94 | 236 | −0.43 | 36 % |
| o1 ours: range both sides | 2478 | −0.21 | 1529 | −290 | −0.189 | −4.00 | 26 % | 0.77 | 345 | −0.91 | 33 % |

**Pre-registered bar to count as "better than ours":** out-of-sample average R > 0 with t ≥ 2,
PF ≥ 1.2, positive on more than half of the instruments, and better than random entries with the
same direction and holding time. Passed: m07, m01, m02. Borderline (positive, not significant):
m09, m05. All four of our own methods are negative on daily bars.

## Checks that make the result smaller (read before using it)

1. **Random-timing control** (same instrument, direction and holding time, random entry day,
   same stop and costs, 20 draws per trade, out-of-sample):
   - m01 longs +2.53 vs random +0.50; shorts −0.80 vs random −0.40;
   - m07 longs +1.84 vs +0.58; shorts −0.70 vs −0.43;
   - m02 longs +0.68 vs +0.14; shorts −0.52 vs −0.28.
   The timing adds value on the long side. **Shorts lose in every trend method and do worse than
   random shorts**: part of the profit is the 2018–2026 up-drift of stocks, gold and crypto.
2. **Fat tails.** Without the best 1 % of trades, out-of-sample: m07 +211 R, m01 +182 R,
   m02 −58 R. Most of the money comes from a few long trends (BTC, NVIDIA, gold).
3. **Lumpy years.** Out-of-sample positive years: m07 6/9, m01 5/9, m02 4/9. Big years
   2018, 2022, 2026; flat or negative 2019–2020, 2023–2024.
4. **By asset group (OOS R):** stocks and crypto carry it; **forex is negative in every trend
   method** (m02 −103, m01 −68, m07 −35); indices about zero.
5. **Parameters chosen on in-sample only:** m07 (lookback 63, stop 2 ATR) → OOS +0.32 R/trade,
   t 1.70, 100 % of the grid positive OOS; m02 (100/20, 2 ATR) → +0.14, t 1.26, 86 % of the grid
   positive; **m01 (100/300, 4 ATR) → −0.09, t −0.26** although 92 % of its grid is positive OOS
   (the default 50/200 works, the "best" in-sample choice does not: fragile).
6. **Horizon.** These are position methods: holds of weeks to months, 20–30 % win rate,
   drawdowns of 110–170 R across the whole portfolio. They do not fit the prop rules
   (close everything on Friday) and are a different product from the intraday scanner.

## Same window as our scanner and level method (15.09–21.11.2026, broker prices)

| what | trades | R | note |
|---|---|---|---|
| Scanner, SENT signals | 2823 | −0.4 | internal tally (`realized_r`), known to be optimistic |
| Level method v0.3 backtest (21.11 research) | 2656 | −205 | before it went live; live since 21.11 on BTC, others from 23.11 |
| m01 on D1 | 14 | +7.0 | open positions marked at 21.11 |
| m02 on D1 | 59 | +14.4 | |
| m07 on D1 | 58 | −31.6 | |
| m09 on D1 | 33 | +9.9 | |
| m05 on D1 | 47 | +9.5 | |
| m05 on H1 (12 instruments with enough H1) | 376 | +38.1 | t 1.1 |
| m02 on H1 | 215 | +22.3 | t 0.5 |

Ten weeks is far too short for methods that hold for months; these numbers are a sanity check,
not evidence either way.

## What was done with it

- **NICS Trading model:** the daily states of m07, m01, m02 (+ borderline m09, m05) and their
  average vote are six input features on every bar of every timeframe (`trend_context.py`,
  version 0.12.0). Not filters, not setups. Quick check on our broker 1h/4h/D1 data (1500 steps,
  same seed, with vs without): validation score 1.7432 vs 1.7415 (lower is better), signal test
  AUC 0.589 vs 0.585, horizon test accuracy 0.357 vs 0.367: **no gain for the intraday heads**.
  So they run as a **trial**: every scratch night the laptop trains one extra fresh model with
  them, and it becomes champion only if it beats the current one on the same new data.
- **Shadow in n8n:** `[48 SHADOW] Daily Trend Methods (no delivery)` measures m07, m01 and m02
  forward on broker D1 from 23.11.2026 into the table `trend_shadow_trades`. Nothing is sent to
  anyone. First look 22.12.2026; a real verdict needs about 3 months (few trades, long holds).
- **Rejected:** m03, m04, m06, m08, m10 (registry `rejected_*`).
- Going live in any form needs Alexander's explicit OK.
