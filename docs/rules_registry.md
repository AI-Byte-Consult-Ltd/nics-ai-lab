# Rules registry: what NICS Trading has been given, so nothing is added twice

Every trading rule NICS Trading gets is listed here once: where the idea comes
from (in neutral words), how exactly it is coded, when it was added, how it was
tested and what came of it. Book methods are turned into rules **as code with our
own parameters**; no book text is copied (control point 2). **No names of
authors, schools, courses or companies** appear anywhere: not in code, ids,
comments, docs, outputs or anything users see. Every rule has a neutral internal
id. `tests/test_rules_shadow.py` checks the rule outputs for such names (the list is
stored as hashes, so the names are not in the repository). Before adding a rule,
check this list. If a similar rule already exists, change that rule's
parameters (a new row with a new version) instead of adding a duplicate.

Why not give the model the same rule twice: a duplicate adds no new knowledge.
It doubles the weight of one idea in training, and that makes the model
overconfident in it.

## How a rule moves

1. **shadow**: coded in `src/nics/market/rules.py`. Every night
   `scripts/rules_shadow.py` measures it on its own:
   - on public history since 2008 (backtest, no spread);
   - on our broker candles before the day it was added;
   - **forward** from the day it was added (the real test).

   The model does not learn from it yet. Results are written to
   `shadow/rules_last.txt` and shown on the CEO desk.
2. **Review after 2–4 weeks**. Promote only if all of these hold:
   - forward results have at least 30 trades with an outcome;
   - average R after spread is above 0;
   - the backtest is not negative.

   Otherwise the rule is rejected, and the reason is written here.
3. **promoted**: the rule moves to `src/nics/market/methods.py` with a new kind in
   `data.py`. This changes the feature schema, so it is done in batches, never
   one rule per night. The next challenger trained with it must beat the champion
   on the same data, or the change is rolled back.

## Already in training (since 0.4.0 / 0.6.0)

| id | idea | definition (short) |
|---|---|---|
| method_range_both_sides | range trading, both sides | ADX < 20 and 48-bar channel at least 3 ATR wide; long from the bottom, short from the top; stop 0.6 ATR beyond; target at mid-channel |
| method_breakout20 | channel breakout | close beyond the 20-bar channel with ADX > 20; stop 1.5 ATR, target 2 ATR |
| method_ema_pullback | trend pullback to EMA20 | EMA20 > EMA50, both rising; bar touches EMA20 and closes above it; stop 0.5 ATR beyond the 5-bar extreme; target 1.5 R |
| method_level_bounce | bounce from a swing level | level touched at least twice in 200 bars; price comes back and closes off it; stop 0.7 ATR beyond; target 2 R |

## Model features: daily trend context (since 0.12.0)

From the research of 22.11.2026 (`docs/methods_research_2026-11-22.md`): ten well-known methods
coded as rules and backtested on 1990–2026 daily history, out-of-sample from 2018, costs and
swaps included. The state of each method below (+1 long, −1 short, 0 flat) on the instrument's
daily candles is an input on every bar of every timeframe (`src/nics/market/trend_context.py`),
plus their average vote. Inputs, not filters: the model decides what they are worth. Since
0.12.0 as a **trial** (`trend_context.trial`): a model with them replaces the champion only if it
scores better on the same new data.

| id | idea | OOS avg R | t | status |
|---|---|---|---|---|
| trend_momentum_12m | sign of the 252-day return, decided at month end, stop 3 ATR | +0.85 | 2.98 | feature + shadow `[48 SHADOW]` |
| trend_ma_50_200 | EMA50 vs EMA200, stop 3 ATR | +0.97 | 2.70 | feature + shadow `[48 SHADOW]` |
| trend_channel_55_20 | 55-day breakout, 20-day exit, stop 2 ATR | +0.20 | 2.11 | feature + shadow `[48 SHADOW]` |
| trend_cloud | close beyond the 9/26/52 cloud with 9 > 26 (mirror) | +0.06 | 1.49 | feature only (borderline) |
| trend_macd | MACD and signal agree with the sign of MACD | +0.02 | 1.01 | feature only (borderline) |

Caveats: the profit is long-side, fat-tailed and lumpy by year; forex is negative in all of
them; the shadow stream needs about 3 months for a verdict.

## Rejected (22.11.2026, daily 1990–2026, out-of-sample 2018–2026)

| id | idea | OOS avg R | t | reason |
|---|---|---|---|---|
| rejected_rsi2_pullback | RSI(2) extreme in the SMA200 direction | −0.004 | −0.49 | zero after costs |
| rejected_band_reversion | back to the mean from outside Bollinger(20, 2) | −0.083 | −5.57 | negative in both periods |
| rejected_nr7_breakout | stop orders around the narrowest of 7 bars | −0.043 | −2.78 | negative out of sample |
| rejected_atr_trailing_trend | ATR(10)×3 trailing line with flips | −0.027 | −1.02 | negative |
| rejected_volatility_squeeze | band-width low, then a close outside the band | +0.049 | 0.73 | not significant, IS negative |

## Candidates (shadow / queue)

| # | id | idea from | definition | added | status | forward review | result |
|---|---|---|---|---|---|---|---|
| 1 | rule_strong_trend_pullback | pullback in a strong ADX trend | 1h/4h. ADX(14) > 30, EMA20 sloping over 5 bars, bar touches EMA20 without closing more than 0.25 ATR through it. **Buy stop** 0.05 ATR above that bar's high, valid 3 bars (mirror for shorts). Stop 0.4 ATR beyond the last 3 bars' extreme. Target is the 20-bar high (low) before the signal, at least 1 R. A bar touching both stop and target counts as a loss. | 2026-11-21 | shadow | 2026-12-05 to 12-19 | — |
| 2 | rule_level_squeeze_breakout | level methodology: squeeze into a level and break it | 1h/4h. Level = at least 2 swing highs (lows) within 0.2 ATR in the last 200 bars. The last 3 bars all reach within 0.3 ATR of the level without closing through it, average range < 0.8 ATR, and lows rise under resistance (highs fall over support). **Buy stop** 0.1 ATR beyond the level (luft), valid 3 bars. Stop 0.1 ATR beyond the last 2 bars' extreme. Target 3 R. Only while less than 75 % of the average daily range is used (ATR reserve). | 2026-11-21 | shadow | 2026-12-05 to 12-19 | — |
| 3 | queued_multi_timeframe_screen | multi-timeframe screen | 4h trend, 1h oscillator pullback, stop entry | — | queued | — | — |
| 4 | queued_level_false_breakout | level methodology: false breakout | break beyond a level or range edge, close back inside, entry on return | — | queued | — | — |
| 5 | queued_second_entry | second entry in a trend | second pullback attempt in a trend | — | queued | — | — |
| 6 | queued_accumulation_phases | accumulation / distribution phases | false exit from a range, then a test of it | — | queued | — | — |
| 7 | queued_trend_highs_lows | trend structure | sequence of highs and lows, trend change | — | queued | — | — |
| 8 | queued_classic_patterns | classic chart patterns | double top/bottom, flag, triangle | — | queued | — | — |
| 9 | queued_range_edges_price_action | price action at range edges | rejection, pin bar, engulfing at the edges | — | queued | — | — |
| 10 | queued_pullback_quality | pullback quality | depth and speed of the pullback against the impulse | — | queued | — | — |
| 11 | queued_adaptive_trend_system | adaptive trend system | adaptive moving average, noise filter | — | queued | — | — |
| 12 | queued_volume_price | volume and price | move confirmed by volume, climax volume | — | queued | — | — |
| 13 | queued_r_multiple_sizing | risk in R | position size from the stop, system judged in R | — | queued | — | — |

Psychology and discipline books give no codable entry rules and are not used for rules.

The machine-readable copy of this list is `src/nics/market/registry.py`. It is pushed
to the CEO desk every night, and a test keeps it in sync with the code.

Differences from what is already in training:
- rule 1 needs a strong trend (ADX > 30, not only EMA20 > EMA50) and enters by
  stop order after the pullback bar, not at its close;
- rule 2 trades the BREAK of a level after a squeeze; `method_level_bounce` trades
  the bounce off it, and `method_breakout20` breaks a plain 20-bar channel with no
  level or squeeze;
- rule 4 is not the same as `method_level_bounce`: it needs a break outside the
  level first.
