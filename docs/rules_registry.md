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

## Candidates (shadow / queue)

| # | id | idea from | definition | added | status | forward review | result |
|---|---|---|---|---|---|---|---|
| 1 | rule_strong_trend_pullback | pullback in a strong ADX trend | 1h/4h. ADX(14) > 30, EMA20 sloping over 5 bars, bar touches EMA20 without closing more than 0.25 ATR through it. **Buy stop** 0.05 ATR above that bar's high, valid 3 bars (mirror for shorts). Stop 0.4 ATR beyond the last 3 bars' extreme. Target is the 20-bar high (low) before the signal, at least 1 R. A bar touching both stop and target counts as a loss. | 2026-11-21 | shadow | 2026-12-05 to 12-19 | — |
| 2 | rule_level_squeeze_breakout | level methodology: squeeze into a level and break it | 1h/4h. Level = at least 2 swing highs (lows) within 0.2 ATR in the last 200 bars. The last 3 bars all reach within 0.3 ATR of the level without closing through it, average range < 0.8 ATR, and lows rise under resistance (highs fall over support). **Buy stop** 0.1 ATR beyond the level (luft), valid 3 bars. Stop 0.1 ATR beyond the last 2 bars' extreme. Target 3 R. Only while less than 75 % of the average daily range is used (ATR reserve). | 2026-11-21 | shadow | 2026-12-05 to 12-19 | — |
| 3 | (planned) | multi-timeframe screen | 4h trend, 1h oscillator pullback, stop entry | — | queued | — | — |
| 4 | (planned) | level methodology: false breakout of a level or range edge | break beyond the level, close back inside, entry on return | — | queued | — | — |
| 5 | (planned) | second entry in a trend | second pullback attempt (H2/L2) in a trend | — | queued | — | — |

Differences from what is already in training:
- rule 1 needs a strong trend (ADX > 30, not only EMA20 > EMA50) and enters by
  stop order after the pullback bar, not at its close;
- rule 2 trades the BREAK of a level after a squeeze; `method_level_bounce` trades
  the bounce off it, and `method_breakout20` breaks a plain 20-bar channel with no
  level or squeeze;
- rule 4 is not the same as `method_level_bounce`: it needs a break outside the
  level first.
