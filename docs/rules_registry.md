# Rules registry: what NICS Trading has been given, so nothing is added twice

Every trading rule NICS Trading gets is listed here once: where the idea comes
from, how exactly it is coded, when it was added, how it was tested and what
came of it. Book methods are turned into rules **as code with our own
parameters**; no book text is copied (control point 2). Before adding a rule,
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
| 1 | rule_strong_trend_pullback | Raschke & Connors (ADX trend pullback) | 1h/4h. ADX(14) > 30, EMA20 sloping over 5 bars, bar touches EMA20 without closing more than 0.25 ATR through it. **Buy stop** 0.05 ATR above that bar's high, valid 3 bars (mirror for shorts). Stop 0.4 ATR beyond the last 3 bars' extreme. Target is the 20-bar high (low) before the signal, at least 1 R. A bar touching both stop and target counts as a loss. | 2026-11-21 | shadow | 2026-12-05 to 12-19 | — |
| 2 | (planned) | Elder, triple screen | 4h trend filter plus 1h oscillator pullback plus stop entry | — | queued | — | — |
| 3 | (planned) | Wyckoff, spring / upthrust | false break of a range edge, close back inside | — | queued | — | — |
| 4 | (planned) | Al Brooks, trend pullbacks | second entry (H2/L2) in a trend | — | queued | — | — |

Differences from what is already in training:
- rule 1 needs a strong trend (ADX > 30, not only EMA20 > EMA50) and enters by
  stop order after the pullback bar, not at its close;
- rule 3 is not the same as `method_level_bounce`: it needs a break outside the
  range first.
