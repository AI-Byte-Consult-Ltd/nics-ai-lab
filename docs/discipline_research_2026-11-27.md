# Risk discipline and trade management: two psychology books tested on our data (27.11.2026)

Research only. Nothing was changed in the scanner, the EAs or n8n.

On 27.11.2026 Alexander asked us to review two well-known books on trading psychology and
risk discipline, and to feed whatever is useful into the nightly NICS Trading model. As for
every book, **only rules written as code with our own parameters** go into the repository.
No book text is copied, and no author or book names appear in code, ids, outputs or this
document (control point 2, `docs/rules_registry.md`).

Most of both books is about the trader's mind: accepting risk before the trade, thinking
in probabilities, consistency, and not fighting the market after a loss. None of that can be
coded. The parts that are mechanical, and the claims they make about them, are these:

| id | claim, in our words | how we coded it |
|---|---|---|
| disc_independence | consecutive trades are independent, so a losing streak says nothing about the next trade | next-trade R after exactly k losses or wins; runs test; lag-1 correlation |
| disc_edge_window | an edge shows only over a sample of trades | does the mean of the last 20 trades predict the next one? |
| disc_sizing_* | fixed risk per trade; never "win it back" | fixed risk vs half risk after 3 losses, doubling after a loss, ×1.5 after a win, and size by the 20-trade window |
| disc_let_run | do not cut winners short | trailing stop (best close − initial risk) instead of a fixed take |
| disc_pyramid | add to winners | add units at +1 R and +2 R. The stop goes to the previous entry, so only the newest unit is at risk. Sizes are equal, increasing (1/3, 2/3, 1) or decreasing (1, 1/2, 1/4) |
| disc_average_down | never add to a loser | add a unit at −0.5 R, stop unchanged (control) |
| disc_time_stop | cut losers quickly | close after 24 h unless the trade is at least +0.25 R |

Code: `src/nics/market/discipline.py`, `scripts/discipline_research.py`, test `tests/test_discipline.py`.

## Data

1. **Trade sequences**: the 4.5-year scanner replay (real [01 CORE] code from 24.11, 12
   instruments, 2022-03 to 2026-11, spread included), 31,927 strong trades, base −0.054 R.
2. **Trade management**: hourly bars 2019/2020 to 11.2026, built from public 1-minute
   history (gold, silver, US500, US100, Brent, JP225, HK50, FRA40) and public hourly crypto
   bars (BTC, ETH, SOL). GER40 was left out because the public file mixes another price
   scale for 2020–2023. Typical broker spread is charged per unit opened. All variants use
   **the same entries**: daily trend (close and EMA50 on the same side of EMA200), an
   hourly close at a new 5-day extreme, risk 1.5 daily ATR, at most one entry per side
   every 5 days (1,524 trades). The control uses **random entries** (same count, random bar
   and side).
3. **EA 1.29 trade set** (689 trades, 01.10–24.11, fill at the signal M5 close ≈ real
   latency). This was for Alexander's pyramiding question:
   `nics-trader/research/pyramid-partial-2026-11-27` in project files.

The stop is checked first on every bar. A bar that opens through the stop fills at its
open. After an add, a bar that reaches the new stop counts as stopped. Trails move only
on closed bars.

## Results

### 1. Streaks and sizing (scanner replay, 31,927 trades)
- After exactly 1…5 losses in a row the next trade is −0.050, −0.053, −0.050, −0.039 and −0.042 R, against −0.054 R for all trades. After wins it is the same. Runs test per instrument: z from −0.93 to +1.93. **The trades are independent: confirmed.**
  (Across all instruments in time order, losses seem to cluster: −0.16 R after 3 losses. But that comes from trades still open together. Counting only trades already closed at entry, it falls to −0.057 R.)
- The mean of the last 20 trades does not predict the next one: correlation −0.007, quintiles −0.047 … −0.062 R.
- Sizing rules change nothing per unit of risk (−0.052 … −0.056 R). Doubling after a loss turns −1,717 R into −4,160 R, with drawdown 620 R instead of 251 R.

### 2. Trade management (hourly bars, same 1,524 trend entries)
| variant | R per trade | t | drawdown R | before 2023 / from 2023 | random entries R per trade |
|---|---|---|---|---|---|
| stop and take 1.5 ATR | +0.054 | 2.4 | 45 | +0.043 / +0.062 | +0.000 |
| + time stop 24 h | +0.039 | 2.5 | 26 | +0.026 / +0.047 | −0.011 |
| trail, no take | **+0.121** | 4.0 | 52 | +0.112 / +0.127 | +0.043 |
| trail + time stop | +0.069 | 3.2 | 39 | | −0.008 |
| add equal units at +1 R and +2 R | +0.185 | 3.4 | 88 | +0.171 / +0.193 | +0.066 |
| add increasing (1/3, 2/3, 1) | +0.088 | 2.9 | 47 | | +0.029 |
| add decreasing (1, 1/2, 1/4) | +0.148 | 3.7 | 66 | | +0.055 |
| average down at −0.5 R (control) | +0.162 | 4.0 | 59 | | +0.040 |

What this means:
- **Letting winners run works on an entry with an edge**: about twice the fixed take, in both periods. Profit comes only from longs (shorts −0.07 R in every variant) and is fat-tailed: without the best 5 % of trades the total is negative.
- **Cutting losers by time hurts** everywhere: rejected.
- **Adding to winners adds R only because it adds exposure.** Per unit opened, equal adds earn 0.185 / 1.63 = 0.11 R. A single unit with a trail earns 0.12 R. Averaging down earns 0.10 R per unit, about the same. Adding does not beat a single unit per unit of risk, and the drawdown grows (88 R against 52 R). On random entries all add variants are near zero (t 0.3 to 1.5). Adds amplify whatever the entry already has.
- Of the size schemes, decreasing adds (1, 1/2, 1/4) keep the most of the gain per unit of drawdown. Increasing adds (small first, big later) are the weakest.

### 3. EA 1.29 trade set (689 trades)
- With TP2 as today, every add is worse than the base (+46.0 R): from −6 to −75 R, with larger drawdown. TP2 is only about 1.3 R from entry, so there is no room to add.
- A partial close at TP1 (50 %, or just enough to pay back the risk) gives +56…+58.5 R. That is better in both months, with a smaller drawdown, but only about +0.015 R per trade. It is offered as a shadow, not as an EA change.
- Dropping TP2 (24 h hold) gives +181 R, and with adds up to +259 R. But 180 of the 181 R came from shorts in a falling November; longs were about zero. That is market drift, not a property of the rule.

## What goes into the nightly training

- **Add-on setups as a trial** (`method_add_on_winner`, `methods.add_on_winners`). A method
  setup that reaches +1 R before its stop gets a second setup at that bar's close, with its
  stop at the first entry and a target of 1 R of its own risk. The model learns when a
  trade that is already working tends to keep working. That is the question behind adding
  to winners, and the research above says the answer depends on the entry. The setups only
  teach: they never enter validation or test, so the trial model is judged on exactly the
  champion's validation set. `self_learn.py` trains one extra fresh model with them, on
  the first night and then on scratch nights (`add_ons.trial` in `configs/trading/nano.yaml`).
  The model replaces the champion only under the usual rules: better validation score by
  at least 0.002, and no drop in validation signal AUC or horizon accuracy.
- Independence, fixed risk and trailing exits are kept as checked principles (status
  "checked" in the registry). Nothing is learned from them.
- Rejected: size by the 20-trade window, the time stop, averaging down, and doubling after a loss.

Nothing here goes to live trading or into EA logic without Alexander's OK.

## Limits
- Public hourly bars, not our broker feed. A constant spread, no swap (multi-day trailed trades would pay swap). Crypto starts in 2019, the others in 2020.
- Trend entries are a simple test entry, not our scanner. The scanner has no edge on the 4.5-year replay, so adds cannot help it there.
- The 1.29 set covers only two months, one of them a strong one-way market.
