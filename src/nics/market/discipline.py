"""Risk discipline and trade management, written as testable rules (research only).

Two trading-psychology books were reviewed for NICS on 27.11.2026. Most of their content is
about the trader's mind and cannot be coded. The parts that are mechanical and can be checked
on data are written here as our own rules, with our own parameters; no book text is copied,
and, as everywhere in NICS, no names of authors or books appear in code, ids or outputs.

Checked claims, each with a neutral id:
  disc_independence      outcomes of consecutive trades are independent: a losing streak says
                         nothing about the next trade, so size should not depend on it.
  disc_sizing_*          fixed risk per trade vs size that reacts to streaks
                         (cut after losses, double after losses, raise after wins).
  disc_edge_window       an edge shows only over a sample of trades: does the average of the
                         last 20 trades predict the next one?
  disc_let_run           exit by a trailing stop instead of a fixed take.
  disc_pyramid           add a unit each time the trade is +1 R in profit (stop of the whole
                         position follows the trail, so each add is paid for by open profit).
  disc_average_down      add a unit when the trade is -0.5 R against us (the books say never;
                         tested as the control).
  disc_time_stop         a trade that is not at least +0.25 R after 24 hours is closed.

Every management variant is run on the SAME entries, so a difference is the management, not
the entry. Each is also run on random entries (random bar, random side): if a management rule
"earns" on random entries too, the gain is a property of the market drift, not of the rule.

Prices: hourly bars. Costs: one spread (fraction of price) per unit opened. A bar that touches
the stop is a stop, checked before anything else on that bar; a bar opening through the stop
fills at its open. Trailing stops move only on closed bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VARIANTS = ["fixed", "fixed_time_stop", "let_run", "let_run_time_stop", "pyramid", "pyramid_up", "pyramid_down",
            "average_down"]
# Unit sizes of the first entry and the two adds. The largest is 1, so the most money ever at the
# stop is the same as one base trade (after an add the stop sits at the previous entry, so only the
# newest unit is at risk; before it only the first one is).
PYRAMID_SIZES = {"pyramid": (1.0, 1.0, 1.0), "pyramid_up": (1 / 3, 2 / 3, 1.0), "pyramid_down": (1.0, 0.5, 0.25)}
VARIANT_RU = {
    "fixed": "стоп и тейк 1,5 ATR (как сейчас в тестах)",
    "fixed_time_stop": "то же + выход через 24 ч, если нет +0,25R",
    "let_run": "без тейка, стоп подтягивается за ценой",
    "let_run_time_stop": "без тейка + выход через 24 ч, если нет +0,25R",
    "pyramid": "без тейка + доливка на +1R и +2R равным лотом",
    "pyramid_up": "доливка: малый лот, потом больше (1/3, 2/3, 1)",
    "pyramid_down": "доливка: большой лот, потом меньше (1, 1/2, 1/4)",
    "average_down": "без тейка + доливка на −0,5R (контроль: «усреднение»)",
}


@dataclass
class Bars:
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741
    c: np.ndarray
    symbol: str = ""
    spread: float = 0.0  # fraction of price paid once per unit opened


def ema(x: np.ndarray, p: int) -> np.ndarray:
    a = 2.0 / (p + 1)
    e = np.empty_like(x)
    e[0] = x[0]
    for i in range(1, len(x)):
        e[i] = e[i - 1] + a * (x[i] - e[i - 1])
    return e


def daily(b: Bars, shift_hours: int = 2):
    """Daily bars from hourly ones (day boundary at 22:00 UTC, as the broker), ATR(14), EMA50/200.
    Returns (day index of every hourly bar, dict of daily arrays)."""
    dk = np.floor((b.t + shift_hours * 3600) / 86400).astype(np.int64)
    ud, di = np.unique(dk, return_index=True)
    dc = b.c[np.r_[di[1:] - 1, len(b.c) - 1]]
    dh = np.maximum.reduceat(b.h, di)
    dl = np.minimum.reduceat(b.l, di)
    tr = np.maximum(dh[1:] - dl[1:], np.maximum(abs(dh[1:] - dc[:-1]), abs(dl[1:] - dc[:-1])))
    tr = np.r_[dh[0] - dl[0], tr]
    atr = np.array([tr[max(0, i - 13):i + 1].mean() for i in range(len(tr))])
    pos = np.searchsorted(ud, dk)
    return pos, {"h": dh, "l": dl, "c": dc, "atr": atr, "e50": ema(dc, 50), "e200": ema(dc, 200), "first": di}


def trend_entries(b: Bars, days_n: int = 5, cooldown_h: int = 120):
    """Entry used for the management tests: daily trend (close and EMA50 on the same side of
    EMA200, decided on the last CLOSED day), hourly close at a new 5-day extreme in that
    direction, market entry at the next bar's open. At most one entry per side every 5 days.
    Returns a list of (bar index of entry, direction, risk distance = 1.5 daily ATR)."""
    pos, d = daily(b)
    out, last = [], {1: -10 ** 9, -1: -10 ** 9}
    hh = ll = None
    cur_day = -1
    for i in range(len(b.t) - 1):
        p = pos[i] - 1
        if pos[i] != cur_day:
            cur_day = pos[i]
            hh, ll = -math.inf, math.inf  # today's extremes so far (bars before i)
        if p < 210:
            hh, ll = max(hh, b.h[i]), min(ll, b.l[i])
            continue
        ph = max(d["h"][p - days_n + 1:p + 1].max(), hh)
        pl = min(d["l"][p - days_n + 1:p + 1].min(), ll)
        up = d["c"][p] > d["e200"][p] and d["e50"][p] > d["e200"][p]
        dn = d["c"][p] < d["e200"][p] and d["e50"][p] < d["e200"][p]
        dr = 1 if (up and b.c[i] > ph) else (-1 if (dn and b.c[i] < pl) else 0)
        hh, ll = max(hh, b.h[i]), min(ll, b.l[i])
        if dr == 0 or i - last[dr] < cooldown_h or b.t[i + 1] - b.t[i] > 3 * 3600:
            continue
        last[dr] = i
        out.append((i + 1, dr, 1.5 * d["atr"][p]))
    return out


def random_entries(b: Bars, like: list, seed: int = 7):
    """Control: as many entries as `like`, at random bars, random side, same risk scale."""
    rng = np.random.default_rng(seed)
    if not like:
        return []
    risk = np.array([x[2] for x in like])
    lo = like[0][0]
    bars = np.sort(rng.integers(lo, len(b.t) - 2, size=len(like)))
    return [(int(i), int(rng.choice([-1, 1])), float(rng.choice(risk))) for i in bars]


def simulate(b: Bars, i0: int, dr: int, D: float, variant: str, max_days: int | None = None) -> dict:
    """One trade from bar i0 (entry at its open), risk distance D. Result in R of ONE unit's
    initial risk, after spread. `units` = how many units were opened in total."""
    if not (D > 0):
        return {"r": 0.0, "units": 0, "bars": 0}
    fixed = variant.startswith("fixed")
    tstop = variant.endswith("time_stop")
    max_h = 24 * (max_days or (7 if fixed else 30))
    sp = b.spread
    e0 = b.o[i0]
    sizes = PYRAMID_SIZES.get(variant, (1.0, 1.0, 1.0))
    units = [(e0, sizes[0])]          # (entry price, size) of the open units
    cost = sp * e0 * sizes[0]         # spread paid, in price units
    stop = e0 - dr * D
    take = e0 + dr * D if fixed else None
    best = e0                         # best close-to-date for the trail (closed bars only)
    adds_done = 0
    avg_done = False
    n = len(b.t)
    for k in range(i0, min(n, i0 + max_h)):
        o, h, lo, c = b.o[k], b.h[k], b.l[k], b.c[k]
        adv, fav = (lo, h) if dr > 0 else (h, lo)  # adverse and favourable extreme of the bar
        # 1. stop first (gap through stop fills at the open)
        if (adv - stop) * dr <= 0:
            px = o if (o - stop) * dr <= 0 else stop
            # average-down unit triggered on this same bar is filled before the stop (worst case)
            if variant == "average_down" and not avg_done and (adv - (e0 - dr * 0.5 * D)) * dr <= 0:
                lvl = e0 - dr * 0.5 * D
                units.append((lvl if (o - lvl) * dr > 0 else o, 1.0))
                cost += sp * lvl
            return _close(units, px, dr, D, cost, k - i0 + 1)
        # 2. take (fixed variants)
        if take is not None and (fav - take) * dr >= 0:
            return _close(units, take, dr, D, cost, k - i0 + 1)
        # 3. adds
        if variant in PYRAMID_SIZES and adds_done < 2:
            lvl = e0 + dr * (adds_done + 1) * D
            if (fav - lvl) * dr >= 0:
                sz = sizes[adds_done + 1]
                units.append((lvl if (o - lvl) * dr < 0 else o, sz))
                cost += sp * lvl * sz
                adds_done += 1
                be = e0 + dr * (adds_done - 1) * D  # stop to the previous entry: only the new unit is at risk
                if (be - stop) * dr > 0:
                    stop = be
        if variant == "average_down" and not avg_done:
            lvl = e0 - dr * 0.5 * D
            if (adv - lvl) * dr <= 0:
                units.append((lvl if (o - lvl) * dr > 0 else o, 1.0))
                cost += sp * lvl
                avg_done = True
        # 4. time stop at the close of the 24th bar
        if tstop and k - i0 + 1 == 24 and (c - e0) * dr < 0.25 * D:
            return _close(units, c, dr, D, cost, 24)
        # 5. trail on the closed bar (not for fixed): stop = best close - D, never backwards
        if not fixed:
            if (c - best) * dr > 0:
                best = c
            new = best - dr * D
            if (new - stop) * dr > 0:
                stop = new
    k = min(n, i0 + max_h) - 1
    return _close(units, b.c[k], dr, D, cost, k - i0 + 1)


def _close(units, px, dr, D, cost, bars):
    pnl = sum((px - u) * dr * sz for u, sz in units) - cost
    return {"r": pnl / D, "units": len(units), "bars": bars}


def stats(r) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return {"n": 0, "mean": 0.0, "t": 0.0, "sum": 0.0, "win": 0.0, "dd": 0.0, "top5_share": 0.0}
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max())
    sd = r.std(ddof=1) if n > 1 else 0.0
    srt = np.sort(r)[::-1]
    top = srt[:max(1, n // 20)].sum()
    return {"n": n, "mean": round(float(r.mean()), 4), "t": round(float(r.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0, 2),
            "sum": round(float(r.sum()), 1), "win": round(float((r > 0).mean()), 3), "dd": round(dd, 1),
            "top5_share": round(float(top / r.sum()), 2) if r.sum() > 0 else None}


# ---------- trade sequences: streaks, sizing, edge window ----------

def streak_table(seqs: list[list[float]], max_k: int = 5) -> dict:
    """Mean R and win rate of the next trade after EXACTLY k losses (and k wins) in a row,
    within each sequence (one sequence = one instrument, trades in time order, one at a time)."""
    out = {"after_losses": {}, "after_wins": {}}
    for key, test in (("after_losses", lambda v: v < 0), ("after_wins", lambda v: v > 0)):
        for k in range(1, max_k + 1):
            nxt = []
            for r in seqs:
                for i in range(k, len(r)):
                    if all(test(v) for v in r[i - k:i]) and (i - k - 1 < 0 or not test(r[i - k - 1])):
                        nxt.append(r[i])
            s = stats(nxt)
            out[key][k] = {"n": s["n"], "mean": s["mean"], "win": s["win"]}
    allr = [v for r in seqs for v in r]
    out["all"] = {"n": len(allr), "mean": round(float(np.mean(allr)), 4) if allr else 0.0,
                  "win": round(float(np.mean(np.array(allr) > 0)), 3) if allr else 0.0}
    return out


def runs_z(r: list[float]) -> float:
    """Wald-Wolfowitz runs test on win/loss: |z| < 2 means no sign of streakiness."""
    w = (np.asarray(r) > 0).astype(int)
    n1, n0 = int(w.sum()), int(len(w) - w.sum())
    if n1 == 0 or n0 == 0:
        return 0.0
    runs = 1 + int((np.diff(w) != 0).sum())
    mu = 2 * n1 * n0 / len(w) + 1
    var = (mu - 1) * (mu - 2) / (len(w) - 1)
    return round(float((runs - mu) / math.sqrt(var)), 2) if var > 0 else 0.0


SIZING = {
    "disc_sizing_fixed": "фиксированный риск на сделку",
    "disc_sizing_cut_after_losses": "половина риска после 3 убытков подряд, до первой прибыли",
    "disc_sizing_double_after_loss": "удвоение после убытка (до ×8), «отыграться»",
    "disc_sizing_raise_after_win": "×1,5 после прибыли (до ×3), сброс после убытка",
    "disc_sizing_edge_window": "полный риск, только если сумма последних 20 сделок > 0, иначе половина",
}


def sizing_run(r: list[float], rule: str) -> dict:
    """Apply a sizing rule to one sequence of trade results (R per unit). Returns the R earned,
    the total risk taken (sum of sizes) and the drawdown, so rules can be compared per unit of risk."""
    size, losses, out, sizes = 1.0, 0, [], []
    for i, v in enumerate(r):
        if rule == "disc_sizing_fixed":
            size = 1.0
        elif rule == "disc_sizing_edge_window":
            size = 1.0 if i < 20 or sum(r[i - 20:i]) > 0 else 0.5
        sizes.append(size)
        out.append(size * v)
        if rule == "disc_sizing_cut_after_losses":
            losses = losses + 1 if v < 0 else 0
            size = 0.5 if losses >= 3 else (1.0 if v > 0 else size)
        elif rule == "disc_sizing_double_after_loss":
            size = min(8.0, size * 2) if v < 0 else 1.0
        elif rule == "disc_sizing_raise_after_win":
            size = min(3.0, size * 1.5) if v > 0 else 1.0
    s = stats(out)
    risk = float(sum(sizes))
    return {"sum": s["sum"], "risk": round(risk, 1), "per_risk": round(s["sum"] / risk, 4) if risk else 0.0,
            "dd": s["dd"], "max_size": max(sizes) if sizes else 0}


def edge_window(seqs: list[list[float]], w: int = 20) -> dict:
    """Does the mean of the last w trades predict the next one? Quintiles of the window mean."""
    X, Y = [], []
    for r in seqs:
        for i in range(w, len(r)):
            X.append(float(np.mean(r[i - w:i])))
            Y.append(r[i])
    if len(X) < 50:
        return {"n": len(X)}
    X, Y = np.array(X), np.array(Y)
    q = np.quantile(X, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(X, q)
    return {"n": len(X), "corr": round(float(np.corrcoef(X, Y)[0, 1]), 4),
            "quintiles": [round(float(Y[b == i].mean()), 4) for i in range(5)]}
