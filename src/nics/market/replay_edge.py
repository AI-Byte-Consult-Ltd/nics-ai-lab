"""Search for conditions with an edge in the multi-year scanner replay (research only).

Nothing here sends a signal or changes the scanner or any EA.

Data (2026-11-26): the live scanner code was replayed every 3 minutes on public minute history
2022-2026 for 12 instruments (gold, silver, Brent, six indices, BTC, ETH, SOL). Each replayed
signal became one simulated trade (market entry next minute, stop and first take from the
signal, spread on the ask side) with several entry / stop constructions. One row per trade in
`trades/<file>.jsonl`. On the whole set a trade loses about one spread (-0.05 R) in both
directions, every year; a mirrored trade loses the same. So the question for the model is not
"is the signal good" (it is not) but "is there any condition, known at entry, under which the
trades are positive on years the model has not seen".

Honesty rules:
- features are only what the scanner and completed daily bars knew at entry;
- walk-forward by calendar year: fit on all years before Y, pick the score threshold on those
  years, judge only on year Y; the same for each Y;
- a condition counts as found only when the selected trades are positive after spread with
  t >= 2 on the unseen years taken together AND positive in at least two of the three unseen
  years. Anything less is reported as "not found";
- the control: the same procedure on shuffled targets must not find anything.

Model: gradient boosting of small regression trees on R, written here in numpy (no third-party
ML library), plus plain ridge regression as a second opinion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GROUP = {"XAU/USD": "metals", "XAG/USD": "metals", "UKOUSDft": "oil", "USOUSD": "oil", "BTC/USD": "crypto",
         "ETH/USD": "crypto", "SOLUSD": "crypto"}
SETUPS = ("MOMENTUM", "MEAN_REVERSION", "BREAKOUT")
SOURCES = ("M15_H1_AGREE", "M15_ONLY", "H1_ONLY", "NONE")
STAGES = ("EARLY", "LATE", "OTHER")
CLASSES = ("BASE", "L_MACRO", "L_CHASE", "L_BOTH")
GROUPS = ("metals", "oil", "indices", "crypto", "forex")
BIAS = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0, "NEUTRAL": 0.0}
TARGETS = ("base", "sl20", "d30", "flip")
TARGET_RU = {"base": "как сейчас (стоп и первый тейк)", "sl20": "стоп в 2 раза шире", "d30": "вход через 30 минут",
             "flip": "сделка наоборот сигналу"}


def group_of(sym: str) -> str:
    return GROUP.get(sym) or ("forex" if "/" in sym else "indices")


def load_trades(folder: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(folder).glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "D" in r and all(k in r.get("v", {}) for k in TARGETS):
                    rows.append(r)
    rows.sort(key=lambda r: r["t"])
    return rows


def feature_names() -> list[str]:
    names = ["long", "confidence", "risk_pct", "spread_r", "chase", "chase_known", "macro_fast_with", "macro_slow_with",
             "htf_with", "hour_sin", "hour_cos", "weekday", "d1_longterm_with", "d1_above_ema200_with", "d1_above_ema50_with",
             "d1_above_ema20_with", "d1_ema20_slope_with", "d1_mom20_with", "d1_mom60_with", "d1_mom120_with",
             "d1_extension_with", "d1_range_pos_with", "d1_atr_pct", "relaxed"]
    names += [f"setup_{s}" for s in SETUPS] + [f"source_{s}" for s in SOURCES] + [f"stage_{s}" for s in STAGES]
    names += [f"class_{s}" for s in CLASSES] + [f"group_{s}" for s in GROUPS] + ["regime_trending"]
    return names


def features(r: dict) -> list[float]:
    d = 1.0 if r["d"] == "LONG" else -1.0
    D = r["D"]
    hour = (r["t"] % 86400) / 3600.0
    ch = r.get("ch")
    pos = float(D["pos20"])
    x = [1.0 if d > 0 else 0.0, float(r.get("cf") or 0), float(r["rkp"]) * 100, float(r["sp"]),
         float(ch) if ch is not None else 0.0, 1.0 if ch is not None else 0.0,
         d * BIAS.get(r.get("mf") or "", 0.0), d * BIAS.get(r.get("ms") or "", 0.0), d * BIAS.get(r.get("htf") or "", 0.0),
         math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24), float(int(r["t"] // 86400 + 4) % 7),
         d * D["lt"], d * D["c200"], d * D["c50"], d * D["c20"], d * D["s20"], d * D["m20"], d * D["m60"], d * D["m120"],
         d * D["ext"], pos if d > 0 else 1.0 - pos, float(D["atrp"]), 1.0 if r.get("q") == "RELAXED" else 0.0]
    x += [1.0 if r.get("su") == s else 0.0 for s in SETUPS]
    x += [1.0 if r.get("hs") == s else 0.0 for s in SOURCES]
    x += [1.0 if r.get("es") == s else 0.0 for s in STAGES]
    x += [1.0 if r.get("c") == s else 0.0 for s in CLASSES]
    g = group_of(r["sym"])
    x += [1.0 if g == s else 0.0 for s in GROUPS]
    x += [1.0 if r.get("rg") == "TRENDING" else 0.0]
    return x


def design(rows: list[dict]) -> np.ndarray:
    return np.array([features(r) for r in rows], dtype=np.float64)


def target(rows: list[dict], name: str) -> np.ndarray:
    return np.clip(np.array([r["v"][name][0] for r in rows], dtype=np.float64), -3.0, 5.0)


def year_of(t: float) -> int:
    return 1970 + int(np.datetime64(int(t), "s").astype("datetime64[Y]").astype(int))


# ---------------------------------------------------------------- gradient boosting (numpy)

class Boost:
    """Least-squares gradient boosting of depth-limited trees on quantile-binned features."""

    def __init__(self, trees: int = 120, depth: int = 3, rate: float = 0.05, bins: int = 24, min_leaf: int = 300,
                 subsample: float = 0.7, seed: int = 0):
        self.trees, self.depth, self.rate, self.bins, self.min_leaf = trees, depth, rate, bins, min_leaf
        self.subsample, self.seed = subsample, seed
        self.edges: list[np.ndarray] = []
        self.model: list[list[tuple]] = []
        self.base = 0.0

    def _bin(self, X: np.ndarray) -> np.ndarray:
        return np.stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges)], axis=1).astype(np.int32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Boost":
        rng = np.random.default_rng(self.seed)
        qs = np.linspace(0, 1, self.bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        B = self._bin(X)
        self.base = float(y.mean())
        pred = np.full(len(y), self.base)
        for _ in range(self.trees):
            idx = np.flatnonzero(rng.random(len(y)) < self.subsample)
            tree = self._tree(B[idx], (y - pred)[idx])
            self.model.append(tree)
            pred += self.rate * self._apply(tree, B)
        return self

    def _tree(self, B: np.ndarray, r: np.ndarray) -> list[tuple]:
        nodes: list[tuple] = []

        def grow(rows: np.ndarray, depth: int) -> int:
            me = len(nodes)
            nodes.append(None)  # type: ignore[arg-type]
            val = float(r[rows].mean()) if len(rows) else 0.0
            best = None
            if depth < self.depth and len(rows) >= 2 * self.min_leaf:
                tot, n = r[rows].sum(), len(rows)
                for j in range(B.shape[1]):
                    nb = len(self.edges[j]) + 1
                    if nb < 2:
                        continue
                    b = B[rows, j]
                    cnt = np.bincount(b, minlength=nb).cumsum()[:-1]
                    sm = np.bincount(b, weights=r[rows], minlength=nb).cumsum()[:-1]
                    ok = (cnt >= self.min_leaf) & (n - cnt >= self.min_leaf)
                    if not ok.any():
                        continue
                    gain = np.where(ok, sm ** 2 / np.maximum(cnt, 1) + (tot - sm) ** 2 / np.maximum(n - cnt, 1), -np.inf)
                    k = int(gain.argmax())
                    if best is None or gain[k] > best[0]:
                        best = (float(gain[k]), j, k)
            if best is None:
                nodes[me] = ("leaf", val)
                return me
            _, j, k = best
            left = grow(rows[B[rows, j] <= k], depth + 1)
            right = grow(rows[B[rows, j] > k], depth + 1)
            nodes[me] = ("split", j, k, left, right)
            return me

        grow(np.arange(len(r)), 0)
        return nodes

    @staticmethod
    def _apply(tree: list[tuple], B: np.ndarray) -> np.ndarray:
        out = np.zeros(len(B))

        def walk(i: int, rows: np.ndarray) -> None:
            node = tree[i]
            if node[0] == "leaf":
                out[rows] = node[1]
                return
            _, j, k, left, right = node
            m = B[rows, j] <= k
            walk(left, rows[m])
            walk(right, rows[~m])

        walk(0, np.arange(len(B)))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        B = self._bin(X)
        pred = np.full(len(X), self.base)
        for tree in self.model:
            pred += self.rate * self._apply(tree, B)
        return pred

    def importance(self, n_features: int) -> np.ndarray:
        imp = np.zeros(n_features)
        for tree in self.model:
            for node in tree:
                if node[0] == "split":
                    imp[node[1]] += 1
        return imp / max(1.0, imp.sum())


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 50.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    reg = lam * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    return lambda A: np.column_stack([(A - mu) / sd, np.ones(len(A))]) @ w


# ---------------------------------------------------------------- evaluation

def stat(r: np.ndarray, t: np.ndarray | None = None) -> dict:
    n = len(r)
    if n == 0:
        return {"n": 0, "sum": 0.0, "mean": 0.0, "t": 0.0, "t_day": 0.0}
    m = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    out = {"n": n, "sum": round(float(r.sum()), 1), "mean": round(m, 4), "t": round(m / (sd / math.sqrt(n)), 2) if sd > 0 else 0.0}
    if t is not None and n > 1:  # trades cluster in time: t over daily sums
        days, inv = np.unique((t // 86400).astype(np.int64), return_inverse=True)
        ds = np.bincount(inv, weights=r)
        s = float(ds.std(ddof=1)) if len(ds) > 1 else 0.0
        out["t_day"] = round(float(ds.mean()) / (s / math.sqrt(len(ds))), 2) if s > 0 else 0.0
    return out


def walk_forward(rows: list[dict], tname: str, top: float = 0.2, first_test_year: int | None = None, shuffle_seed: int | None = None,
                 model: str = "boost") -> dict:
    """Fit on years < Y, keep trades whose score is above the train (1 - top) quantile, judge on Y."""
    X, y = design(rows), target(rows, tname)
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    yrs = np.array([year_of(v) for v in t])
    years = sorted(set(yrs.tolist()))
    if first_test_year is None:
        first_test_year = years[2] if len(years) > 2 else years[-1]
    sel_r, sel_t, all_r, low_r, per_year, imp = [], [], [], [], {}, np.zeros(X.shape[1])
    for Y in [v for v in years if v >= first_test_year]:
        tr, te = yrs < Y, yrs == Y
        if tr.sum() < 2000 or te.sum() < 200:
            continue
        ytr = y[tr].copy()
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed + Y).shuffle(ytr)
        if model == "boost":
            m = Boost(seed=Y).fit(X[tr], ytr)
            s_tr, s_te = m.predict(X[tr]), m.predict(X[te])
            imp += m.importance(X.shape[1])
        else:
            f = ridge_fit(X[tr], ytr)
            s_tr, s_te = f(X[tr]), f(X[te])
        hi, lo = np.quantile(s_tr, 1 - top), np.quantile(s_tr, top)
        pick, worst = s_te >= hi, s_te <= lo
        per_year[str(Y)] = {"all": stat(y[te], t[te]), "selected": stat(y[te][pick], t[te][pick]), "lowest": stat(y[te][worst], t[te][worst]),
                            "train_selected_mean": round(float(ytr[s_tr >= hi].mean()), 4)}
        sel_r.append(y[te][pick]); sel_t.append(t[te][pick]); all_r.append(y[te]); low_r.append(y[te][worst])
    if not sel_r:
        return {"note": "too few trades for a walk-forward check"}
    sr, st = np.concatenate(sel_r), np.concatenate(sel_t)
    total = stat(sr, st)
    pos_years = sum(1 for v in per_year.values() if v["selected"]["mean"] > 0 and v["selected"]["n"] >= 30)
    found = bool(total["mean"] > 0 and total["t"] >= 2 and total.get("t_day", 0) >= 2 and pos_years >= min(2, len(per_year)))
    out = {"target": tname, "model": model, "top_share": top, "years": per_year, "selected_total": total,
           "all_total": stat(np.concatenate(all_r)), "lowest_total": stat(np.concatenate(low_r)), "positive_years": pos_years,
           "found": found}
    if model == "boost" and imp.sum() > 0:
        names = feature_names()
        order = np.argsort(-imp)[:8]
        out["top_features"] = [[names[i], round(float(imp[i] / imp.sum()), 3)] for i in order]
    return out


def baseline(rows: list[dict]) -> dict:
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    out = {}
    for name in TARGETS:
        y = target(rows, name)
        out[name] = {"all": stat(y, t), "long": stat(y[[r["d"] == "LONG" for r in rows]]), "short": stat(y[[r["d"] == "SHORT" for r in rows]])}
    return out
