"""Daily trend context (src/nics/market/trend_context.py) and the methods research engine.

Checks: states are causal (future bars never change past states), an intraday bar sees only
daily candles that had closed by its own close, the cache round-trips, the backtester's fill /
stop / gap rules, and neutral names only.

  python tests/test_trend_context.py
"""
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from nics.market import classic_methods as cm  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.market import registry, trend_context as tc  # noqa: E402
from test_rules_shadow import BANNED  # noqa: E402

DAY = 86400


def walk(n, seed=1, start=1.0e9):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    t = start + DAY * np.arange(n)
    return t, o, h, l, c


def main() -> None:
    t, o, h, l, c = walk(700)
    st = tc.daily_states(h, l, c)
    assert st.shape == (700, len(tc.FEATURE_METHODS)) and set(np.unique(st)) <= {-1.0, 0.0, 1.0}
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    c2[500:] *= 1.5; h2[500:] *= 1.5; l2[500:] *= 1.5
    assert np.array_equal(tc.daily_states(h2, l2, c2)[:500], st[:500]), "future bars changed past states"
    assert (st[:251, 0] == 0).all() and (st[260:, 0] != 0).any(), "momentum needs 252 days"

    d1 = market.Series("X", "1day", True, t, o, h, l, c, np.zeros(700), np.zeros(700))
    hrs = t[600] + 3600 * np.arange(48)  # two days of hourly bars
    hx = np.full(48, c[600])
    h1 = market.Series("X", "1h", False, hrs, hx, hx, hx, hx, np.zeros(48), np.zeros(48))
    ctx = tc.build([d1, h1])
    tc.attach([d1, h1], ctx)
    # hourly bars of day 600 (closing before day 600 closes) see day 599; from the day-600 close on, day 600
    assert np.array_equal(h1.htf[0, :-1], st[599]) and np.array_equal(h1.htf[22, :-1], st[599])
    assert np.array_equal(h1.htf[23, :-1], st[600]) and np.array_equal(h1.htf[47, :-1], st[601])
    assert np.array_equal(d1.htf[10, :-1], st[10]), "a daily bar sees its own close"
    assert np.allclose(h1.htf[:, -1], h1.htf[:, :-1].mean(axis=1))
    # off by default: the schema of older models is unchanged
    assert market.BAR_FEATURES == market.BAR_FEATURES_CORE and market.bar_features(h1)[0].shape[1] == len(market.BAR_FEATURES_CORE)
    market.use_trend_context(True)
    f, _ = market.bar_features(h1)
    assert f.shape[1] == len(market.BAR_FEATURES) and market.BAR_FEATURES[-len(tc.TREND_FEATURES):] == tc.TREND_FEATURES
    assert np.array_equal(f[:, -len(tc.TREND_FEATURES):], h1.htf.astype(np.float32))
    # a loaded model locks the schema it was trained with; config calls do not override it
    from nics.training import trading as T
    T.lock_schema_from({"bar_features": list(market.BAR_FEATURES_CORE)})
    market.use_trend_context(True)
    assert not market.trend_context_on() and market.BAR_FEATURES == market.BAR_FEATURES_CORE

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ctx.csv"
        tc.save(ctx, p, keep_days=200)
        back = tc.load(p)
        assert np.array_equal(back["X"][1], ctx["X"][1][-200:]) and np.array_equal(back["X"][0], ctx["X"][0][-200:])
        # live: no daily candles in the snapshot -> the cache is used
        h1b = market.Series("X", "1h", False, hrs, hx, hx, hx, hx, np.zeros(48), np.zeros(48))
        tc.prepare([h1b], save_to=None, load_from=p)
        assert np.array_equal(h1b.htf[47, :-1], h1.htf[47, :-1])

    # backtester: entry at the next open, a gap through the stop exits at the open (worse than the stop)
    n = 40
    o = np.full(n, 10.0); o[33:] = 8.0
    d = {"t": np.arange(n) * float(DAY), "o": o, "h": o + 0.1, "l": o - 0.1, "c": o.copy()}
    ent = np.zeros(n, int); ent[30] = 1
    tr = cm.simulate(d, {"ent": ent, "stop_dist": np.full(n, 1.0)}, 0.0, 0.0, warm=20)
    assert len(tr) == 1 and tr[0]["why"] == "stop" and abs(tr[0]["r"] - (-2.0)) < 1e-9, tr
    # a position still open at the end is marked at the last close
    tr = cm.simulate(d, {"ent": ent, "stop_dist": np.full(n, 5.0)}, 0.0, 0.0, warm=20)
    assert tr[-1]["why"] == "open_marked" and abs(tr[-1]["r"] - (-0.4)) < 1e-9, tr
    # costs: spread share of price and swap per day held, both in R
    tr = cm.simulate(d, {"ent": ent, "stop_dist": np.full(n, 5.0)}, 0.01, 0.365, warm=20)
    assert abs(tr[-1]["cost_r"] - (0.01 * 10 + 0.365 * 10 * 8 / 365) / 5) < 1e-9, tr

    words = set(re.findall(r"[a-zа-яё]+", (json.dumps(tc.FEATURE_METHODS) + json.dumps(registry.summary(), ensure_ascii=False)
                                         + " ".join(cm.NEW) + " ".join(cm.OURS)).lower()))
    assert not any(hashlib.sha256(w.encode()).hexdigest()[:16] in BANNED for w in words), "name found"
    feat = {r["id"] for r in registry.REGISTRY if r["status"] == "feature"}
    assert feat == set(tc.FEATURE_METHODS), feat
    print("TREND CONTEXT TEST PASSED")


if __name__ == "__main__":
    main()
