"""Offline test of the replay edge search (src/nics/market/replay_edge.py).

SYNTHETIC trades, results meaningless. Checks: a planted condition is found on unseen years,
pure noise is not, and shuffled targets find nothing even when a condition exists.

  python tests/test_replay_edge.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import replay_edge as re_  # noqa: E402


def rows(n: int, seed: int, planted: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    t0 = datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp()
    out = []
    for k in range(n):
        t = t0 + k * (4 * 365.25 * 86400 / n)
        ext = float(rng.normal())
        r = float(rng.normal(-0.05, 1.0)) + (0.6 if planted and ext > 0.8 else 0.0)
        D = {k2: float(rng.normal()) for k2 in ("lt", "c200", "c50", "c20", "s20", "m20", "m60", "m120")}
        D.update(ext=ext, pos20=float(rng.uniform()), atrp=1.0)
        out.append({"t": t, "sym": "XAU/USD", "d": "LONG", "c": "BASE", "su": "MOMENTUM", "q": "STRONG", "mf": "UP", "ms": "FLAT",
                    "htf": "UP", "hs": "M15_ONLY", "rg": "TRENDING", "es": "EARLY", "ch": None, "cf": 50, "rkp": 0.004, "sp": 0.05,
                    "D": D, "v": {name: [r, "T"] for name in re_.TARGETS}})
    return out


def main() -> None:
    assert len(re_.features(rows(1, 0, False)[0])) == len(re_.feature_names())
    good = rows(16000, 1, True)
    res = re_.walk_forward(good, "base")
    assert res["found"] and res["selected_total"]["mean"] > 0.2, res["selected_total"]
    assert not re_.walk_forward(good, "base", shuffle_seed=7)["found"]
    noise = re_.walk_forward(rows(16000, 2, False), "base")
    assert not noise["found"], noise["selected_total"]
    assert not re_.walk_forward(rows(16000, 2, False), "base", model="ridge")["found"]
    print("OK replay_edge")


if __name__ == "__main__":
    main()
