"""Offline test of the discipline research and the add-on setups (results meaningless).

  python tests/test_discipline.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import data as market  # noqa: E402
from nics.market import discipline as dsc  # noqa: E402
from nics.market import methods  # noqa: E402


class Ser:
    def __init__(self, h, l, c):  # noqa: E741
        self.h, self.l, self.c = np.array(h, float), np.array(l, float), np.array(c, float)
        self.t = np.arange(len(c)) * 3600.0
        self.atr = np.ones(len(c))
        self.symbol, self.interval, self.bar_seconds = "X", "1h", 3600


def bars(o, h, l, c, spread=0.0):  # noqa: E741
    n = len(o)
    return dsc.Bars(np.arange(n) * 3600.0, np.array(o, float), np.array(h, float), np.array(l, float),
                    np.array(c, float), "X", spread)


def main() -> None:
    # fixed: stop checked before take on the same bar -> loss
    b = bars([100, 100], [103, 103], [98, 98], [100, 100])
    assert dsc.simulate(b, 0, 1, 1.0, "fixed")["r"] == -1.0
    # fixed: take at +1 R
    b = bars([100, 100.5], [100.6, 101.2], [99.5, 100.2], [100.5, 101])
    assert dsc.simulate(b, 0, 1, 1.0, "fixed")["r"] == 1.0
    # gap through the stop fills at the open
    b = bars([100, 98], [100.2, 98.1], [99.6, 97.5], [99.8, 97.8])
    assert abs(dsc.simulate(b, 0, 1, 1.0, "fixed")["r"] + 2.0) < 1e-9
    # pyramid: adds at +1 R and +2 R, stop moves to the previous entry; reversal stops at 101
    b = bars([100, 100.5, 101.5, 102.2, 101.5], [100.6, 101.6, 102.6, 102.3, 101.6],
             [99.8, 100.4, 101.4, 101.9, 100.9], [100.5, 101.5, 102.2, 102.0, 101.0])
    r = dsc.simulate(b, 0, 1, 1.0, "pyramid")
    assert r["units"] == 3, r
    # the trail (best close 102.2 - 1) is above the 101 add-stop: units 100, 101, 102 close at 101.2
    assert abs(r["r"] - 0.6) < 1e-9, r
    # increasing sizes 1/3, 2/3, 1: same exit -> 1/3*1.2 + 2/3*0.2 + 1*(-0.8)
    r = dsc.simulate(b, 0, 1, 1.0, "pyramid_up")
    assert abs(r["r"] - (0.4 + 0.2 * 2 / 3 - 0.8)) < 1e-9, r
    # short mirror of a take
    b = bars([100, 99.5], [100.4, 99.8], [99.4, 98.8], [99.5, 99])
    assert dsc.simulate(b, 0, -1, 1.0, "fixed")["r"] == 1.0
    # spread is paid per unit
    b = bars([100, 100.5], [100.6, 101.2], [99.5, 100.2], [100.5, 101], spread=0.001)
    assert abs(dsc.simulate(b, 0, 1, 1.0, "fixed")["r"] - 0.9) < 1e-9
    # streaks / sizing / edge window
    st = dsc.streak_table([[-1, -1, 1, -1, -1, -1, 1, 1]], 3)
    assert st["after_losses"][2]["n"] == 2 and st["after_losses"][3]["n"] == 1
    assert dsc.sizing_run([-1, -1, 1], "disc_sizing_double_after_loss")["sum"] == 1.0  # -1 -2 +4
    assert dsc.sizing_run([1, 1, -1], "disc_sizing_raise_after_win")["max_size"] == 2.25
    assert dsc.runs_z([1, -1] * 20) > 2  # alternating = too many runs
    # add-on setups: +1 R before the stop -> a second entry with the stop at the first entry
    s = Ser([10.2, 10.6, 11.2, 11.5], [9.8, 10.1, 10.5, 11.0], [10.0, 10.5, 11.1, 11.2])
    first = {"id": "m-1", "kind": "method_x", "symbol": "X", "direction": 1, "entry": 10.0, "sl": 9.0, "tp1": 12.0, "bar": 0}
    ad = methods.add_on_winners(s, [first])
    assert len(ad) == 1 and ad[0]["bar"] == 2 and ad[0]["sl"] == 10.0 and abs(ad[0]["tp1"] - 12.2) < 1e-9, ad
    assert ad[0]["t"] == s.t[2] + 3600  # decided when bar 2 has closed
    # stopped before +1 R -> no add-on
    s2 = Ser([10.2, 10.3, 11.5], [9.8, 8.9, 10.5], [10.0, 9.2, 11.1])
    assert methods.add_on_winners(s2, [first]) == []
    # schema switch: off = the old feature list exactly; on adds one kind; a lock wins over config
    before = list(market.SIGNAL_FEATURES)
    market.use_add_ons(True)
    assert market.SIGNAL_FEATURES == before[:before.index("strategy_NICS-CASCADE-2026-08-v2")] + \
        [f"kind_{market.ADDON_KIND}"] + before[before.index("strategy_NICS-CASCADE-2026-08-v2"):]
    market.use_add_ons(False, lock=True)
    market.use_add_ons(True)  # ignored: locked
    assert market.SIGNAL_FEATURES == before and not market.add_ons_on()
    print("DISCIPLINE TEST PASSED")


if __name__ == "__main__":
    main()
