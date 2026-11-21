"""Offline test of candidate rules in shadow (src/nics/market/rules.py, scripts/rules_shadow.py).

Checks the honest fill / outcome logic on hand-made bars, then runs the nightly
measurement on a SYNTHETIC snapshot (results meaningless).

  python tests/test_rules_shadow.py
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from nics.market import methods, registry, rules  # noqa: E402
from test_trading_pipeline import make_snapshot  # noqa: E402


BANNED = {'3f0dd39a35a47bab', '836c0a14b3a77351', '4f4418c4b53f4e3a', 'd6861e054ba9f4be', '0c915ba83dbac8a9', 'ca700c8aa0b8566b', 'e57571d905508b57', '27a0f645e20497a6', '4bad2eaec5cd6571', 'a97b73a0db48a29e', '864691f8a013e0b2', '6d50a53fe9365bde', 'f2bec3a0f8b2235a', '664a654a76f954df', '6f166e778d3e08c0', '175a680792c1fbf7', '4d48b51a3deb7a4e'}


class S:  # minimal series
    def __init__(self, h, l):
        self.h, self.l = np.array(h, float), np.array(l, float)
        self.c = (self.h + self.l) / 2


def main() -> None:
    st = {"direction": 1, "bar": 0, "entry": 10.0, "sl": 9.0, "tp1": 12.0, "valid_bars": 2}
    # never trades through the buy stop -> no fill
    assert rules.outcome(S([9.5, 9.9, 9.9, 13], [9.4, 9.5, 9.5, 9.5]), st, 10)[0] is False
    # fills on bar 1, target on bar 2
    assert rules.outcome(S([9.5, 10.2, 12.5], [9.4, 9.6, 10.0]), st, 10) == (True, 1, 1, 2)
    # fill bar touches the stop -> loss, even if the target is hit later
    assert rules.outcome(S([9.5, 10.2, 12.5], [9.4, 8.9, 10.0]), st, 10) == (True, 0, 1, 1)
    # target inside the fill bar does not count (order inside the bar unknown)
    assert rules.outcome(S([9.5, 12.5, 11.0], [9.4, 9.6, 9.6]), st, 10) == (True, None, 1, None)
    # short mirror
    sh = {"direction": -1, "bar": 0, "entry": 10.0, "sl": 11.0, "tp1": 8.0, "valid_bars": 2}
    assert rules.outcome(S([10.5, 10.4, 10.0], [10.2, 9.8, 7.9]), sh, 10) == (True, 1, 1, 2)
    # the registry matches the code: methods in training, candidates in shadow
    reg = {r["id"]: r for r in registry.REGISTRY}
    assert {k for k, r in reg.items() if r["status"] == "training"} == set(methods.METHODS), reg
    assert {k for k, r in reg.items() if r["status"] == "shadow"} == \
        {k for k, m in rules.CANDIDATES.items() if m[3] == "shadow"}
    assert all(r["status"] in registry.STATUS_RU for r in registry.REGISTRY)
    words = set(re.findall(r"[a-zа-яё]+", json.dumps(registry.summary(), ensure_ascii=False).lower()))
    assert not any(hashlib.sha256(w.encode()).hexdigest()[:16] in BANNED for w in words), "name in registry"
    for rid, meta in rules.CANDIDATES.items():
        assert meta[0] in vars(rules) and meta[3] in ("shadow", "promoted", "rejected"), rid

    tmp = Path(tempfile.mkdtemp())
    try:
        make_snapshot(tmp)
        text = (ROOT / "configs/trading/nano.yaml").read_text()
        text = text.replace("experiments_dir: experiments/trading/nics-trading-nano", f"experiments_dir: {tmp / 'exp'}")
        (tmp / "c.yaml").write_text(text)
        out = subprocess.run([sys.executable, str(ROOT / "scripts/rules_shadow.py"), "--config", str(tmp / "c.yaml"),
                              "--snapshot", str(tmp)], check=True, capture_output=True, text=True).stdout
        summary = json.loads(out.strip().splitlines()[-1])
        assert set(summary) == set(rules.CANDIDATES), summary
        for r in summary.values():
            assert r["broker_before"]["setups"] + r["backtest_public"]["setups"] > 0, r
        # internal names only: no author / school / company names in what the rules output.
        # The list is kept as hashes so the names themselves are not written in the repository.
        low = (out + json.dumps(rules.CANDIDATES, ensure_ascii=False)).lower()
        words = set(re.findall(r"[a-zа-яё]+", low))
        assert not any(hashlib.sha256(w.encode()).hexdigest()[:16] in BANNED for w in words), \
            "a person's, school's or company's name leaked into rule outputs"
        r = summary["rule_level_squeeze_breakout"]
        assert (tmp / "exp" / "shadow" / "rules_last.txt").exists()
        rep = json.loads((tmp / "exp" / "shadow" / "rules" / "rule_strong_trend_pullback.json").read_text())
        assert {"forward", "broker_before", "backtest_public", "forward_by_symbol"} <= set(rep)
        print("OK:", json.dumps(r))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
