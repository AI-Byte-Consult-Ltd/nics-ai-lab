"""Smoke test for the self-learning cycle on SYNTHETIC data (results meaningless).

Runs scripts/self_learn.py twice on a synthetic snapshot: the first run has no
champion (trains from random weights and promotes), the second warm-starts a
challenger from the champion and compares both on the same data. Also checks
the forward report of shadow predictions.

  python tests/test_self_learning.py
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_trading_pipeline import make_snapshot  # noqa: E402


def check_quality_guard() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from self_learn import quality_drop, val_quality
    champ = {"signal_val_auc": 0.62, "horizon_val_acc": 0.44}
    assert quality_drop({"signal_val_auc": 0.615, "horizon_val_acc": 0.44}, champ, 0.01) == []
    assert len(quality_drop({"signal_val_auc": 0.57, "horizon_val_acc": 0.36}, champ, 0.01)) == 2
    assert quality_drop({"signal_val_auc": None, "horizon_val_acc": 0.5}, champ, 0.01) == []
    assert quality_drop({"signal_val_auc": 0.1}, {}, 0.01) == []  # no champion measures: no veto
    assert val_quality({"signal": {"note": "x"}, "horizon": {"val": {"accuracy": 0.4}}}) == \
        {"signal_val_auc": None, "horizon_val_acc": 0.4}


def main() -> None:
    check_quality_guard()
    tmp = Path(tempfile.mkdtemp())
    try:
        snap = tmp / "snap"
        snap.mkdir()
        make_snapshot(snap)
        exp = tmp / "exp"
        text = (ROOT / "configs/trading/nano.yaml").read_text()
        for a, b in (("max_steps: 3000", "max_steps: 40"), ("eval_interval: 100", "eval_interval: 20"),
                     ("finetune_steps: 1500", "finetune_steps: 20"), ("keep_snapshots: 3", "keep_snapshots: 1000"),
                     ("min_signal_samples: 50", "min_signal_samples: 10"),
                     ("min_filtered_trades: 20", "min_filtered_trades: 5"),
                     ("experiments_dir: experiments/trading/nics-trading-nano", f"experiments_dir: {exp}")):
            assert a in text, a
            text = text.replace(a, b)
        cfg = tmp / "c.yaml"
        cfg.write_text(text)
        cmd = [sys.executable, str(ROOT / "scripts/self_learn.py"), "--config", str(cfg),
               "--skip-fetch", "--snapshot", str(snap)]
        subprocess.run(cmd, check=True)
        first = json.loads((exp / "self_learn_last.json").read_text())
        assert first["promoted"] and first["champion_now"], first
        # a shadow prediction made "before" the outcomes, to exercise the forward report
        ids = [r["id"] for r in csv.DictReader(open(snap / "signals.csv"))][:50]
        (exp / "shadow").mkdir(exist_ok=True)
        with open(exp / "shadow" / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["signal_id", "p_tp1", "scored_at"])
            for sid in ids:
                w.writerow([sid, 0.5, "2025-12-31T00:00:00+00:00"])
        subprocess.run(cmd, check=True)
        second = json.loads((exp / "self_learn_last.json").read_text())
        assert any(c["kind"] == "warm" for c in second["challengers"]), second
        assert second["champion"]["forward_shadow"]["n"] > 0, second["champion"]
        print("SELF-LEARNING SMOKE TEST PASSED (synthetic data, results meaningless)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
