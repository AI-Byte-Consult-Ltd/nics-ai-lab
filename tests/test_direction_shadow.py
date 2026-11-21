"""Smoke test of the direction shadow test on SYNTHETIC data (results meaningless).

Trains a tiny model, lets it pick its own direction for every scanner signal
(score_live.score_directions, signal vs its mirror), then runs the daily
comparison scripts/direction_report.py.

  python tests/test_direction_shadow.py
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from nics.common.config import load_config  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.training import trading as T  # noqa: E402
import score_live  # noqa: E402
from test_trading_pipeline import make_snapshot  # noqa: E402


def main() -> None:
    st = {"id": "x", "kind": "scanner_scheduled", "direction": 1, "entry": 100.0, "sl": 99.0, "tp1": 102.0,
          "tp2": 103.0, "tp3": None}
    m = market.mirror_setup(st)
    assert (m["direction"], m["sl"], m["tp1"], m["tp2"], m["id"]) == (-1, 101.0, 98.0, 97.0, "x|mirror"), m
    tmp = Path(tempfile.mkdtemp())
    try:
        make_snapshot(tmp)
        with open(tmp / "signals.csv", newline="") as f:  # give the signals plan groups
            rows = list(csv.DictReader(f))
        for k, r in enumerate(rows):
            r["plan_group"] = "gold" if k % 2 else "forex"
            r["delivery_status"] = "SENT" if k % 3 == 0 else "READY"
        with open(tmp / "signals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        text = (ROOT / "configs/trading/nano.yaml").read_text()
        text = text.replace("max_steps: 3000", "max_steps: 30").replace("eval_interval: 100", "eval_interval: 10")
        text = text.replace("experiments_dir: experiments/trading/nics-trading-nano", f"experiments_dir: {tmp / 'exp'}")
        text = text.replace("min_signal_samples: 50", "min_signal_samples: 10")
        (tmp / "c.yaml").write_text(text)
        subprocess.run([sys.executable, str(ROOT / "scripts/train_trading.py"), "--config", str(tmp / "c.yaml"),
                        "--snapshot", str(tmp), "--experiment-id", "d"], check=True, capture_output=True)
        cfg = load_config(str(tmp / "c.yaml"))
        model, man = T.load_release(tmp / "exp" / "d" / "release", cfg)
        series, _, S, _ = T.build_samples(cfg, tmp, use_public=False, label_setups=False, mirror_scanner=True)
        win = T.Windows(series, cfg.data.window)
        dpath = tmp / "direction.csv"
        score_live.score_directions(model, man, win, series, S, dpath, 1e9, datetime.now(timezone.utc))
        preds = list(csv.DictReader(open(dpath, newline="")))
        assert len(preds) > 100 and {"LONG", "SHORT"} >= {p["model_direction"] for p in preds}, len(preds)
        out = subprocess.run([sys.executable, str(ROOT / "scripts/direction_report.py"), "--config",
                              str(tmp / "c.yaml"), "--snapshot", str(tmp), "--direction", str(dpath),
                              "--days", "800"], check=True, capture_output=True, text=True).stdout
        last = json.loads(out.strip().splitlines()[-1])
        assert last["resolved"] > 0 and last["agree"] + last["disagree"] == last["resolved"], last
        assert last["ideas"]["signals"] >= last["signals"], last  # headline = sent only
        reps = sorted((tmp / "exp" / "shadow" / "direction_reports").glob("*.json"))
        rep = json.loads(reps[-1].read_text(encoding="utf-8"))
        assert set(rep["sent_by_asset_class"]) <= {"gold", "forex"}, rep["sent_by_asset_class"].keys()
        assert rep["all_records_raw"]["signals"] >= rep["all_unique_ideas"]["signals"] >= rep["total"]["signals"]
        assert (tmp / "exp" / "shadow" / "direction_last.txt").exists()
        print("\n".join(out.strip().splitlines()[:-1]))
        print("DIRECTION SHADOW TEST PASSED (synthetic data, results meaningless)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
