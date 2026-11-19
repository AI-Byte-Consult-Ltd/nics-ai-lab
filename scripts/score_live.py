"""Shadow scoring: the champion model scores NEW scanner signals as they appear.

Nothing is sent to users, the bot or any account. Each score is appended to
experiments/trading/<model_id>/shadow/predictions.csv with the time it was made,
so the self-learning cycle can later check it against the real outcome
(a true forward test, see scripts/eval_trading.py).

  python scripts/score_live.py --config configs/trading/nano.yaml
  (needs NICS_DATA_URL / NICS_DATA_KEY; run hourly by scripts/windows/install_tasks.bat)
"""
import argparse
import copy
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import Config, load_config  # noqa: E402
from nics.training import trading as T  # noqa: E402

COLS = ["signal_id", "symbol", "kind", "direction", "created_at", "p_tp1", "take", "threshold",
        "experiment_id", "scored_at"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/trading/nano.yaml"))
    ap.add_argument("--hours", type=float, default=6, help="score signals created in the last N hours")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = ROOT / cfg.paths.experiments_dir
    champ = base / "champion"
    if not (champ / "model.safetensors").exists():
        sys.exit("No champion yet: run scripts/self_learn.py once first.")
    model, man = T.load_release(champ, cfg)
    metrics = champ / "metrics.json"
    import json
    thr = (json.loads(metrics.read_text()).get("signal") or {}).get("threshold_from_val") if metrics.exists() else None

    live = ROOT / "data" / "live"
    sig_iv = cfg.data.signal_interval
    subprocess.run([sys.executable, str(ROOT / "scripts/fetch_market_data.py"), "--intervals", sig_iv,
                    "--days", "6", "--tables", "signals", "--out", str(live)], check=True, env=os.environ)

    lcfg = Config(copy.deepcopy(dict(cfg)))
    lcfg["data"]["intervals"] = {sig_iv: dict(cfg.data.intervals)[sig_iv]}
    lcfg["methods"] = {"enabled": False}
    series, H, S, _ = T.build_samples(lcfg, live, use_public=False, label_setups=False)
    win = T.Windows(series, cfg.data.window)

    out_dir = base / "shadow"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    done = set()
    if pred_path.exists():
        with open(pred_path, newline="", encoding="utf-8") as f:
            done = {r["signal_id"] for r in csv.DictReader(f)}
    now = datetime.now(timezone.utc)
    recent = np.array([(k.startswith("scanner") and sid not in done and now.timestamp() - t <= args.hours * 3600)
                       for k, sid, t in zip(S["kind"], S["id"], S["t0"])], dtype=bool)
    if not recent.any():
        print("No new signals to score.")
        return
    p = T.predict_signal(model, win, S, recent)
    new = not pred_path.exists()
    with open(pred_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for j, prob in zip(np.flatnonzero(recent), p):
            s = series[int(S["sid"][j])]
            row = {"signal_id": S["id"][j], "symbol": s.symbol, "kind": S["kind"][j],
                   "direction": "LONG" if S["f"][j][0] > 0 else "SHORT",
                   "created_at": T.iso(S["t0"][j]), "p_tp1": round(float(prob), 4),
                   "take": "" if thr is None else int(prob >= thr), "threshold": "" if thr is None else thr,
                   "experiment_id": man.get("experiment_id"), "scored_at": now.isoformat()}
            w.writerow(row)
            print(f"{row['created_at'][:16]}  {row['symbol']:10s} {row['direction']:5s} "
                  f"P(TP1 before SL) = {row['p_tp1']:.2f}  take={row['take']}")
    print(f"Shadow predictions: {pred_path}")


if __name__ == "__main__":
    main()
