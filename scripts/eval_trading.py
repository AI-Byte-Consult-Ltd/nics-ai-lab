"""Evaluate trained NICS Trading weights on a snapshot (same split rules as training).

Also checks the shadow predictions made LIVE by scripts/score_live.py against the
outcomes that happened afterwards (forward test: the model scored each signal
before its result was known).

  python scripts/eval_trading.py --config configs/trading/nano.yaml \
      --release experiments/trading/nics-trading-nano/champion --snapshot data/market/<date> \
      [--predictions experiments/trading/nics-trading-nano/shadow/predictions.csv] [--out eval.json]
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.training import trading as T  # noqa: E402


def forward_report(pred_path: Path, S: dict, threshold) -> dict:
    """Live predictions vs outcomes known later. Only predictions made before the outcome count."""
    if not pred_path.exists():
        return {"n": 0, "note": "no shadow predictions yet"}
    label = {sid: (int(y), float(t1), float(rr), float(c))
             for sid, y, t1, rr, c in zip(S["id"], S["y"], S["t1"], S["rr"], S["cost"]) if y >= 0}
    ys, ps, rs = [], [], []
    with open(pred_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r["signal_id"]
            if sid not in label:
                continue
            y, t1, rr, cost = label[sid]
            if datetime.fromisoformat(r["scored_at"]).timestamp() >= t1:
                continue  # scored after the result was already known: not a forward test
            ys.append(y); ps.append(float(r["p_tp1"])); rs.append(T.r_per_trade(np.array([y]), rr, cost)[0])
    if not ys:
        return {"n": 0, "note": "no shadow-scored signal has an outcome yet"}
    y, p, r = np.array(ys), np.array(ps), np.array(rs)
    out = T.signal_metrics(y, p)
    out["mean_R_all_after_spread"] = float(r.mean())
    if threshold is not None:
        keep = p >= threshold
        out["threshold"] = threshold
        out["taken"] = int(keep.sum())
        out["mean_R_taken_after_spread"] = float(r[keep].mean()) if keep.any() else None
    out["note"] = "Forward test on live signals. Past results, not a forecast or a guarantee."
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--no-public", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, man = T.load_release(Path(args.release), cfg)
    use_public = not args.no_public and bool(cfg.data.get("use_public", True))
    series, H, S, info = T.build_samples(cfg, Path(args.snapshot), use_public)
    S["y"] = S["y"].astype(np.int64)
    win = T.Windows(series, cfg.data.window)
    splits = T.make_splits(H, S, cfg.data.split)
    m = T.final_metrics(model, win, H, S, splits, cfg)
    m["experiment_id"] = man.get("experiment_id")
    m["snapshot"] = Path(args.snapshot).name
    if args.predictions:
        thr = (json.loads((Path(args.release) / "metrics.json").read_text()).get("signal") or {}).get(
            "threshold_from_val") if (Path(args.release) / "metrics.json").exists() else None
        m["forward_shadow"] = forward_report(Path(args.predictions), S, thr)
    text = json.dumps(m, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
