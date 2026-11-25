"""NICS Vision Geo, step 3: train GeoNet from random weights on the built set (CPU).

  python scripts/train_geo.py                       # full config
  python scripts/train_geo.py --max-minutes 60      # stop after an hour, keep the best epoch
  python scripts/train_geo.py --gentle              # low priority, half of the threads (daytime)
  python scripts/train_geo.py --continue            # go on from the latest run's weights, 30 more epochs
  python scripts/train_geo.py --resend              # no training: report of the latest finished run

Writes experiments/vision/nics-vision-geo-micro/<run id>/ (model.pt, model_meta.json,
report.json; not in git) and sends a short snapshot to the CEO desk as model "nics-vision".
Validation chooses the checkpoint; the test split is evaluated once and only reported.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from nics.common.config import load_config  # noqa: E402
from nics.geo import dataset, train  # noqa: E402
from nics.telemetry import push  # noqa: E402


def gentle() -> int:
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00000040)  # IDLE_PRIORITY_CLASS
        else:
            os.nice(19)
    except Exception:  # noqa: BLE001 - priority is a courtesy
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def snapshot_text(r: dict) -> str:
    t = r["test"]["calibrated"]
    return (f"NICS Vision Geo: {len(r['classes'])} классов, {r['parameters']:,} параметров, "
            f"{r['data']['images']} фото. Тест: верно с первой попытки {t['top1'] * 100:.1f} %, "
            f"в тройке {t['top3'] * 100:.1f} %, сбалансированно {t['balanced_accuracy'] * 100:.1f} % "
            f"(угадывание {r['chance_top1'] * 100:.1f} %). {r['minutes']} мин на CPU.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/vision/geo_micro.yaml"))
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gentle", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="continue from the weights of the latest run (same classes) instead of random weights")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resend", action="store_true", help="no training: print and send the report of the latest run")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):      # Windows console/log is cp1252 by default: Russian text must not crash
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config)
    if args.resend:
        runs = sorted((ROOT / "experiments" / "vision" / cfg["model_id"]).glob("exp-*/report.json"))
        if not runs:
            print("no finished run found")
            return 1
        return finish(json.loads(runs[-1].read_text(encoding="utf-8")), runs[-1].parent.name, args.no_push)
    threads = gentle() if args.gentle else (args.threads or max(1, (os.cpu_count() or 2) - 2))
    torch.set_num_threads(threads)
    images, meta, summary = dataset.load_built(ROOT / cfg["data"]["built_dir"])
    tc = dict(cfg["train"]) | {"seed": cfg["seed"]}
    if args.epochs:
        tc["epochs"] = args.epochs
    init_state = init_from = None
    if args.cont:
        prev = sorted((ROOT / "experiments" / "vision" / cfg["model_id"]).glob("exp-*/model_meta.json"))
        if not prev:
            print("no earlier run to continue from: run without --continue first")
            return 1
        pm = json.loads(prev[-1].read_text(encoding="utf-8"))
        if pm["classes"] != summary["classes"]:
            print("the built photo set has other classes than the latest run: cannot continue, train from scratch")
            return 1
        init_state = torch.load(prev[-1].parent / "model.pt", map_location="cpu", weights_only=False)["state"]
        init_from = prev[-1].parent.name
        tc["lr"] = args.lr or float(tc.get("lr", 2e-3)) * 0.4   # gentler steps: the weights are already trained
        tc["epochs"] = args.epochs or 30
        print(f"continuing from {init_from}, lr {tc['lr']}, {tc['epochs']} epochs")
    elif args.lr:
        tc["lr"] = args.lr
    run = ROOT / "experiments" / "vision" / cfg["model_id"] / datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=True)
    print(f"run {run.name}, {threads} threads")
    report = train.train(images, meta, summary, tc, run, max_minutes=args.max_minutes,
                         init_state=init_state, init_from=init_from)
    return finish(report, run.name, args.no_push)


def finish(report: dict, run_name: str, no_push: bool) -> int:
    text = snapshot_text(report)
    c = report.get("continued")
    if c:
        v = report["val"]["raw"]
        kept = " Новые эпохи не улучшили проверку: оставлены прежние веса." if report.get("best_epoch") == 0 else ""
        text += (f" Продолжение от {c['from']}: на проверке сбалансированно {c['val_balanced_accuracy'] * 100:.1f} % -> "
                 f"{v['balanced_accuracy'] * 100:.1f} %, с первой попытки {c['val_top1'] * 100:.1f} % -> {v['top1'] * 100:.1f} %.{kept}")
    print(text)
    print(json.dumps(report["test"]["top_confusions"][:5], ensure_ascii=False))
    if not no_push:
        small = {k: report[k] for k in ("model", "parameters", "classes", "image_size", "temperature", "epochs_run",
                                        "minutes", "best_epoch", "chance_top1", "data")} | \
                {"val": report["val"]["calibrated"], "test": report["test"]["calibrated"], "run": run_name}
        push.push("nics-vision", "snapshot", small, text, cycle=run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
