"""NICS Trading self-learning cycle: download -> learn -> compare -> keep the better model.

Run nightly on the laptop (scripts/windows/install_tasks.bat sets that up):
  1. download a fresh snapshot of all our data from Vantage / the scanner
     (+ public D1/W1 history, + our MT5 trades if --mt5-label is given);
  2. re-score the current champion on the NEW data (fair comparison: same split),
     and check its live shadow predictions against outcomes that happened since;
  3. train a challenger warm-started from the champion (and, once a week, one
     from random weights, so the model can also escape its own history);
  4. promote the challenger only if its VALIDATION score on the new data beats the
     champion's by at least self_learning.min_improvement; otherwise the champion
     stays. The test split is never used for this choice (it would stop being an
     honest held-out measurement); it is only reported, next to the forward test of
     live shadow predictions;
  5. log everything, delete old snapshots and old resume checkpoints.

Nothing here trades or sends signals. The champion is used for shadow scoring only
(scripts/score_live.py) until we decide otherwise.

  python scripts/self_learn.py                       (full cycle)
  python scripts/self_learn.py --skip-fetch --snapshot data/market/<date>
  python scripts/self_learn.py --mt5-label prop      (also export trades of the open MT5 terminal)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402


def pid_alive(pid: int) -> bool:
    """True if a process with this PID is still running (a lock left by a laptop that was
    switched off mid-cycle points to a dead PID and is ignored)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_is_live(lock: Path) -> bool:
    if not lock.exists() or time.time() - lock.stat().st_mtime > 12 * 3600:
        return False
    parts = lock.read_text(encoding="utf-8", errors="ignore").split()
    pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
    return pid_alive(pid) if pid > 0 else True


def run(cmd: list[str], log) -> str:
    log(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ, cwd=ROOT)
    tail = (p.stdout + p.stderr)[-4000:]
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{tail}")
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/trading/nano.yaml"))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--mt5-label", default=None, help="label of the MT5 account open in the terminal")
    ap.add_argument("--from-scratch", action="store_true", help="also train a model from random weights now")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sl = cfg.get("self_learning") or {}
    base = ROOT / cfg.paths.experiments_dir
    base.mkdir(parents=True, exist_ok=True)
    champ_dir, champ_json = base / "champion", base / "champion.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lock = base / "self_learn.lock"
    if lock_is_live(lock):
        sys.exit(f"Another cycle is running ({lock}). Delete the file if it is stale.")
    lock.write_text(f"{stamp} {os.getpid()}")
    logf = open(base / "self_learn.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    report = {"cycle": stamp, "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        py = sys.executable
        # 1. data
        snap = Path(args.snapshot) if args.snapshot else ROOT / "data" / "market" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not args.skip_fetch:
            run([py, "scripts/fetch_market_data.py", "--out", str(snap)], log)
            try:
                run([py, "scripts/fetch_public_history.py", "--snapshot", str(snap)], log)
            except RuntimeError as e:
                log(f"public history skipped: {e}")
        prev = sorted(p for p in (ROOT / "data" / "market").glob("*") if p.is_dir() and p != snap)
        for older in reversed(prev):  # keep our MT5 trades: carry the latest file of each account forward
            for f in older.glob("own_trades_*.csv"):
                if not (snap / f.name).exists():
                    shutil.copy2(f, snap / f.name)
        if args.mt5_label:
            try:
                run([py, "scripts/fetch_mt5_history.py", "--snapshot", str(snap), "--label", args.mt5_label], log)
            except RuntimeError as e:
                log(f"MT5 export skipped: {e}")
        report["snapshot"] = snap.name

        # 2. champion on the new data + forward check of its live shadow predictions
        champion = json.loads(champ_json.read_text()) if champ_json.exists() else None
        champ_score = None
        if champion and (champ_dir / "model.safetensors").exists():
            out = base / f"eval-champion-{stamp}.json"
            try:
                run([py, "scripts/eval_trading.py", "--config", args.config, "--release", str(champ_dir),
                     "--snapshot", str(snap), "--predictions", str(base / "shadow" / "predictions.csv"),
                     "--out", str(out)], log)
                ev = json.loads(out.read_text())
                champ_score = ev["val_score"]
                report["champion"] = {"experiment_id": champion["experiment_id"], "val_score": champ_score,
                                      "test_score": ev["test_score"],
                                      "forward_shadow": ev.get("forward_shadow")}
            except RuntimeError as e:  # e.g. feature schema changed after a code update
                log(f"champion cannot be evaluated on current code, it will be replaced: {e}")
        log(f"champion: {champion and champion.get('experiment_id')} score on new data: {champ_score}")

        # 3. challengers
        last_scratch = champion.get("last_scratch_at") if champion else None
        need_scratch = (args.from_scratch or champ_score is None or not last_scratch or
                        datetime.fromisoformat(last_scratch) < datetime.now(timezone.utc)
                        - timedelta(days=int(sl.get("scratch_every_days", 7))))
        plans = []
        if champ_score is not None:
            plans.append(("warm", ["--init-from", str(champ_dir), "--max-steps", str(sl.get("finetune_steps", 1500)),
                                   "--lr-scale", str(sl.get("lr_scale", 0.5))]))
        if need_scratch:
            plans.append(("scratch", []))
        results = []
        for kind, extra in plans:
            exp = f"self-{stamp}-{kind}"
            outp = run([py, "scripts/train_trading.py", "--config", args.config, "--snapshot", str(snap),
                        "--experiment-id", exp] + extra, log)
            rel = Path(re.findall(r"RELEASE_DIR=(.+)", outp)[-1].strip())
            m = json.loads((rel / "metrics.json").read_text())
            results.append({"kind": kind, "experiment_id": exp, "release": str(rel), "val_score": m["val_score"],
                            "test_score": m["test_score"],
                            "signal_test_auc": (m.get("signal") or {}).get("test", {}).get("auc"),
                            "horizon_test_acc": m["horizon"]["test"].get("accuracy"),
                            "horizon_majority": m["horizon"]["test"].get("majority_baseline")})
            log(f"challenger {exp}: val_score {m['val_score']:.4f} (test {m['test_score']:.4f}, report only)")
        report["challengers"] = results

        # 4. promote
        best = min(results, key=lambda r: r["val_score"]) if results else None
        margin = float(sl.get("min_improvement", 0.002))
        promote = best is not None and (champ_score is None or best["val_score"] < champ_score - margin)
        if promote:
            if champ_dir.exists():
                shutil.rmtree(champ_dir)
            shutil.copytree(best["release"], champ_dir)
            new = {"experiment_id": best["experiment_id"], "release": best["release"],
                   "promoted_at": datetime.now(timezone.utc).isoformat(), "val_score": best["val_score"],
                   "test_score": best["test_score"],
                   "replaced": champion and champion.get("experiment_id"), "replaced_score_on_same_data": champ_score,
                   "snapshot": snap.name,
                   "last_scratch_at": (datetime.now(timezone.utc).isoformat() if need_scratch
                                       else champion.get("last_scratch_at"))}
            champ_json.write_text(json.dumps(new, indent=2))
            with open(base / "champion_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(new) + "\n")
            log(f"PROMOTED {best['experiment_id']} (val {best['val_score']:.4f} vs {champ_score})")
        else:
            if champion and need_scratch:
                champion["last_scratch_at"] = datetime.now(timezone.utc).isoformat()
                champ_json.write_text(json.dumps(champion, indent=2))
            log(f"champion kept (val {champ_score}); best challenger {best and best['val_score']}")
        report["promoted"] = bool(promote)
        report["champion_now"] = json.loads(champ_json.read_text())["experiment_id"] if champ_json.exists() else None

        # 5. housekeeping: old snapshots, resume checkpoints of experiments that were not promoted
        keep = int(sl.get("keep_snapshots", 3))
        snaps = sorted(p for p in (ROOT / "data" / "market").glob("*") if p.is_dir())
        for old in snaps[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
        champ_now = report["champion_now"]
        for exp in base.glob("self-*"):
            if exp.name != champ_now and (exp / "checkpoints").exists():
                shutil.rmtree(exp / "checkpoints", ignore_errors=True)
    except Exception as e:
        report["error"] = str(e)[-2000:]
        log(f"CYCLE FAILED: {e}")
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (base / "self_learn_last.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with open(base / "self_learn_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
        lock.unlink(missing_ok=True)
        logf.close()
    print(json.dumps(report, indent=2))
    if "error" in report:
        sys.exit(1)


if __name__ == "__main__":
    main()
