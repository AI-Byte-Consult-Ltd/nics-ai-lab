"""One night of NICS Language training on the laptop (called by scripts\\windows\\language_train.bat).

Continues the newest unfinished experiment of the chosen config, or starts a new one from
random weights when there is none. Training stops cleanly after --max-minutes with a
checkpoint, so the next night continues from the same place.

After the run a short report is written to logs/language_last.json and sent to the private
dashboard (metrics only: no weights, no text).

Usage:
  python scripts/language_night.py                      # micro, 290 minutes
  python scripts/language_night.py --max-minutes 30     # short trial
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402


def newest_unfinished(experiments_dir: Path) -> Path | None:
    """Newest experiment that has a random-init checkpoint and no released weights yet."""
    if not experiments_dir.is_dir():
        return None
    open_runs = [d for d in experiments_dir.iterdir()
                 if (d / "checkpoint-step-000000" / "trainer-state.pt").exists() and not (d / "release").exists()]
    return max(open_runs, key=lambda d: d.name) if open_runs else None


def read_progress(exp_dir: Path) -> dict:
    """Last and best validation loss and the step reached, from the experiment telemetry."""
    last, best, first = None, None, None
    events = exp_dir / "events.jsonl"
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            e = {**e, **(e.get("data") or {})} if isinstance(e.get("data"), dict) else e
            if e.get("validation_loss") is None:
                continue
            first = first or e
            last = e
            if best is None or e["validation_loss"] < best["validation_loss"]:
                best = e
    return {"first": first, "last": last, "best": best}


def build_report(cfg, exp_dir: Path, started: float, returncode: int) -> dict:
    progress = read_progress(exp_dir)
    last, best = progress["last"], progress["best"]
    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8")) if (exp_dir / "manifest.json").exists() else {}
    corpus = manifest.get("corpus") or {}
    max_steps = int(cfg["training"]["max_steps"])
    step = int(last["step"]) if last else 0
    finished = (exp_dir / "release").exists()
    tokens = (corpus.get("tokens") or {}).get("train")
    langs = ", ".join(sorted(corpus.get("tokens_per_language") or {}))
    percent = round(100.0 * (step + 1) / max_steps, 1) if last else 0.0
    return {
        # the first seven keys are what the dashboard card shows (Russian, for Alexander)
        "status": "обучена" if finished else ("обучается" if returncode == 0 else "ошибка"),
        "model_name": cfg["model_id"],
        "params": f"{manifest.get('parameter_count'):,}".replace(",", " ") if manifest.get("parameter_count") else None,
        "dataset": f"{tokens / 1e6:.0f} млн токенов, языки: {langs}" if tokens else None,
        "metric_label": "Ошибка на проверке (меньше лучше)",
        "metric": last["validation_loss"] if last else None,
        "note": (f"Пройдено {percent} % плана обучения (шаг {step} из {max_steps}). "
                 "Обучение с нуля, только на текстах с проверенной лицензией."),
        "best_metric": best["validation_loss"] if best else None,
        "step": step,
        "max_steps": max_steps,
        "progress_percent": percent,
        "tokens_per_second": last.get("tokens_per_second") if last else None,
        "night_minutes": round((time.time() - started) / 60, 1),
        "experiment_id": exp_dir.name,
        "returncode": returncode,
        "pretrained_weights_used": None,
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }


def journal_check(config: str, exp_dir: Path, samples: int = 40) -> dict | None:
    """How many journal posts written by the current weights pass the code check (about a minute)."""
    out = exp_dir / "journal_check.json"
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_journal.py"), "--experiment", str(exp_dir),
                        "--config", str(ROOT / config), "--samples", str(samples), "--out", str(out)],
                       cwd=ROOT, timeout=900, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = json.loads(out.read_text(encoding="utf-8"))
        return {k: result[k] for k in ("step", "samples", "passed", "pass_percent", "by_language", "tokens_per_second")}
    except Exception as error:  # noqa: BLE001 - the check must never fail the night
        print(f"journal check skipped: {type(error).__name__}")
        return None


def run_night(config: str, max_minutes: float, threads: int | None = None, push_report: bool = True) -> int:
    args = argparse.Namespace(config=config, max_minutes=max_minutes, threads=threads, no_push=not push_report)
    cfg = load_config(str(ROOT / args.config))
    tokens_dir = ROOT / cfg["data"]["tokens_dir"]
    if not (tokens_dir / "train.bin").exists():
        sys.exit(f"No token files in {tokens_dir}. Run scripts\\windows\\language_prepare.bat first.")

    experiments_dir = ROOT / cfg["paths"]["experiments_dir"]
    exp_dir = newest_unfinished(experiments_dir)
    cmd = [sys.executable, str(ROOT / "scripts" / "train_text.py"), "--config", str(ROOT / args.config),
           "--max-minutes", str(args.max_minutes)]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    if exp_dir:
        print(f"Continuing {exp_dir.name}", flush=True)
        cmd += ["--resume", str(exp_dir)]
    else:
        experiment_id = datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%SZ")
        exp_dir = experiments_dir / experiment_id
        print(f"Starting a new experiment {experiment_id} from random weights", flush=True)
        cmd += ["--experiment-id", experiment_id]

    started = time.time()
    returncode = subprocess.run(cmd, cwd=ROOT).returncode
    report = build_report(cfg, exp_dir, started, returncode)
    journal = journal_check(args.config, exp_dir) if returncode == 0 else None
    if journal:
        report["journal_check"] = journal
        report["note"] += (f" Задача «дневник»: {journal['passed']} из {journal['samples']} текстов модели"
                           " прошли проверку кодом.")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "language_last.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)  # ASCII only: a Windows log file may not take Cyrillic
    if not args.no_push:
        from nics.telemetry.push import push
        push("nics-language", "snapshot", report, cycle=report["reported_at"][:19])
    return returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/text/micro.yaml")
    parser.add_argument("--max-minutes", type=float, default=290.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    sys.exit(run_night(args.config, args.max_minutes, args.threads, not args.no_push))


if __name__ == "__main__":
    main()
