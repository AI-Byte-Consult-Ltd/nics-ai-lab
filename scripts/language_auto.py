"""NICS Language on the laptop without any manual step.

Called at the end of the nightly NICS Trading cycle (scripts/self_learn.py) and by
scripts\\windows\\language_train.bat. It does whatever is still missing, in order:

  1. corpus     download the licence-checked text sources (prepare_language_corpus.py)
  2. tokenizer  train our own tokenizer on all languages that have text
  3. tokens     write our own task examples, encode them and the first-wave languages into token files
  4. training   continue (or start) the micro model until the stop time

Every step is skipped when its result already exists, so a night that was interrupted
simply continues the next night. Each stage reports to the private dashboard (status
only: no text, no weights), so the state is visible without opening the laptop.

  python scripts/language_auto.py --until 08:00      # work until 08:00 local time
  python scripts/language_auto.py --max-minutes 30
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nics.common.config import load_config  # noqa: E402

FIRST_WAVE = ["ru", "en", "es", "bg", "pt"]  # agreed order: these five first
TEXT = ROOT / "data" / "processed" / "language" / "text"
LOGS = ROOT / "logs"
LOCK = LOGS / "language.lock"
MIN_TRAIN_MINUTES = 15


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)  # a Windows console may not take every character
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "language.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def minutes_until(clock: str, now: datetime | None = None) -> float:
    """Minutes from now to the next HH:MM local time."""
    now = now or datetime.now()
    hour, minute = (int(x) for x in clock.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds() / 60


def lang_bytes() -> dict:
    return {d.name: sum(f.stat().st_size for f in d.rglob("*.txt")) for d in sorted(TEXT.iterdir()) if d.is_dir()} if TEXT.is_dir() else {}


def report(status: str, note: str, push_report: bool, **extra) -> None:
    summary = {"status": status, "model_name": "nics-text-micro", "note": note,
               "reported_at": datetime.now(timezone.utc).isoformat(), **extra}
    LOGS.mkdir(exist_ok=True)
    (LOGS / "language_last.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if push_report:
        from nics.telemetry.push import push
        push("nics-language", "snapshot", summary, cycle=summary["reported_at"][:19], log=log)


def step(name: str, *args: str) -> None:
    log(f"{name}: start")
    t0 = time.time()
    with open(LOGS / "language.txt", "a", encoding="utf-8") as out:
        code = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"{name}: {'done' if code == 0 else 'FAILED code ' + str(code)} in {(time.time() - t0) / 60:.1f} min")
    if code != 0:
        raise RuntimeError(f"{name} failed with code {code}; details in logs\\language.txt")


def prepare(cfg, push_report: bool) -> None:
    tokenizer = ROOT / cfg["data"]["tokenizer"]
    tokens_dir = ROOT / cfg["data"]["tokens_dir"]
    if (tokens_dir / "train.bin").exists() and (tokens_dir / "manifest.json").exists() and tokenizer.exists():
        return
    report("подготовка", "Ноутбук скачивает тексты и готовит их к обучению.", push_report)
    step("corpus", "scripts/prepare_language_corpus.py")
    sizes = lang_bytes()
    missing = [lang for lang in FIRST_WAVE if sizes.get(lang, 0) < 5_000_000]
    if missing:
        raise RuntimeError(f"no text for {missing} (download failed?); the next night tries again")
    if not tokenizer.exists():
        # every language gets the same share, so the smallest second-wave sample sets the budget
        step("tokenizer", "scripts/train_tokenizer.py", "--vocab-size", str(cfg["model"]["vocab_size"]),
             "--max-mb-per-lang", "16", "--out", str(tokenizer))
    step("examples", "scripts/make_language_examples.py")
    step("tokens", "scripts/encode_corpus.py", "--tokenizer", str(tokenizer), "--out", str(tokens_dir),
         "--langs", ",".join(FIRST_WAVE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/text/micro.yaml")
    parser.add_argument("--until", default=None, help="local time HH:MM to stop at")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    push_report = not args.no_push

    from self_learn import pid_alive  # same stale-lock rule as the trading cycle
    LOGS.mkdir(exist_ok=True)
    if LOCK.exists():
        try:
            other = int(LOCK.read_text().strip() or 0)
        except ValueError:
            other = 0
        if pid_alive(other):
            log(f"another language run is active (pid {other}); nothing to do")
            return
    LOCK.write_text(str(os.getpid()))
    deadline = time.time() + 60 * (args.max_minutes if args.max_minutes is not None
                                   else minutes_until(args.until or "08:00"))
    try:
        cfg = load_config(str(ROOT / args.config))
        prepare(cfg, push_report)
        left = (deadline - time.time()) / 60
        if left < MIN_TRAIN_MINUTES:
            log(f"only {left:.0f} min left: training starts on the next run")
            report("готова к обучению", "Тексты скачаны и подготовлены. Обучение начнётся следующей ночью.", push_report)
            return
        from language_night import run_night
        log(f"training for {left:.0f} min")
        code = run_night(args.config, left - 5, args.threads, push_report)
        log(f"training ended with code {code}")
    except Exception as error:  # noqa: BLE001 - report and leave; the next night continues
        log(f"FAILED: {error}")
        report("ошибка", f"Ночной запуск остановился: {str(error)[:300]}", push_report)
        sys.exit(1)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
