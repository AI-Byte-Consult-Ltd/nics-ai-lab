"""Start a NICS Language training run on the laptop on request, without touching the laptop.

The laptop cannot be reached from outside, so it asks: every 2 minutes (inside the task
"NICS Trading opinion") this script reads control/language_request.json from origin/main
(git fetch only, the working tree is not changed). A request looks like

  {"id": "2026-11-23-day", "minutes": 60, "not_after": "2026-11-23T14:00:00+00:00"}

and is executed once: language_auto.py --max-minutes N --gentle starts as a separate
process and continues from the last checkpoint. Gentle = lowest CPU priority and half of
the threads, so the laptop stays usable while it trains. Rules:
  * an empty id, an already handled id or a request past not_after does nothing;
  * minutes are clamped to 10..MAX_MINUTES;
  * nothing starts between 02:00 and 08:00 local time (the nightly cycle owns that window)
    or while another language run holds the lock.
Only this one fixed action can be requested; the file carries no commands.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUEST = "control/language_request.json"
DONE = ROOT / "logs" / "language_remote_done.txt"
MAX_MINUTES = 240
MIN_MINUTES = 10


def read_request(root: Path = ROOT) -> dict | None:
    """The request file as it is on origin/main right now, or None when it cannot be read."""
    try:
        subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=root, timeout=60, check=True,
                       capture_output=True)
        out = subprocess.run(["git", "show", f"origin/main:{REQUEST}"], cwd=root, timeout=30, check=True,
                             capture_output=True)
        data = json.loads(out.stdout.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - no network, no file, bad JSON: simply nothing to do
        return None


def decide(req: dict | None, done_id: str, now: datetime, local_hour: int) -> tuple[int, str]:
    """(minutes to train, reason). minutes == 0 means do nothing."""
    if not req:
        return 0, "no request"
    rid = str(req.get("id") or "").strip()
    if not rid:
        return 0, "no request"
    if rid == done_id:
        return 0, "already handled"
    try:
        minutes = float(req.get("minutes") or 0)
        not_after = datetime.fromisoformat(str(req.get("not_after") or ""))
    except (TypeError, ValueError):
        return 0, "bad request"
    if not_after.tzinfo is None:
        not_after = not_after.replace(tzinfo=timezone.utc)
    if now > not_after:
        return 0, "expired"
    if minutes <= 0:
        return 0, "bad request"
    if 2 <= local_hour < 8:
        return 0, "night window"
    return int(max(MIN_MINUTES, min(MAX_MINUTES, minutes))), "start"


def start(minutes: int, root: Path = ROOT) -> None:
    """Start the run as its own process, so the 2-minute task is not held."""
    try:  # run the newest code and sources, like the manual .bat does; no network is not a reason to skip
        subprocess.run(["git", "pull", "--ff-only", "--quiet"], cwd=root, timeout=120, capture_output=True)
    except Exception:  # noqa: BLE001
        pass
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    (root / "logs").mkdir(exist_ok=True)
    out = open(root / "logs" / "language_remote.txt", "ab")
    subprocess.Popen([sys.executable, str(root / "scripts" / "language_auto.py"), "--max-minutes", str(minutes), "--gentle"],
                     cwd=root, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     creationflags=flags, close_fds=True)


def main() -> None:
    DONE.parent.mkdir(exist_ok=True)
    done_id = DONE.read_text(encoding="utf-8").strip() if DONE.exists() else ""
    req = read_request()
    minutes, reason = decide(req, done_id, datetime.now(timezone.utc), datetime.now().hour)
    if reason in ("no request", "already handled"):
        return
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rid = str((req or {}).get("id") or "")
    print(f"{stamp} language request {rid.encode('ascii', 'replace').decode()}: {reason}"
          + (f" ({minutes} min)" if minutes else ""))
    if reason == "night window":
        return  # stays pending; it starts after 08:00 if still valid
    DONE.write_text(rid, encoding="utf-8")  # written first: a request never runs twice
    if minutes:
        start(minutes)


if __name__ == "__main__":
    main()
