"""Send a short training report to Alexander's private dashboard (n8n "[42 OPS] CEO Desk").

The laptop cannot be reached from the server, so the laptop pushes: after every nightly
cycle `scripts/self_learn.py` POSTs the cycle report (the same content as
self_learn_last.json) and the latest direction summary (direction_last.txt) to the n8n
webhook `nics-lab-metrics`.

Authentication reuses the key the laptop already has for the data export
(NICS_DATA_KEY, header X-NICS-Lab-Key), so nothing new has to be set up.
The URL is NICS_METRICS_URL if set, otherwise derived from NICS_DATA_URL
(same host, path /webhook/nics-lab-metrics).

Only metrics leave the laptop: no weights, no data, no secrets. A failed push never
fails the training cycle; it is logged and retried on the next cycle.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

DEFAULT_PATH = "/webhook/nics-lab-metrics"
MAX_SUMMARY = 30000


def metrics_url(env: Optional[dict] = None) -> Optional[str]:
    env = os.environ if env is None else env
    explicit = (env.get("NICS_METRICS_URL") or "").strip()
    if explicit:
        return explicit
    data_url = (env.get("NICS_DATA_URL") or "").strip()
    if not data_url:
        return None
    p = urllib.parse.urlsplit(data_url)
    if not p.scheme or not p.netloc:
        return None
    return urllib.parse.urlunsplit((p.scheme, p.netloc, DEFAULT_PATH, "", ""))


def build_payload(model: str, kind: str, summary: Any = None, text: str = "", cycle: str = "") -> bytes:
    s = summary if isinstance(summary, str) else json.dumps(summary if summary is not None else {}, default=str)
    if len(s) > MAX_SUMMARY:  # keep the row small; the full file stays on the laptop
        s = json.dumps({"truncated": True, "head": s[: MAX_SUMMARY - 200]})
    body = {"model": model, "kind": kind, "cycle": str(cycle or "")[:64], "summary": s, "text": (text or "")[:4000]}
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def push(model: str, kind: str, summary: Any = None, text: str = "", cycle: str = "",
         log: Callable[[str], None] = print, env: Optional[dict] = None, timeout: float = 20.0) -> bool:
    """POST one report. Returns True on HTTP 200, False otherwise. Never raises."""
    env = os.environ if env is None else env
    url, key = metrics_url(env), (env.get("NICS_DATA_KEY") or "").strip()
    if not url or not key:
        log("dashboard push skipped: NICS_DATA_URL / NICS_DATA_KEY not set")
        return False
    req = urllib.request.Request(url, data=build_payload(model, kind, summary, text, cycle), method="POST",
                                 headers={"Content-Type": "application/json", "X-NICS-Lab-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = r.status == 200
        log(f"dashboard push {kind}: {'ok' if ok else 'HTTP ' + str(r.status)}")
        return ok
    except urllib.error.HTTPError as e:
        log(f"dashboard push {kind} failed: HTTP {e.code}" + (" (wrong NICS_DATA_KEY?)" if e.code == 403 else ""))
    except Exception as e:  # network down, DNS, timeout: try again next cycle
        log(f"dashboard push {kind} failed: {type(e).__name__}")
    return False
