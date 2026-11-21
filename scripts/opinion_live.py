"""NICS Trading's opinion on every signal actually SENT to users, for Alexander only.

Run every 2 minutes on the laptop (task "NICS Trading opinion", install_tasks.bat):
  1. one small request: scanner signals of the last day; keep those with
     delivery_status SENT, created in the last --hours, not yet answered;
  2. only if there are such signals: 15-minute candles (6 days) for their symbols;
  3. the champion scores each signal and its mirror (opposite direction, same entry,
     same stop/take distances) -> agree / against + P(TP1 before SL) for both sides;
  4. POST to n8n "[43 LAB] Model Opinion" (/webhook/nics-lab-opinion, same lab key),
     which sends it to Alexander in Telegram. Nobody else sees it.

Shadow only: nothing here changes signals, users, the bot or the EA.
Answered signals are kept in shadow/opinions_sent.csv, so each one is sent once.

  python scripts/opinion_live.py [--hours 3] [--dry-run]
"""
import argparse
import copy
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nics.common.config import Config, load_config  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.training import trading as T  # noqa: E402
import fetch_market_data as fm  # noqa: E402

LOG_COLS = ["signal_id", "symbol", "created_at", "scanner_direction", "model_direction", "p_scanner_side",
            "p_other_side", "experiment_id", "sent_at"]


def opinion_url(env=None) -> str | None:
    env = os.environ if env is None else env
    if (env.get("NICS_OPINION_URL") or "").strip():
        return env["NICS_OPINION_URL"].strip()
    p = urllib.parse.urlsplit((env.get("NICS_DATA_URL") or "").strip())
    return urllib.parse.urlunsplit((p.scheme, p.netloc, "/webhook/nics-lab-opinion", "", "")) if p.netloc else None


def post(url: str, key: str, body: dict) -> bool:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "X-NICS-Lab-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        print(f"opinion post failed: HTTP {e.code}")
    except Exception as e:  # network: try again next run
        print(f"opinion post failed: {type(e).__name__}")
    return False


def new_sent_signals(rows: list[dict], done: set[str], hours: float, now: float) -> list[dict]:
    out = []
    for r in rows:
        if (r.get("delivery_status") or "").upper() != "SENT" or str(r.get("id")) in done or not r.get("created_at"):
            continue
        t = market._ts(r["created_at"])
        if t is not None and now - t <= hours * 3600:
            out.append(r)
    return out


def score(cfg, snap: Path, model, ids: set[str]) -> list[dict]:
    """Model opinion for the given signal ids (signal vs its mirror)."""
    sig_iv = cfg.data.signal_interval
    lcfg = Config(copy.deepcopy(dict(cfg)))
    lcfg["data"]["intervals"] = {sig_iv: dict(cfg.data.intervals)[sig_iv]}
    lcfg["methods"] = {"enabled": False}
    series, _, S, _ = T.build_samples(lcfg, snap, use_public=False, label_setups=False, mirror_scanner=True)
    if not len(S["id"]):
        return []
    win = T.Windows(series, cfg.data.window)
    pos = {str(x): j for j, x in enumerate(S["id"])}
    pairs = [(sid, pos[sid], pos[sid + "|mirror"]) for sid in ids if sid in pos and sid + "|mirror" in pos]
    if not pairs:
        return []
    mask = np.zeros(len(S["id"]), bool)
    for _, a, b in pairs:
        mask[a] = mask[b] = True
    p = dict(zip(np.flatnonzero(mask), T.predict_signal(model, win, S, mask)))
    out = []
    for sid, a, b in pairs:
        d = "LONG" if S["f"][a][0] > 0 else "SHORT"
        out.append({"signal_id": sid, "symbol": series[int(S["sid"][a])].symbol, "created_at": T.iso(S["t0"][a]),
                    "scanner_direction": d, "model_direction": d if p[a] >= p[b] else ("SHORT" if d == "LONG" else "LONG"),
                    "p_scanner_side": round(float(p[a]), 4), "p_other_side": round(float(p[b]), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/trading/nano.yaml"))
    ap.add_argument("--hours", type=float, default=3, help="only signals sent in the last N hours")
    ap.add_argument("--dry-run", action="store_true", help="print, do not send")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = ROOT / cfg.paths.experiments_dir
    champ = base / "champion"
    if not (champ / "model.safetensors").exists():
        sys.exit("No champion yet: run scripts/self_learn.py once first.")
    url, key = os.environ.get("NICS_DATA_URL"), os.environ.get("NICS_DATA_KEY")
    if not url or not key:
        sys.exit("Set NICS_DATA_URL and NICS_DATA_KEY first.")
    log_path = base / "shadow" / "opinions_sent.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if log_path.exists():
        with open(log_path, newline="", encoding="utf-8") as f:
            done = {r["signal_id"] for r in csv.DictReader(f)}

    name, cols = fm.TABLES["signals"]
    rows = fm.fetch(url, key, {"mode": "signals", "days": 1}).get("rows") or []
    todo = new_sent_signals(rows, done, args.hours, datetime.now(timezone.utc).timestamp())
    if not todo:
        print("No new sent signals.")
        return

    snap = ROOT / "data" / "opinion"
    snap.mkdir(parents=True, exist_ok=True)
    with open(snap / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in todo:
            w.writerow({c: fm.cell(r.get(c)) for c in cols})
    sig_iv = cfg.data.signal_interval
    with open(snap / "candles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fm.CANDLE_COLS, extrasaction="ignore")
        w.writeheader()
        for sym in sorted({r["symbol"] for r in todo if r.get("symbol")}):
            for r in fm.fetch(url, key, {"mode": "candles", "symbol": sym, "interval": sig_iv, "days": 6,
                                         "after": "", "limit": fm.PAGE}).get("rows") or []:
                r["source"] = "vantage_mt5"
                w.writerow({c: fm.cell(r.get(c)) for c in fm.CANDLE_COLS})

    model, man = T.load_release(champ, cfg)
    ops = score(cfg, snap, model, {str(r["id"]) for r in todo})
    target = opinion_url()
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        for o in ops:
            o["experiment_id"] = man.get("experiment_id")
            print(f"{o['symbol']:10s} {o['scanner_direction']:5s} model {o['model_direction']:5s} "
                  f"p {o['p_scanner_side']:.2f} / {o['p_other_side']:.2f}")
            if args.dry_run:
                continue
            if target and post(target, key, o):
                w.writerow({**o, "sent_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
