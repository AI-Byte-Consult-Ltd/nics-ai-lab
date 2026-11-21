"""Daily shadow comparison: scanner direction vs NICS Trading's own direction.

scripts/score_live.py writes, for every new scanner signal, the direction the model
would take (shadow/direction.csv): it scores the signal and its mirror (opposite
direction, same entry, same stop and take distances, so the same reward:risk) and
picks the side with the higher P(TP1 before SL). Nothing is sent anywhere and
nothing changes the live signals or the EA.

This script takes a fresh snapshot (candles + signals), re-derives from broker
candles what happened to BOTH sides of every scored signal, and reports per day
(Europe/Sofia):
  - scanner result vs model result, in R per signal and in total, and in $ at a
    fixed risk per signal (--risk-usd, default 100 $);
  - how often they agreed, and who won where they disagreed;
  - the same per plan group (STRONG / RELAXED / WEAK ...). Weak signals are in here
    for internal analysis only; they never go into public statistics.
Signals whose chosen side has no outcome yet are counted as open; the last days are
recomputed every night until everything is resolved.

  python scripts/direction_report.py --snapshot data/market/<date> [--days 3] [--risk-usd 100]

Writes shadow/direction_reports/<day>.json, shadow/direction_daily.csv and
shadow/direction_last.txt (short Russian summary).
"""
import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import Config, load_config  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.training import trading as T  # noqa: E402

try:
    from zoneinfo import ZoneInfo
    SOFIA = ZoneInfo("Europe/Sofia")
except Exception:  # Windows without the tzdata package: winter time
    SOFIA = timezone(timedelta(hours=2))

DAILY_COLS = ["day", "signals", "resolved", "open", "agree", "disagree", "scanner_R", "model_R",
              "scanner_usd", "model_usd", "disagree_scanner_R", "disagree_model_R",
              "ideas_scanner_R", "ideas_model_R", "risk_usd", "updated_at"]


def outcomes(cfg, snap: Path) -> dict[str, float]:
    """signal id (and '<id>|mirror') -> realised R after spread, for setups already resolved."""
    sig_iv = cfg.data.signal_interval
    lcfg = Config(copy.deepcopy(dict(cfg)))
    lcfg["data"]["intervals"] = {sig_iv: dict(cfg.data.intervals)[sig_iv]}
    lcfg["methods"] = {"enabled": False}
    _, _, S, _ = T.build_samples(lcfg, snap, use_public=False, label_setups=True, mirror_scanner=True)
    if not len(S["y"]):
        return {}
    r = T.r_per_trade(S["y"].astype(np.int64), S["rr"].astype(float), S["cost"].astype(float))
    return {str(i): float(x) for i, x in zip(S["id"], r)}


def unique_ideas(rows: list[dict]) -> list[dict]:
    """The scanner writes the same idea again on every scan: keep the first record of each
    symbol + scanner direction + hour."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["created_at"]):
        key = (r["symbol"], r["scanner_direction"], r["created_at"][:13])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def summarise(rows: list[dict], res: dict[str, float], risk: float) -> dict:
    out = {"signals": len(rows), "resolved": 0, "open": 0, "agree": 0, "disagree": 0,
           "scanner_R": 0.0, "model_R": 0.0, "disagree_scanner_R": 0.0, "disagree_model_R": 0.0,
           "scanner_wins": 0, "model_wins": 0}
    for r in rows:
        sid = r["signal_id"]
        same = r["model_direction"] == r["scanner_direction"]
        rs = res.get(sid)
        rm = rs if same else res.get(sid + "|mirror")
        if rs is None or rm is None:
            out["open"] += 1
            continue
        out["resolved"] += 1
        out["scanner_R"] += rs
        out["model_R"] += rm
        if same:
            out["agree"] += 1
        else:
            out["disagree"] += 1
            out["disagree_scanner_R"] += rs
            out["disagree_model_R"] += rm
            out["scanner_wins"] += int(rs > rm)
            out["model_wins"] += int(rm > rs)
    n = max(out["resolved"], 1)
    for k in ("scanner_R", "model_R", "disagree_scanner_R", "disagree_model_R"):
        out[k] = round(out[k], 3)
    out["scanner_R_per_signal"] = round(out["scanner_R"] / n, 3)
    out["model_R_per_signal"] = round(out["model_R"] / n, 3)
    out["scanner_usd"] = round(out["scanner_R"] * risk, 2)
    out["model_usd"] = round(out["model_R"] * risk, 2)
    return out


def russian(day: str, s: dict, risk: float) -> str:
    better = ("лучше модель" if s["model_R"] > s["scanner_R"] else
              "лучше сканер" if s["scanner_R"] > s["model_R"] else "результат одинаковый")
    return "\n".join([
        f"Направление: сканер против NICS Trading, {day} (тень, на торговлю не влияет)",
        f"Только сигналы, отправленные пользователям: {s['signals']}, с итогом: {s['resolved']}, "
        f"ещё открыты: {s['open']}",
        f"Совпали по направлению: {s['agree']}, разошлись: {s['disagree']}",
        f"Сканер: {s['scanner_R']:+.2f} R ({s['scanner_R_per_signal']:+.2f} R на сигнал), "
        f"{s['scanner_usd']:+.0f} $ при риске {risk:.0f} $",
        f"Модель: {s['model_R']:+.2f} R ({s['model_R_per_signal']:+.2f} R на сигнал), "
        f"{s['model_usd']:+.0f} $ при риске {risk:.0f} $",
        f"Где разошлись: сканер {s['disagree_scanner_R']:+.2f} R, модель {s['disagree_model_R']:+.2f} R "
        f"(сканер лучше {s['scanner_wins']}, модель лучше {s['model_wins']})",
        f"Итог дня: {better}. Это измерение на прошедших сделках, не прогноз.",
    ] + ([f"Все идеи сканера без повторов (включая неотправленные): {s['ideas']['resolved']} с итогом, "
          f"сканер {s['ideas']['scanner_R']:+.2f} R, модель {s['ideas']['model_R']:+.2f} R"] if s.get("ideas") else []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/trading/nano.yaml"))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--direction", default=None, help="shadow/direction.csv (default: under experiments)")
    ap.add_argument("--days", type=int, default=3, help="recompute the last N finished days")
    ap.add_argument("--risk-usd", type=float, default=100.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    shadow = ROOT / cfg.paths.experiments_dir / "shadow"
    dpath = Path(args.direction) if args.direction else shadow / "direction.csv"
    if not dpath.exists():
        print(json.dumps({"n": 0, "note": "no direction predictions yet"}))
        return
    with open(dpath, newline="", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    sig = {r["id"]: r for r in market.load_signals(Path(args.snapshot))}
    asset = {k: (r.get("plan_group") or "").lower() or "?" for k, r in sig.items()}  # asset class, not strength
    sent_ids = {k for k, r in sig.items() if (r.get("delivery_status") or "").upper() == "SENT"}
    res = outcomes(cfg, Path(args.snapshot))

    def day_of(r):
        return datetime.fromisoformat(r["created_at"]).astimezone(SOFIA).date()

    today = datetime.now(SOFIA).date()
    days = [today - timedelta(days=k) for k in range(args.days, 0, -1)]  # finished days only
    rep_dir = shadow / "direction_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    daily_path = shadow / "direction_daily.csv"
    daily = {}
    if daily_path.exists():
        with open(daily_path, newline="", encoding="utf-8") as f:
            daily = {r["day"]: r for r in csv.DictReader(f)}
    last = None
    for day in days:
        rows = [r for r in preds if day_of(r) == day]
        if not rows:
            continue
        sent = [r for r in rows if r["signal_id"] in sent_ids]  # what users actually received
        total = summarise(sent, res, args.risk_usd)
        ideas = summarise(unique_ideas(rows), res, args.risk_usd)
        records = summarise(rows, res, args.risk_usd)
        by_asset = {g: summarise([r for r in sent if asset.get(r["signal_id"], "?") == g], res, args.risk_usd)
                    for g in sorted({asset.get(r["signal_id"], "?") for r in sent})}
        rep = {"day": day.isoformat(), "time_zone": "Europe/Sofia", "risk_usd_per_signal": args.risk_usd,
               "note": "Shadow test, measured on past trades. Not a forecast. 'total' = signals actually "
                       "sent to users. The scanner re-records the same idea every scan, so all records "
                       "are also given deduplicated (symbol + direction + hour).",
               "total": total, "all_unique_ideas": ideas, "all_records_raw": records,
               "sent_by_asset_class": by_asset,
               "updated_at": datetime.now(timezone.utc).isoformat()}
        total = dict(total, ideas=ideas)
        (rep_dir / f"{day.isoformat()}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
        daily[day.isoformat()] = {"day": day.isoformat(), **{k: total[k] for k in DAILY_COLS[1:-4]},
                                  "ideas_scanner_R": ideas["scanner_R"], "ideas_model_R": ideas["model_R"],
                                  "risk_usd": args.risk_usd, "updated_at": rep["updated_at"]}
        last = (day.isoformat(), total)
    with open(daily_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DAILY_COLS)
        w.writeheader()
        for k in sorted(daily):
            w.writerow({c: daily[k].get(c, "") for c in DAILY_COLS})
    if last:
        text = russian(last[0], last[1], args.risk_usd)
        (shadow / "direction_last.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    print(json.dumps({"day": last[0], **last[1]} if last else {"n": 0, "note": "no finished day with scores"}))


if __name__ == "__main__":
    main()
