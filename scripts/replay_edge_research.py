"""Search the 4.5-year scanner replay for conditions with an edge (research only).

See src/nics/market/replay_edge.py for the data, the method and the honesty rules. Nothing is
sent to users, the scanner or any EA.

  python scripts/replay_edge_research.py --data data/research/scanner_replay/trades

The trades folder (one .jsonl per instrument) comes from trades.tar.gz of the replay research
(project files, nics-trader/research/entry-direction-4y-2026-11-26). It is not in git.
Writes experiments/research/replay_edge_last.json and .txt and sends the summary to the private
dashboard (--no-push to skip).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import replay_edge as re_  # noqa: E402
from nics.telemetry.push import push  # noqa: E402


def text_report(res: dict) -> str:
    L = [f"Поиск условий с перевесом в повторе сканера: {res['rows']} сделок, {res['first']} … {res['last']}", ""]
    L.append("Все сделки, R на сделку после спреда:")
    for k, v in res["baseline"].items():
        L.append(f"  {re_.TARGET_RU[k]}: {v['all']['mean']:+.3f} (лонги {v['long']['mean']:+.3f}, шорты {v['short']['mean']:+.3f}), n {v['all']['n']}")
    L += ["", "Модель учится на прошлых годах и отбирает лучшие 20 % сделок в следующем году, который она не видела:"]
    for w in res["walk_forward"]:
        if "note" in w:
            continue
        s = w["selected_total"]
        years = ", ".join(f"{y}: {v['selected']['mean']:+.3f} (n {v['selected']['n']})" for y, v in w["years"].items())
        verdict = "НАЙДЕНО" if w["found"] else "не найдено"
        name = ("бустинг" if w["model"] == "boost" else "линейная") + (" [контроль: перемешанные исходы]" if w.get("control") else "")
        L.append(f"  {re_.TARGET_RU[w['target']]}, {name}: отобранные {s['mean']:+.3f} R (t {s['t']}, по дням {s.get('t_day', 0)}), "
                 f"все {w['all_total']['mean']:+.3f}, худшие 20 % {w['lowest_total']['mean']:+.3f} | {years} -> {verdict}")
    L += ["", "Правило «найдено»: отобранные сделки в плюсе после спреда, t >= 2 по сделкам и по дням, плюс минимум в двух невиденных годах."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "research" / "scanner_replay" / "trades"))
    ap.add_argument("--top", type=float, default=0.2)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rows = re_.load_trades(Path(a.data))
    if len(rows) < 5000:
        print(f"too few trades in {a.data}: {len(rows)}. Unpack trades.tar.gz of the replay research there.")
        sys.exit(2)
    day = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    res = {"rows": len(rows), "first": day(rows[0]["t"]), "last": day(rows[-1]["t"]), "baseline": re_.baseline(rows), "walk_forward": []}
    for tname in re_.TARGETS:
        for model in ("boost", "ridge"):
            res["walk_forward"].append(re_.walk_forward(rows, tname, top=a.top, model=model))
            print(tname, model, "done", round(time.time() - t0), "s", flush=True)
    ctrl = re_.walk_forward(rows, "base", top=a.top, shuffle_seed=7)
    ctrl["control"] = True
    res["walk_forward"].append(ctrl)
    res["found_any"] = any(w.get("found") and not w.get("control") for w in res["walk_forward"])
    out = ROOT / "experiments" / "research"
    out.mkdir(parents=True, exist_ok=True)
    txt = text_report(res)
    (out / "replay_edge_last.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "replay_edge_last.txt").write_text(txt, encoding="utf-8")
    print(txt)
    if not a.no_push:
        push("nics-trading", "eval", {"replay_edge": {"rows": res["rows"], "found_any": res["found_any"]}}, txt, cycle="replay_edge " + res["last"])


if __name__ == "__main__":
    main()
