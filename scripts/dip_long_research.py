"""Check the daily "buy a short dip inside an uptrend" rule on public history (research only).

See src/nics/market/dip_long.py for the rule, the costs and the honesty notes. Nothing is sent
to users, the scanner or any EA.

  python scripts/fetch_public_history.py --snapshot data/market/<date>
  python scripts/dip_long_research.py --snapshot data/market/<date>

Writes experiments/research/dip_long_last.json and .txt (not in git).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import dip_long as dl  # noqa: E402


def text_report(res: dict) -> str:
    L = ["Покупка короткой просадки в растущем рынке (дневные свечи, после издержек)", ""]
    for g, b in res["groups"].items():
        L.append(f"== {g}: {', '.join(b['instruments'])}")
        for v, s in b["variants"].items():
            a = s["all"]
            if a.get("n", 0) < 8:
                L.append(f"  {v}: сделок {a.get('n', 0)}")
                continue
            per = " | ".join(f"{k}: {p['mean_pct']:+.2f}% (n {p['n']})" for k, p in s["periods"].items() if p.get("n", 0) >= 8)
            L.append(f"  {v}: сделок {a['n']}, в среднем {a['mean_pct']:+.2f}% за сделку, t {a['t']:+.1f}, "
                     f"в плюс {a['win'] * 100:.0f}%, держим {s['hold_days']} дн.")
            L.append(f"      по периодам: {per}")
        for k, p in b["portfolio_three_down"].items():
            if "per_year_pct" in p:
                L.append(f"  портфель {k}: {p['per_year_pct']:+.1f}% в год, просадка {p['max_drawdown_pct']}%, худший день "
                         f"{p['worst_day_pct']}%, убыточных лет {p['losing_years']} из {p['n_years']}, Шарп {p['sharpe']}")
        L.append("")
    L.append(f"Вывод по индексам: {res.get('verdict_indices', 'нет данных')}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "research")
    a = ap.parse_args()
    res = dl.run(dl.load_daily(a.snapshot))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dip_long_last.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = text_report(res)
    (a.out / "dip_long_last.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
