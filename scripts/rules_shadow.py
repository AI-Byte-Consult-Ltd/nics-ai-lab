"""Nightly shadow measurement of candidate trading rules (src/nics/market/rules.py).

For every candidate rule, on 1h and 4h candles:
  - backtest: long public history (Dukascopy etc.), before our broker data starts;
  - broker history: our broker candles before the day the rule was added;
  - FORWARD: our broker candles from the day the rule was added (the real shadow test).
Per part: setups, filled, resolved, TP1 rate, average and total R after spread (spread
known only on broker candles). Nothing is traded or sent; the model does not learn
from these rules until one is promoted (see docs/rules_registry.md).

  python scripts/rules_shadow.py --snapshot data/market/<date>

Writes shadow/rules/<rule>.json, shadow/rules_last.txt (short Russian summary) and
prints the summary as JSON on the last line.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.market import rules  # noqa: E402

INTERVALS = ["1h", "4h"]
MAX_HOURS = 48


def stats(trades: list[dict]) -> dict:
    res = [t for t in trades if t["label"] is not None]
    r = np.array([t["r"] for t in res]) if res else np.zeros(0)
    return {"setups": len(trades), "resolved": len(res), "open": len(trades) - len(res),
            "tp1_rate": round(float(np.mean([t["label"] for t in res])), 3) if res else None,
            "avg_R": round(float(r.mean()), 3) if len(r) else None,
            "total_R": round(float(r.sum()), 2)}


def measure(rule_id: str, series, added_ts: float) -> dict:
    parts = {"backtest_public": [], "broker_before": [], "forward": []}
    by_symbol: dict[str, list] = {}
    for s in series:
        if s.atr is None:
            s.atr = market.atr(s)
        max_bars = max(1, int(MAX_HOURS * 3600 / s.bar_seconds))
        for st in rules.run(rule_id, s, market.WARMUP):
            filled, y, fill, _ = rules.outcome(s, st, max_bars)
            if not filled:
                continue
            risk = abs(st["entry"] - st["sl"])
            rr = abs(st["tp1"] - st["entry"]) / risk if risk > 0 else 0.0
            cost = float(s.spread[fill]) / risk if risk > 0 else 0.0
            t = {"label": y, "r": (rr if y == 1 else -1.0) - cost if y is not None else None,
                 "t": float(s.t[fill])}
            if s.public:
                parts["backtest_public"].append(t)
            else:
                part = "forward" if t["t"] >= added_ts else "broker_before"
                parts[part].append(t)
                if part == "forward":
                    by_symbol.setdefault(s.symbol, []).append(t)
    out = {k: stats(v) for k, v in parts.items()}
    out["forward_by_symbol"] = {k: stats(v) for k, v in sorted(by_symbol.items())}
    return out


def russian(rule_id: str, meta, m: dict) -> str:
    def line(name, x):
        if not x["resolved"]:
            return f"{name}: сделок с итогом пока нет (открыто {x['open']})"
        return (f"{name}: {x['resolved']} сделок, TP1 {x['tp1_rate'] * 100:.0f}%, "
                f"{x['avg_R']:+.2f} R на сделку, всего {x['total_R']:+.1f} R")
    return "\n".join([
        f"Правило в тени: {rule_id} (идея: {meta[1]}), добавлено {meta[2]}",
        line("Вперёд, наши свечи с даты добавления (главное)", m["forward"]),
        line("Наша история до добавления", m["broker_before"]),
        line("Публичная история (бэктест, без спреда)", m["backtest_public"]),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/trading/nano.yaml"))
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    shadow = ROOT / cfg.paths.experiments_dir / "shadow"
    (shadow / "rules").mkdir(parents=True, exist_ok=True)
    series = market.load_series(Path(args.snapshot), use_public=True, intervals=INTERVALS)
    summary, texts = {}, []
    for rule_id, meta in rules.CANDIDATES.items():
        if meta[3] != "shadow":
            continue
        added = datetime.fromisoformat(meta[2]).replace(tzinfo=timezone.utc).timestamp()
        m = measure(rule_id, series, added)
        rep = {"rule": rule_id, "inspired_by": meta[1], "added": meta[2], "intervals": INTERVALS,
               "note": "Shadow measurement on past candles, not a forecast. Stop entries, fill required; "
                       "a bar touching both stop and target counts as a loss.",
               **m, "updated_at": datetime.now(timezone.utc).isoformat()}
        (shadow / "rules" / f"{rule_id}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
        summary[rule_id] = {k: m[k] for k in ("forward", "broker_before", "backtest_public")}
        texts.append(russian(rule_id, meta, m))
    if texts:
        text = "\n\n".join(texts)
        (shadow / "rules_last.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
