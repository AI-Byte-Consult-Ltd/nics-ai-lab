"""Market data for NICS Trading: snapshot loading, features, labels, splits.

A snapshot is a directory (data/market/<date>/) written by:
  scripts/fetch_market_data.py   our systems, via n8n [41 LAB] Market Data Export
      candles.csv          Vantage MT5 bridge candles, all instruments, 1min..4h
      signals.csv          every NICS scanner signal (all modes, strong and weak)
      signal_events.csv    signal lifecycle (TP/SL/breakeven/time exit events)
      swing_shadow.csv     our shadow swing strategies, trade by trade
      manual_algo.csv      manual signals executed by our EA
      economic_events.csv, journal.csv, instrument_specs.csv
  scripts/fetch_public_history.py
      public_candles.csv   public D1/W1 history (Kraken, Yahoo), source per row
  scripts/fetch_mt5_history.py
      own_trades_<label>.csv  our closed positions from the MT5 terminal

Every file is optional except candles.csv; missing ones are simply skipped.

Leakage rules (docs/architecture.md, project instructions):
  - every feature at bar t uses only bars <= t;
  - a setup's window ends at the last bar CLOSED at its creation time;
  - labels look forward, so splits are by wall-clock time and a sample is
    kept only if its label is fully known before the next split starts;
  - a setup's outcome is re-derived from candles after its creation time,
    and a bar touching both TP1 and SL counts as a loss (conservative).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

ATR_PERIOD = 14
INTERVALS = ["1min", "5min", "15min", "1h", "4h", "1day", "1week"]
INTERVAL_SECONDS = {"1min": 60, "5min": 300, "15min": 900, "1h": 3600, "4h": 14400,
                    "1day": 86400, "1week": 7 * 86400}
BASE_FEATURES = [
    "ret_atr",        # (close - prev close) / ATR
    "range_atr",      # (high - low) / ATR
    "body_atr",       # (close - open) / ATR
    "upper_wick_atr",
    "lower_wick_atr",
    "atr_pct",        # ATR / close * 100: volatility regime
    "vol_z",          # volume z-score over the window
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
BAR_FEATURES = BASE_FEATURES + [f"tf_{i}" for i in INTERVALS] + ["is_public"]
SETUP_KINDS = ["scanner_scheduled", "scanner_targeted", "scanner_scalper", "shadow", "ea_manual", "own_trade"]
SIGNAL_FEATURES = [
    "direction",      # +1 long, -1 short
    "entry_dist_atr", # (entry - close at setup time) / ATR, signed by direction
    "sl_dist_atr",    # |entry - SL| / ATR
    "tp1_dist_atr",   # |TP1 - entry| / ATR
    "reward_risk",    # tp1_dist / sl_dist
    "spread_atr",     # (ask - bid) / ATR at setup time
] + [f"kind_{k}" for k in SETUP_KINDS]

# Broker / prop symbol spellings -> our symbol names (for MT5 terminal trades).
SYMBOL_ALIASES = {
    "SP500": "US500", "US500": "US500", "SPX500": "US500", "NAS100": "US100", "USTEC": "US100",
    "US100": "US100", "NDX100": "US100", "DJ30": "US30", "US30": "US30", "GER40": "GER40", "DE40": "GER40",
    "DAX40": "GER40", "JP225": "JP225", "NIKKEI225": "JP225", "JPN225": "JP225", "HK50": "HK50",
    "XAUUSD": "XAU/USD", "GOLD": "XAU/USD", "XAUUSD247": "XAUUSD247", "XAGUSD": "XAG/USD",
    "SILVER": "XAG/USD", "BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD", "CLBRENT": "UKOUSDft",
    "UKOUSD": "UKOUSDft", "UKOUSDFT": "UKOUSDft", "BRENT": "UKOUSDft", "NVDA": "NVIDIA",
    "NVIDIA": "NVIDIA", "TSLA": "TSLA", "META": "META", "AMZN": "AMAZON", "AMAZON": "AMAZON",
    "NFLX": "NFLX", "SPCX": "SPCX",
}


def _ts(value: str) -> float:
    value = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(value).timestamp()


def _f(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical_symbol(raw: str, known: set[str]) -> str | None:
    if raw in known:
        return raw
    key = "".join(ch for ch in raw.upper() if ch.isalnum())
    cands = [key] + [key[: -len(s)] for s in ("PRO", "RAW", "ECN", "M", "R") if key.endswith(s)]
    for k in cands:
        if k in SYMBOL_ALIASES and SYMBOL_ALIASES[k] in known:
            return SYMBOL_ALIASES[k]
        if len(k) == 6 and f"{k[:3]}/{k[3:]}" in known:  # EURUSD -> EUR/USD
            return f"{k[:3]}/{k[3:]}"
    return None


@dataclass
class Series:
    symbol: str
    interval: str
    public: bool
    t: np.ndarray       # unix seconds of candle OPEN, ascending
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    spread: np.ndarray  # ask - bid, 0 where unknown
    feats: np.ndarray | None = field(default=None, repr=False)
    atr: np.ndarray | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return f"{'public' if self.public else 'broker'}|{self.symbol}|{self.interval}"

    @property
    def bar_seconds(self) -> int:
        return INTERVAL_SECONDS[self.interval]


def load_series(snapshot: Path, use_public: bool = True, intervals: list[str] | None = None) -> list[Series]:
    """All candle series in the snapshot: broker (candles.csv) + public (public_candles.csv)."""
    wanted = set(intervals or INTERVALS)
    rows: dict[tuple, list] = {}
    files = [("candles.csv", False)] + ([("public_candles.csv", True)] if use_public else [])
    for name, public in files:
        path = snapshot / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["interval"] not in wanted:
                    continue
                bid, ask = _f(r.get("bid")) or 0.0, _f(r.get("ask")) or 0.0
                rows.setdefault((r["symbol"], r["interval"], public), []).append((
                    _ts(r["candle_time"]), float(r["open"]), float(r["high"]), float(r["low"]),
                    float(r["close"]), _f(r.get("volume")) or 0.0, max(ask - bid, 0.0) if bid and ask else 0.0,
                ))
    out = []
    for (sym, iv, public), rs in sorted(rows.items()):
        rs.sort(key=lambda x: x[0])
        a = np.array(rs, dtype=np.float64)
        keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])  # drop duplicate timestamps
        a = a[keep]
        out.append(Series(sym, iv, public, a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]))
    return out


def atr(s: Series, period: int = ATR_PERIOD) -> np.ndarray:
    prev_c = np.concatenate([[s.c[0]], s.c[:-1]])
    tr = np.maximum(s.h - s.l, np.maximum(abs(s.h - prev_c), abs(s.l - prev_c)))
    out = np.full_like(tr, np.nan)
    if len(tr) >= period:
        csum = np.cumsum(tr)
        out[period - 1:] = (csum[period - 1:] - np.concatenate([[0.0], csum[:-period]])) / period
    return out


def bar_features(s: Series) -> tuple[np.ndarray, np.ndarray]:
    """(features[n, len(BAR_FEATURES)] float32, atr[n]). Rows before ATR is ready are NaN.
    Also stored on the series (s.feats, s.atr)."""
    a = atr(s)
    safe = np.where(a > 0, a, np.nan)
    prev_c = np.concatenate([[s.c[0]], s.c[:-1]])
    top = np.maximum(s.o, s.c)
    bot = np.minimum(s.o, s.c)
    hours = (s.t % 86400) / 3600.0
    dow = ((s.t // 86400) + 3) % 7  # 1970-01-01 was a Thursday
    base = np.stack([
        (s.c - prev_c) / safe,
        (s.h - s.l) / safe,
        (s.c - s.o) / safe,
        (s.h - top) / safe,
        (bot - s.l) / safe,
        a / s.c * 100.0,
        np.zeros_like(s.c),  # vol_z filled per window
        np.sin(2 * math.pi * hours / 24), np.cos(2 * math.pi * hours / 24),
        np.sin(2 * math.pi * dow / 7), np.cos(2 * math.pi * dow / 7),
    ], axis=1)
    base = np.clip(base, -20, 20)
    # after ATR is ready, a zero ATR (flat bars) gives NaN: treat as no movement
    base[ATR_PERIOD - 1:] = np.nan_to_num(base[ATR_PERIOD - 1:], nan=0.0)
    onehot = np.zeros((len(s.c), len(INTERVALS) + 1))
    onehot[:, INTERVALS.index(s.interval)] = 1.0
    onehot[:, -1] = 1.0 if s.public else 0.0
    f = np.concatenate([base, onehot], axis=1).astype(np.float32)
    s.feats, s.atr = f, a
    return f, a


def window(features: np.ndarray, vol: np.ndarray, end: int, length: int) -> np.ndarray | None:
    """Window of `length` bars ending at index `end` (inclusive), or None if not available."""
    start = end - length + 1
    if start < 0:
        return None
    w = features[start:end + 1].copy()
    if np.isnan(w).any():
        return None
    v = vol[start:end + 1]
    sd = v.std()
    w[:, BAR_FEATURES.index("vol_z")] = (v - v.mean()) / sd if sd > 0 else 0.0
    return w.astype(np.float32)


def horizon_label(s: Series, a: np.ndarray, i: int, horizon: int, flat_atr: float) -> int | None:
    j = i + horizon
    if j >= len(s.c) or not a[i] > 0:
        return None
    move = (s.c[j] - s.c[i]) / a[i]
    return 0 if move < -flat_atr else (2 if move > flat_atr else 1)


def signal_outcome(s: Series, start: int, direction: int, sl: float, tp1: float, max_bars: int):
    """(label, resolving bar index): label 1 = TP1 before SL, 0 = SL first (or both in
    one bar). Returns (None, None) if neither is hit within max_bars."""
    for k in range(start, min(start + max_bars, len(s.c))):
        hit_sl = s.l[k] <= sl if direction > 0 else s.h[k] >= sl
        hit_tp = s.h[k] >= tp1 if direction > 0 else s.l[k] <= tp1
        if hit_sl:
            return 0, k
        if hit_tp:
            return 1, k
    return None, None


def last_closed_bar(s: Series, at: float, bar_seconds: int | None = None) -> int:
    """Index of the last candle fully closed at time `at` (candle_time = open time)."""
    return int(np.searchsorted(s.t + (bar_seconds or s.bar_seconds), at, side="right")) - 1


def load_signals(snapshot: Path) -> list[dict]:
    return _read(snapshot / "signals.csv")


def load_setups(snapshot: Path, known_symbols: set[str]) -> tuple[list[dict], dict]:
    """Every trade idea with entry / SL / TP and a time, from all our sources, in one shape:
    {id, kind, symbol, direction (+1/-1), t, entry, sl, tp1}. Returns (setups, skipped counts)."""
    out, skipped = [], {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    def add(sid, kind, sym, direction, t, entry, sl, tp1):
        if sym is None or sym not in known_symbols:
            return skip(f"{kind}:no_candles")
        if None in (t, entry, sl, tp1) or entry <= 0:
            return skip(f"{kind}:bad_levels")
        d = 1 if str(direction).upper() in ("LONG", "BUY", "1") else -1
        if (d > 0 and not (sl < entry < tp1)) or (d < 0 and not (tp1 < entry < sl)):
            return skip(f"{kind}:bad_levels")
        out.append({"id": sid, "kind": kind, "symbol": sym, "direction": d, "t": t,
                    "entry": entry, "sl": sl, "tp1": tp1})

    for r in load_signals(snapshot):
        mode = (r.get("origin_mode") or "SCHEDULED").upper()
        kind = {"TARGETED": "scanner_targeted", "SCALPER": "scanner_scalper"}.get(mode, "scanner_scheduled")
        lo, hi = _f(r.get("entry_low")), _f(r.get("entry_high"))
        entry = (lo + hi) / 2 if lo is not None and hi is not None else None
        add(r.get("id"), kind, r.get("symbol"), r.get("direction"), _ts(r["created_at"]) if r.get("created_at") else None,
            entry, _f(r.get("stop_loss")), _f(r.get("tp1")))
    for r in _read(snapshot / "swing_shadow.csv"):
        if (r.get("status") or "").upper() == "SKIPPED" or not r.get("entry_time"):
            skip("shadow:not_entered")
            continue
        add(f"shadow-{r['id']}", "shadow", r.get("symbol"), r.get("direction"), _ts(r["entry_time"]),
            _f(r.get("entry_price")), _f(r.get("stop_loss")), _f(r.get("take_profit")))
    for r in _read(snapshot / "manual_algo.csv"):
        add(f"ea-{r['id']}", "ea_manual", r.get("symbol"), r.get("direction"),
            _ts(r["created_at"]) if r.get("created_at") else None,
            _f(r.get("ea_fill_price")) or _f(r.get("entry_price")), _f(r.get("stop_loss")), _f(r.get("take_profit")))
    for path in sorted(snapshot.glob("own_trades_*.csv")):
        for r in _read(path):
            add(f"own-{r.get('label')}-{r.get('position_id')}", "own_trade",
                canonical_symbol(r.get("symbol") or "", known_symbols), r.get("direction"),
                _ts(r["open_time"]) if r.get("open_time") else None,
                _f(r.get("open_price")), _f(r.get("stop_loss")), _f(r.get("take_profit")))
    return out, skipped


def recorded_tp1(snapshot: Path) -> dict[str, int]:
    """Signal id -> 1 if our tracker recorded TP1 before SL, 0 if SL first (from signal_events).
    Used only to check that candle-derived labels agree with what the tracker saw."""
    first: dict[str, int] = {}
    for r in sorted(_read(snapshot / "signal_events.csv"), key=lambda r: (r.get("event_time") or "")):
        sid, ev = r.get("signal_id"), (r.get("event_type") or "").upper()
        if sid in first:
            continue
        if ev == "TP1_HIT":
            first[sid] = 1
        elif ev == "SL_HIT":
            first[sid] = 0
    return first


def split_bounds(start_times: np.ndarray, fractions=(0.7, 0.15)) -> tuple[float, float]:
    q1, q2 = np.quantile(start_times, [fractions[0], fractions[0] + fractions[1]])
    return float(q1), float(q2)


def apply_split(start_times: np.ndarray, label_end_times: np.ndarray, q1: float, q2: float):
    """Masks (train, val, test) for fixed wall-clock boundaries. A sample is kept only if its
    label is fully known before the next split begins (label_end < boundary); this handles
    weekend and missing-bar gaps, where a fixed number of bars spans more wall-clock time."""
    train = label_end_times < q1
    val = (start_times >= q1) & (label_end_times < q2)
    test = start_times >= q2
    return train, val, test


def time_split(start_times: np.ndarray, label_end_times: np.ndarray, fractions=(0.7, 0.15)):
    """Boundaries from quantiles of start_times, then apply_split. Returns (train, val, test, (q1, q2))."""
    q1, q2 = split_bounds(start_times, fractions)
    return (*apply_split(start_times, label_end_times, q1, q2), (q1, q2))
