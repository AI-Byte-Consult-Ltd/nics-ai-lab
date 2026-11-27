"""NICS Trading: shared pipeline used by train / eval / shadow scoring / self-learning.

  build_samples  -> horizon samples (every candle series) + trade setups (every source,
                    plus methodology setups generated on history), all labelled
  make_splits    -> train / val / test by wall-clock time (boundaries from broker data)
  final_metrics  -> metrics on val / test, by timeframe and setup kind, with test_score
  load_release   -> a trained model back from release/ (checks the feature schema)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from nics.market import data as market
from nics.market import methods as methodologies
from nics.market import trend_context
from nics.models.trading.model import NicsTradingConfig, NicsTradingModel

VOL_COL = market.BAR_FEATURES.index("vol_z")


def iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat()


class Windows:
    """Builds input windows lazily from (series index, end bar) pairs, so millions of
    candles fit in laptop memory."""

    def __init__(self, series, length):
        self.series, self.length = series, length

    def __call__(self, sid: np.ndarray, end: np.ndarray) -> np.ndarray:
        out = np.empty((len(sid), self.length, len(market.BAR_FEATURES)), dtype=np.float32)
        for n, (k, i) in enumerate(zip(sid, end)):
            s = self.series[k]
            a = i - self.length + 1
            out[n] = s.feats[a:i + 1]
            v = s.v[a:i + 1]
            sd = v.std()
            out[n, :, VOL_COL] = (v - v.mean()) / sd if sd > 0 else 0.0
        return out


def setup_features(st: dict, s, i: int, room: np.ndarray) -> list[float]:
    a = s.atr[i]
    risk = abs(st["entry"] - st["sl"])
    sl_d, tp_d = risk / a, abs(st["tp1"] - st["entry"]) / a

    def rr(tp):
        return abs(tp - st["entry"]) / risk if tp and risk > 0 else 0.0

    conf, opp, tech = st.get("confidence"), st.get("opportunity"), st.get("technical")
    has = any(v is not None for v in (conf, opp, tech))
    strat = st.get("strategy") or ""
    return ([st["direction"], st["direction"] * (st["entry"] - s.c[i]) / a, sl_d, tp_d, tp_d / sl_d,
             s.spread[i] / a, float(room[i]), rr(st.get("tp2")), rr(st.get("tp3")),
             (conf or 0) / 100, (opp or 0) / 100, float(np.clip((tech or 0) / 10, -2, 2)), float(has)]
            + [1.0 if st["kind"] == k else 0.0 for k in market.SETUP_KINDS]
            + [1.0 if strat == v else 0.0 for v in market.STRATEGIES]
            + [1.0 if strat and strat not in market.STRATEGIES else 0.0])


def build_samples(cfg, snapshot: Path, use_public: bool, label_setups: bool = True, mirror_scanner: bool = False):
    """mirror_scanner=True adds, for every scanner signal, the same idea in the opposite
    direction (id "<id>|mirror"): used only by the direction shadow test, never in training."""
    d = cfg.data
    ivcfg = dict(d.intervals)
    series = market.load_series(snapshot, use_public=use_public, intervals=list(ivcfg))
    if not series:
        sys.exit(f"No candles in {snapshot} for intervals {list(ivcfg)}.")
    market.use_trend_context(bool((cfg.get("trend_context") or {}).get("enabled", False)))  # no-op if a model locked it
    market.use_add_ons(bool((cfg.get("add_ons") or {}).get("enabled", False)))  # same rule
    if market.trend_context_on():
        # daily candles for the context even when this run trains on other timeframes only
        extra = [] if "1day" in ivcfg else market.load_series(snapshot, use_public=use_public, intervals=["1day"])
        real = trend_context.CACHE.parents[1] / "market"  # data/market/<date>: nightly snapshots
        save_to = trend_context.CACHE if real in snapshot.resolve().parents else None
        trend_context.prepare(series + extra, save_to=save_to)
    first_ok = d.window - 1 + market.WARMUP  # every feature is ready from here

    H = {k: [] for k in ("sid", "end", "y", "t0", "t1", "iv", "public")}
    inventory = {}
    for k, s in enumerate(series):
        f, a = market.bar_features(s)
        c = ivcfg[s.interval]
        stride, hz = int(c["stride"]), int(c["horizon"])
        idx = np.arange(first_ok, len(s.c) - hz, stride)
        inventory[s.key] = {"bars": int(len(s.c)), "from": iso(s.t[0]), "to": iso(s.t[-1])}
        if len(idx) == 0:
            continue
        ok = (a[idx] > 0) & ~np.isnan(f[idx - d.window + 1]).any(axis=1)
        idx = idx[ok]
        move = (s.c[idx + hz] - s.c[idx]) / a[idx]
        y = np.where(move < -d.flat_atr, 0, np.where(move > d.flat_atr, 2, 1))
        bs = s.bar_seconds
        H["sid"].append(np.full(len(idx), k)); H["end"].append(idx); H["y"].append(y)
        H["t0"].append(s.t[idx] + bs); H["t1"].append(s.t[idx + hz] + bs)
        H["iv"].append(np.full(len(idx), market.INTERVALS.index(s.interval)))
        H["public"].append(np.full(len(idx), s.public))
        inventory[s.key]["horizon_samples"] = int(len(idx))
    H = {k: np.concatenate(v) if v else np.array([]) for k, v in H.items()}

    # Trade setups from every source + methodologies, labelled from broker candles of one timeframe.
    sig_iv = d.signal_interval
    broker = {s.symbol: (k, s) for k, s in enumerate(series) if not s.public and s.interval == sig_iv}
    room = {sym: market.day_range_used(s) for sym, (_, s) in broker.items()}
    setups, skipped = market.load_setups(snapshot, set(broker))
    mcfg = cfg.get("methods") or {}
    if mcfg.get("enabled", True):
        gen = methodologies.generate(series, set(mcfg.get("intervals", ["1h", "4h"])), market.WARMUP,
                                     int(mcfg.get("cooldown_bars", 6)), public=bool(mcfg.get("on_public", True)),
                                     add_ons=market.add_ons_on())
        # broker-timeframe setups are labelled on broker candles; public-history setups on their own series
        setups += [st for st in gen if st["symbol"] in broker or series[st["series"]].public]
    if mirror_scanner:
        setups += [market.mirror_setup(st) for st in setups if st["kind"].startswith("scanner")]
    tracker = market.recorded_tp1(snapshot)
    S = {k: [] for k in ("sid", "end", "f", "y", "t0", "t1", "rr", "cost", "kind", "id", "public")}
    max_hours = d.signal_max_bars * market.INTERVAL_SECONDS[sig_iv] / 3600
    agree = {"n": 0, "same": 0}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for st in setups:
        pub = "series" in st and series[st["series"]].public
        if pub:
            k = st["series"]; s = series[k]
            if k not in room:
                room[k] = market.day_range_used(s)
        else:
            k, s = broker[st["symbol"]]
        sroom = room[k] if pub else room[st["symbol"]]
        max_bars = max(1, int(max_hours * 3600 / s.bar_seconds)) if pub else d.signal_max_bars
        a = s.atr
        i = market.last_closed_bar(s, st["t"])  # window = bars CLOSED at setup time
        if i < first_ok or not a[i] > 0 or np.isnan(s.feats[i - d.window + 1:i + 1]).any():
            skip(f"{st['kind']}:no_window")
            continue
        y, t1 = -1, np.inf
        if label_setups:
            k0 = int(np.searchsorted(s.t, st["t"], side="left"))  # outcome from the bar in progress on
            y, k_end = market.signal_outcome(s, k0, st["direction"], st["sl"], st["tp1"], max_bars)
            if y is None:
                skip(f"{st['kind']}:no_outcome_yet")
                continue
            t1 = s.t[k_end] + s.bar_seconds
            if st["id"] in tracker:
                agree["n"] += 1; agree["same"] += int(tracker[st["id"]] == y)
        f = setup_features(st, s, i, sroom)
        S["sid"].append(k); S["end"].append(i); S["y"].append(y); S["f"].append(f)
        S["t0"].append(st["t"]); S["t1"].append(t1)
        S["rr"].append(f[4]); S["cost"].append(f[5] / f[2])  # round-trip spread in R
        S["kind"].append(st["kind"]); S["id"].append(st["id"]); S["public"].append(pub)
    S = {k: np.array(v, dtype=np.float32 if k == "f" else None) for k, v in S.items()}
    if len(S["f"]) == 0:
        S["f"] = np.zeros((0, len(market.SIGNAL_FEATURES)), dtype=np.float32)
    return series, H, S, {"skipped": skipped, "tracker_agreement": agree, "inventory": inventory}


def make_splits(H, S, fractions):
    """Boundaries come from OUR broker data; public history before them is training data."""
    own = ~H["public"].astype(bool)
    q1, q2 = market.split_bounds(H["t0"][own] if own.any() else H["t0"], tuple(fractions))
    h = market.apply_split(H["t0"], H["t1"], q1, q2)
    s = market.apply_split(S["t0"], S["t1"], q1, q2) if len(S["y"]) else (np.zeros(0, bool),) * 3
    if len(S["y"]) and "public" in S:  # public-history setups only teach; val/test = our own setups
        pub = S["public"].astype(bool)
        s = (s[0], s[1] & ~pub, s[2] & ~pub)
    if len(S["y"]) and "kind" in S:  # add-on setups only teach, so validation is the same set with or without them
        add = S["kind"] == market.ADDON_KIND
        s = (s[0], s[1] & ~add, s[2] & ~add)
    return h, s, (q1, q2), own


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def r_per_trade(y, rr, cost):
    """Realised R per trade: TP1 = +reward/risk, SL = -1, both minus spread cost."""
    return np.where(y == 1, rr, -1.0) - cost


@torch.no_grad()
def eval_horizon(model, win, H, mask, bs=512):
    sid, end, y = H["sid"][mask], H["end"][mask], H["y"][mask].astype(np.int64)
    if len(y) == 0:
        return {"n": 0}
    model.eval()
    loss_sum, correct = 0.0, 0
    for k in range(0, len(y), bs):
        xb, yb = torch.from_numpy(win(sid[k:k + bs], end[k:k + bs])), torch.from_numpy(y[k:k + bs])
        logits, loss = model.forward_horizon(xb, yb)
        loss_sum += loss.item() * len(yb); correct += (logits.argmax(-1) == yb).sum().item()
    model.train()
    n = len(y)
    return {"n": int(n), "loss": loss_sum / n, "accuracy": correct / n,
            "majority_baseline": float(np.bincount(y, minlength=3).max() / n)}


@torch.no_grad()
def predict_signal(model, win, S, mask, bs=512):
    sid, end, f = S["sid"][mask], S["end"][mask], S["f"][mask]
    model.eval()
    out = []
    for k in range(0, len(sid), bs):
        logit, _ = model.forward_signal(torch.from_numpy(win(sid[k:k + bs], end[k:k + bs])),
                                        torch.from_numpy(f[k:k + bs]))
        out.append(torch.sigmoid(logit).numpy())
    model.train()
    return np.concatenate(out) if out else np.array([])


def signal_metrics(y, p):
    if len(y) == 0:
        return {"n": 0}
    eps = 1e-6
    return {
        "n": int(len(y)),
        "base_rate_tp1": float(y.mean()),
        "bce": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": auc(y, p),
    }


def final_metrics(model, win, H, S, splits, cfg) -> dict:
    (h_tr, h_va, h_te), (s_tr, s_va, s_te), (q1, q2), own = splits
    m = {"note": "Measured on past data. Not a forecast and not a guarantee of future results.",
         "train_until": iso(q1), "test_from": iso(q2)}
    hm = {"val": eval_horizon(model, win, H, h_va), "test": eval_horizon(model, win, H, h_te)}
    hm["test_by_timeframe"] = {market.INTERVALS[int(i)]: eval_horizon(model, win, H, h_te & (H["iv"] == i))
                               for i in np.unique(H["iv"][h_te])} if h_te.any() else {}
    hm["test_broker_only"] = eval_horizon(model, win, H, h_te & own)
    m["horizon"] = hm
    score = hm["test"].get("loss", 0.0)
    vscore = hm["val"].get("loss", 0.0)
    if s_va.any() and s_te.any():
        pv, pt = predict_signal(model, win, S, s_va), predict_signal(model, win, S, s_te)
        rv = r_per_trade(S["y"][s_va], S["rr"][s_va], S["cost"][s_va])
        rt = r_per_trade(S["y"][s_te], S["rr"][s_te], S["cost"][s_te])
        thr_best, r_best = None, float(rv.mean())
        for thr in np.linspace(0.3, 0.8, 26):  # threshold chosen on VAL only, applied once to TEST
            keep = pv >= thr
            if keep.sum() >= cfg.training.min_filtered_trades and rv[keep].mean() > r_best:
                thr_best, r_best = float(thr), float(rv[keep].mean())
        keep_t = pt >= thr_best if thr_best is not None else np.ones_like(pt, dtype=bool)
        kinds_t = S["kind"][s_te]
        st = signal_metrics(S["y"][s_te], pt)
        m["signal"] = {
            "val": signal_metrics(S["y"][s_va], pv), "test": st,
            "test_by_kind": {str(k): signal_metrics(S["y"][s_te][kinds_t == k], pt[kinds_t == k])
                             for k in np.unique(kinds_t)},
            "threshold_from_val": thr_best,
            "test_all_setups_mean_R_after_spread": float(rt.mean()),
            "test_filtered_mean_R_after_spread": float(rt[keep_t].mean()) if keep_t.any() else None,
            "test_filtered_trades": int(keep_t.sum()), "test_all_trades": int(len(rt)),
        }
        score += cfg.training.signal_loss_weight * st["bce"]
        vscore += cfg.training.signal_loss_weight * m["signal"]["val"]["bce"]
    else:
        m["signal"] = {"note": "Not enough setups with outcomes in val/test yet."}
    m["val_score"] = float(vscore)   # lower is better; the ONLY score used to pick a champion
    m["test_score"] = float(score)   # lower is better; reported, never used for selection
    return m


def new_model(cfg) -> NicsTradingModel:
    return NicsTradingModel(NicsTradingConfig(
        n_features=len(market.BAR_FEATURES), n_signal_features=len(market.SIGNAL_FEATURES),
        d_model=cfg.model.d_model, n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        max_seq_len=cfg.data.window, dropout=cfg.model.dropout))


def lock_schema_from(manifest: dict) -> None:
    """A model always runs with the features it was trained with (trend context on or off)."""
    feats = manifest.get("bar_features") or []
    market.use_trend_context(all(f in feats for f in trend_context.TREND_FEATURES), lock=True)
    market.use_add_ons(f"kind_{market.ADDON_KIND}" in (manifest.get("signal_features") or []), lock=True)


def schema_matches(manifest: dict) -> bool:
    return (manifest.get("bar_features") == market.BAR_FEATURES
            and manifest.get("signal_features") == market.SIGNAL_FEATURES)


def load_release(release_dir: Path, cfg) -> tuple[NicsTradingModel, dict]:
    manifest = json.loads((release_dir / "training-manifest.json").read_text(encoding="utf-8"))
    lock_schema_from(manifest)
    if not schema_matches(manifest):
        raise ValueError("feature schema of these weights differs from the current code; retrain from scratch")
    model = new_model(cfg)
    model.load_state_dict(load_file(str(release_dir / "model.safetensors")))
    model.eval()
    return model, manifest
