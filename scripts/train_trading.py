"""Train NICS Trading (nics-trading-*) from random initialisation on ALL our trading data.

Data: a snapshot directory (data/market/<date>/), see src/nics/market/data.py:
  - every candle series we have: Vantage MT5 broker candles on every timeframe
    (1min, 5min, 15min, 1h, 4h) for every instrument, plus public D1/W1 history
    (unless --no-public). One model learns all of them; the timeframe and the
    source are input features.
  - every trade setup with entry / SL / TP: all scanner signals (scheduled,
    on-request, scalper, strong and weak), shadow strategies, EA manual
    signals, and our own closed trades from the MT5 terminal. Outcomes are
    re-derived from broker candles the same way for every source.

Split by wall-clock time into train / val / test (boundaries from our broker
data), keeping a sample only when its label is known before the next split
starts. The test split is touched once, at the end.

Produces, under experiments/trading/<model_id>/<experiment_id>/:
  - checkpoint-step-000000/trainer-state.pt   (proof of random init)
  - checkpoints/step-XXXXXX/trainer-state.pt  (periodic resume checkpoints)
  - events.jsonl                              (telemetry)
  - manifest.json                             (seed, git commit, config, dataset hashes, split dates)
  - release/                                  (best-on-val weights + checksums + metrics.json)

metrics.json is a research measurement on past data, not a forecast and
not a promise of future results.

Usage:
  python scripts/train_trading.py --config configs/trading/nano.yaml --snapshot data/market/<date>
  python scripts/train_trading.py ... --no-public     (our data only, no public history)
"""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.common.seed import set_seed  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.models.trading.model import NicsTradingConfig, NicsTradingModel  # noqa: E402
from nics.telemetry.events import TelemetryWriter  # noqa: E402
from nics.training.checkpoint import export_release, save_trainer_state, sha256_of  # noqa: E402

VOL_COL = market.BAR_FEATURES.index("vol_z")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


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


def build_samples(cfg, snapshot: Path, use_public: bool):
    d = cfg.data
    ivcfg = dict(d.intervals)
    series = market.load_series(snapshot, use_public=use_public, intervals=list(ivcfg))
    if not series:
        sys.exit(f"No candles in {snapshot} for intervals {list(ivcfg)}.")
    first_ok = d.window - 1 + market.ATR_PERIOD  # features are NaN until ATR is ready

    H = {k: [] for k in ("sid", "end", "y", "t0", "t1", "iv", "public")}
    inventory = {}
    for k, s in enumerate(series):
        f, a = market.bar_features(s)
        c = ivcfg[s.interval]
        stride, hz = int(c["stride"]), int(c["horizon"])
        idx = np.arange(first_ok, len(s.c) - hz, stride)
        inventory[s.key] = {"bars": int(len(s.c)), "from": _iso(s.t[0]), "to": _iso(s.t[-1])}
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

    # Trade setups from every source, labelled from broker candles of one timeframe.
    sig_iv = d.signal_interval
    broker = {s.symbol: (k, s) for k, s in enumerate(series) if not s.public and s.interval == sig_iv}
    setups, skipped = market.load_setups(snapshot, set(broker))
    tracker = market.recorded_tp1(snapshot)
    S = {k: [] for k in ("sid", "end", "f", "y", "t0", "t1", "rr", "cost", "kind", "id")}
    agree = {"n": 0, "same": 0}
    for st in setups:
        k, s = broker[st["symbol"]]
        a = s.atr
        i = market.last_closed_bar(s, st["t"])  # window = bars CLOSED at setup time
        if i < first_ok or not a[i] > 0 or np.isnan(s.feats[i - d.window + 1:i + 1]).any():
            skipped[f"{st['kind']}:no_window"] = skipped.get(f"{st['kind']}:no_window", 0) + 1
            continue
        k0 = int(np.searchsorted(s.t, st["t"], side="left"))  # outcome from the bar in progress on
        y, k_end = market.signal_outcome(s, k0, st["direction"], st["sl"], st["tp1"], d.signal_max_bars)
        if y is None:
            skipped[f"{st['kind']}:no_outcome_yet"] = skipped.get(f"{st['kind']}:no_outcome_yet", 0) + 1
            continue
        if st["id"] in tracker:
            agree["n"] += 1; agree["same"] += int(tracker[st["id"]] == y)
        sl_d, tp_d = abs(st["entry"] - st["sl"]) / a[i], abs(st["tp1"] - st["entry"]) / a[i]
        spread_atr = s.spread[i] / a[i]
        onehot = [1.0 if st["kind"] == kk else 0.0 for kk in market.SETUP_KINDS]
        S["sid"].append(k); S["end"].append(i); S["y"].append(y)
        S["f"].append([st["direction"], st["direction"] * (st["entry"] - s.c[i]) / a[i], sl_d, tp_d,
                       tp_d / sl_d, spread_atr] + onehot)
        S["t0"].append(st["t"]); S["t1"].append(s.t[k_end] + s.bar_seconds)
        S["rr"].append(tp_d / sl_d); S["cost"].append(spread_atr / sl_d)  # round-trip spread in R
        S["kind"].append(st["kind"]); S["id"].append(st["id"])
    S = {k: np.array(v, dtype=np.float32 if k == "f" else None) for k, v in S.items()}
    if len(S["f"]) == 0:
        S["f"] = np.zeros((0, len(market.SIGNAL_FEATURES)), dtype=np.float32)
    return series, H, S, {"skipped": skipped, "tracker_agreement": agree, "inventory": inventory}


def _iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat()


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


def lr_at(step, cfg):
    t = cfg.training
    if step < t.warmup_steps:
        return t.learning_rate * (step + 1) / t.warmup_steps
    prog = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    return t.learning_rate * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def subsample(mask: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    out = np.zeros_like(mask)
    out[idx] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True, help="directory written by scripts/fetch_*.py")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--no-public", action="store_true", help="train on our own data only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    snapshot = Path(args.snapshot)
    use_public = not args.no_public and bool(cfg.data.get("use_public", True))
    t_load = time.time()
    series, H, S, info = build_samples(cfg, snapshot, use_public)
    print(f"Loaded {len(series)} candle series in {time.time() - t_load:.0f}s")
    if len(H["y"]) == 0:
        sys.exit("No horizon samples: snapshot has too few candles for the configured window.")
    win = Windows(series, cfg.data.window)

    # Split boundaries come from OUR broker data; public history before them is training data.
    own = ~H["public"].astype(bool)
    q1, q2 = market.split_bounds(H["t0"][own] if own.any() else H["t0"], tuple(cfg.data.split))
    h_tr, h_va, h_te = market.apply_split(H["t0"], H["t1"], q1, q2)
    s_tr, s_va, s_te = market.apply_split(S["t0"], S["t1"], q1, q2) if len(S["y"]) else (np.zeros(0, bool),) * 3

    def by(values, mask, names=None):
        out = {}
        for v in np.unique(values[mask]) if mask.any() else []:
            out[str(names[int(v)] if names else v)] = int(((values == v) & mask).sum())
        return out

    split_info = {
        "train_until": _iso(q1), "test_from": _iso(q2),
        "use_public_history": use_public,
        "horizon_samples": {"train": int(h_tr.sum()), "val": int(h_va.sum()), "test": int(h_te.sum()),
                            "train_by_timeframe": by(H["iv"], h_tr, market.INTERVALS),
                            "train_public": int((h_tr & ~own).sum())},
        "setup_samples": {"train": int(s_tr.sum()), "val": int(s_va.sum()), "test": int(s_te.sum()),
                          "all_by_kind": by(S["kind"], np.ones(len(S["kind"]), bool)) if len(S["kind"]) else {}},
        "setups_skipped": info["skipped"],
        "label_agreement_with_tracker": info["tracker_agreement"],
        "series": info["inventory"],
    }
    print(json.dumps({k: v for k, v in split_info.items() if k != "series"}, indent=2))

    mcfg = NicsTradingConfig(
        n_features=len(market.BAR_FEATURES), n_signal_features=len(market.SIGNAL_FEATURES),
        d_model=cfg.model.d_model, n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        max_seq_len=cfg.data.window, dropout=cfg.model.dropout,
    )
    model = NicsTradingModel(mcfg)
    print(f"Model initialised (random weights): {model.num_parameters():,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)

    exp_id = args.experiment_id or datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%SZ")
    exp_dir = ROOT / cfg.paths.experiments_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    snap_manifest = snapshot / "manifest.json"
    data_files = sorted(p for p in snapshot.glob("*.csv"))
    manifest = {
        "model_id": cfg.model_id, "experiment_id": exp_id, "git_commit": git_commit(),
        "seed": cfg.seed, "config": dict(cfg), "parameter_count": model.num_parameters(),
        "bar_features": market.BAR_FEATURES, "signal_features": market.SIGNAL_FEATURES,
        "dataset": {
            "snapshot": snapshot.name,
            "sha256": {p.name: sha256_of(p) for p in data_files},
            "exporter_manifest": json.loads(snap_manifest.read_text()) if snap_manifest.exists() else None,
        },
        "split": split_info,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    save_trainer_state(exp_dir / "checkpoint-step-000000" / "trainer-state.pt", model, opt, 0,
                       float("inf"), dict(cfg))

    tel = TelemetryWriter(exp_dir / "events.jsonl")
    tr_h = np.flatnonzero(h_tr)
    tr_s = np.flatnonzero(s_tr)
    h_va_eval = subsample(h_va, int(cfg.training.get("eval_max_samples", 20000)), rng)
    use_signal = len(tr_s) >= cfg.training.min_signal_samples
    bs, best, best_state, bad_evals = cfg.training.batch_size, float("inf"), None, 0
    t0 = time.time()
    last_step = 0
    for step in range(cfg.training.max_steps):
        last_step = step
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        ix = tr_h[rng.integers(0, len(tr_h), bs)]
        xb = torch.from_numpy(win(H["sid"][ix], H["end"][ix]))
        _, loss = model.forward_horizon(xb, torch.from_numpy(H["y"][ix].astype(np.int64)))
        if use_signal:
            jx = tr_s[rng.integers(0, len(tr_s), min(bs, len(tr_s)))]
            _, sloss = model.forward_signal(torch.from_numpy(win(S["sid"][jx], S["end"][jx])),
                                            torch.from_numpy(S["f"][jx]), torch.from_numpy(S["y"][jx]))
            loss = loss + cfg.training.signal_loss_weight * sloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()

        last = step == cfg.training.max_steps - 1
        if step % cfg.training.eval_interval == 0 or last:
            hv = eval_horizon(model, win, H, h_va_eval)
            sv = signal_metrics(S["y"][s_va], predict_signal(model, win, S, s_va)) if s_va.any() else {"n": 0}
            val_score = hv.get("loss", 0.0) + (cfg.training.signal_loss_weight * sv["bce"] if sv.get("n") else 0.0)
            print(f"step {step:5d} | train {loss.item():.4f} | val_h_loss {hv.get('loss', float('nan')):.4f} "
                  f"acc {hv.get('accuracy', float('nan')):.3f} (majority {hv.get('majority_baseline')}) "
                  f"| val_signal auc {sv.get('auc')} n {sv.get('n')} | {time.time() - t0:.0f}s")
            tel.emit("training_metric", model_id=cfg.model_id, experiment_id=exp_id, step=step,
                     train_loss=round(loss.item(), 4), validation_loss=round(val_score, 4),
                     validation_accuracy=round(hv.get("accuracy", 0.0), 4), learning_rate=round(lr, 6),
                     gpu_utilisation_percent=None, checkpoint_status="not_due")
            if val_score < best - 1e-4:
                best, bad_evals = val_score, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad_evals += 1
                if bad_evals >= cfg.training.early_stop_evals:
                    print("Early stop: validation stopped improving (overfitting guard).")
                    break
        if step > 0 and step % cfg.training.checkpoint_interval == 0:
            save_trainer_state(exp_dir / "checkpoints" / f"step-{step:06d}" / "trainer-state.pt",
                               model, opt, step, best, dict(cfg))

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final numbers on the FULL val and test sets, split by timeframe / source / setup kind.
    metrics = {"note": "Measured on past data. Not a forecast and not a guarantee of future results."}
    hm = {"val": eval_horizon(model, win, H, h_va), "test": eval_horizon(model, win, H, h_te)}
    hm["test_by_timeframe"] = {market.INTERVALS[i]: eval_horizon(model, win, H, h_te & (H["iv"] == i))
                               for i in np.unique(H["iv"][h_te])} if h_te.any() else {}
    hm["test_broker_only"] = eval_horizon(model, win, H, h_te & own)
    metrics["horizon"] = hm
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
        metrics["signal"] = {
            "val": signal_metrics(S["y"][s_va], pv), "test": signal_metrics(S["y"][s_te], pt),
            "test_by_kind": {k: signal_metrics(S["y"][s_te][kinds_t == k], pt[kinds_t == k])
                             for k in np.unique(kinds_t)},
            "threshold_from_val": thr_best,
            "test_all_setups_mean_R_after_spread": float(rt.mean()),
            "test_filtered_mean_R_after_spread": float(rt[keep_t].mean()) if keep_t.any() else None,
            "test_filtered_trades": int(keep_t.sum()), "test_all_trades": int(len(rt)),
        }
    else:
        metrics["signal"] = {"note": "Not enough setups with outcomes in val/test yet."}
    metrics["label_agreement_with_tracker"] = info["tracker_agreement"]
    print(json.dumps(metrics, indent=2))

    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["best_val_score"] = best
    manifest["final_step"] = last_step
    rel = exp_dir / "release"
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_release(rel, model, manifest)
    tel.emit("training_completed", model_id=cfg.model_id, experiment_id=exp_id,
             final_step=last_step, best_val_loss=best)
    print(f"Weights: {rel / 'model.safetensors'}")


if __name__ == "__main__":
    main()
