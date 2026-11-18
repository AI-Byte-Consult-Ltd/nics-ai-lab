"""Train NICS Trading (nics-trading-*) from random initialisation on our own market data.

Data: a snapshot directory from scripts/fetch_market_data.py (Vantage MT5
candles + our scanner's signals). Split by time into train / val / test
with an embargo gap; the test split is touched once, at the end.

Produces, under experiments/trading/<model_id>/<experiment_id>/:
  - checkpoint-step-000000/trainer-state.pt   (proof of random init)
  - checkpoints/step-XXXXXX/trainer-state.pt  (periodic resume checkpoints)
  - events.jsonl                              (telemetry)
  - manifest.json                             (seed, git commit, config, dataset hash, split dates)
  - release/                                  (best-on-val weights + checksums + metrics.json)

metrics.json is a research measurement on past data, not a forecast and
not a promise of future results.

Usage:
  python scripts/train_trading.py --config configs/trading/nano.yaml --snapshot data/market/<date>
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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_samples(cfg, snapshot: Path):
    d = cfg.data
    series = market.load_candles(snapshot, d.interval)
    hx, hy, ht = [], [], []
    sx, sf, sy, st, sr, scost, sid = [], [], [], [], [], [], []
    feats = {}
    for sym, s in series.items():
        f, a = market.bar_features(s)
        feats[sym] = (s, f, a)
        for i in range(d.window - 1, len(s.c), d.stride):
            w = market.window(f, s.v, i, d.window)
            y = market.horizon_label(s, a, i, d.horizon_bars, d.flat_atr)
            if w is None or y is None:
                continue
            hx.append(w); hy.append(y); ht.append(s.t[i])

    skipped = {"no_candles": 0, "no_window": 0, "no_outcome": 0, "bad_levels": 0}
    for r in market.load_signals(snapshot):
        if r["symbol"] not in feats:
            skipped["no_candles"] += 1
            continue
        s, f, a = feats[r["symbol"]]
        t0 = market._ts(r["created_at"])
        i = int(np.searchsorted(s.t, t0, side="right")) - 1
        w = market.window(f, s.v, i, d.window) if i >= 0 else None
        if w is None or not a[i] > 0:
            skipped["no_window"] += 1
            continue
        direction = 1 if r["direction"].upper() == "LONG" else -1
        try:
            mid = (float(r["entry_low"]) + float(r["entry_high"])) / 2
            sl, tp1 = float(r["stop_loss"]), float(r["tp1"])
        except (TypeError, ValueError):
            skipped["bad_levels"] += 1
            continue
        sl_d, tp_d = abs(mid - sl) / a[i], abs(tp1 - mid) / a[i]
        if sl_d <= 0 or tp_d <= 0:
            skipped["bad_levels"] += 1
            continue
        y = market.signal_outcome(s, i + 1, direction, sl, tp1, d.signal_max_bars)
        if y is None:
            skipped["no_outcome"] += 1
            continue
        spread_atr = s.spread[i] / a[i]
        sx.append(w)
        sf.append([direction, direction * (mid - s.c[i]) / a[i], sl_d, tp_d, tp_d / sl_d, spread_atr])
        sy.append(y); st.append(t0); sr.append(tp_d / sl_d)
        scost.append(spread_atr / sl_d)  # round-trip spread in R units
        sid.append(r["id"])
    return {
        "h": (np.array(hx), np.array(hy), np.array(ht)),
        "s": (np.array(sx), np.array(sf, dtype=np.float32), np.array(sy), np.array(st),
              np.array(sr), np.array(scost), sid),
        "skipped": skipped,
        "symbols": sorted(series),
    }


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
def eval_horizon(model, x, y, bs=512):
    model.eval()
    losses, correct = [], 0
    for k in range(0, len(x), bs):
        xb, yb = torch.from_numpy(x[k:k + bs]), torch.from_numpy(y[k:k + bs]).long()
        logits, loss = model.forward_horizon(xb, yb)
        losses.append(loss.item() * len(xb)); correct += (logits.argmax(-1) == yb).sum().item()
    model.train()
    n = max(len(x), 1)
    majority = float(np.bincount(y, minlength=3).max() / n) if len(y) else None
    return {"loss": sum(losses) / n, "accuracy": correct / n, "majority_baseline": majority, "n": int(len(x))}


@torch.no_grad()
def predict_signal(model, x, f, bs=512):
    model.eval()
    out = []
    for k in range(0, len(x), bs):
        logit, _ = model.forward_signal(torch.from_numpy(x[k:k + bs]), torch.from_numpy(f[k:k + bs]))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True, help="directory with candles.csv / signals.csv")
    ap.add_argument("--experiment-id", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    snapshot = Path(args.snapshot)
    data = build_samples(cfg, snapshot)
    hx, hy, ht = data["h"]
    sx, sf, sy, st, sr, scost, sid = data["s"]
    if len(hx) == 0:
        sys.exit("No horizon samples: snapshot has too few candles for the configured window.")

    embargo = cfg.data.horizon_bars * cfg.data.bar_seconds
    h_tr, h_va, h_te, bounds = market.time_split(ht, tuple(cfg.data.split), embargo)
    q1, q2 = bounds
    s_tr = st < q1 - cfg.data.signal_max_bars * cfg.data.bar_seconds
    s_va = (st >= q1) & (st < q2 - cfg.data.signal_max_bars * cfg.data.bar_seconds)
    s_te = st >= q2
    split_info = {
        "train_until": datetime.fromtimestamp(q1, timezone.utc).isoformat(),
        "test_from": datetime.fromtimestamp(q2, timezone.utc).isoformat(),
        "horizon_samples": {"train": int(h_tr.sum()), "val": int(h_va.sum()), "test": int(h_te.sum())},
        "signal_samples": {"train": int(s_tr.sum()), "val": int(s_va.sum()), "test": int(s_te.sum())},
        "signals_skipped": data["skipped"],
        "symbols": data["symbols"],
    }
    print(json.dumps(split_info, indent=2))

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
    manifest = {
        "model_id": cfg.model_id, "experiment_id": exp_id, "git_commit": git_commit(),
        "seed": cfg.seed, "config": dict(cfg), "parameter_count": model.num_parameters(),
        "dataset": {
            "snapshot": snapshot.name,
            "candles_sha256": sha256_of(snapshot / "candles.csv"),
            "signals_sha256": sha256_of(snapshot / "signals.csv"),
            "exporter_manifest": json.loads(snap_manifest.read_text()) if snap_manifest.exists() else None,
        },
        "split": split_info,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    save_trainer_state(exp_dir / "checkpoint-step-000000" / "trainer-state.pt", model, opt, 0,
                       float("inf"), dict(cfg))

    tel = TelemetryWriter(exp_dir / "events.jsonl")
    Xh_tr, Yh_tr = hx[h_tr], hy[h_tr]
    Xs_tr, Fs_tr, Ys_tr = sx[s_tr], sf[s_tr], sy[s_tr]
    bs, best, best_state, bad_evals = cfg.training.batch_size, float("inf"), None, 0
    t0 = time.time()
    for step in range(cfg.training.max_steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        ix = np.random.randint(0, len(Xh_tr), bs)
        _, loss = model.forward_horizon(torch.from_numpy(Xh_tr[ix]), torch.from_numpy(Yh_tr[ix]).long())
        if len(Xs_tr) >= cfg.training.min_signal_samples:
            jx = np.random.randint(0, len(Xs_tr), min(bs, len(Xs_tr)))
            _, sloss = model.forward_signal(torch.from_numpy(Xs_tr[jx]), torch.from_numpy(Fs_tr[jx]),
                                            torch.from_numpy(Ys_tr[jx]))
            loss = loss + cfg.training.signal_loss_weight * sloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()

        last = step == cfg.training.max_steps - 1
        if step % cfg.training.eval_interval == 0 or last:
            hv = eval_horizon(model, hx[h_va], hy[h_va])
            sv = signal_metrics(sy[s_va], predict_signal(model, sx[s_va], sf[s_va])) if s_va.any() else {"n": 0}
            val_score = hv["loss"] + (cfg.training.signal_loss_weight * sv["bce"] if sv.get("n") else 0.0)
            print(f"step {step:5d} | train {loss.item():.4f} | val_h_loss {hv['loss']:.4f} "
                  f"acc {hv['accuracy']:.3f} (majority {hv['majority_baseline']}) | val_signal {sv} "
                  f"| {time.time() - t0:.0f}s")
            tel.emit("training_metric", model_id=cfg.model_id, experiment_id=exp_id, step=step,
                     train_loss=round(loss.item(), 4), validation_loss=round(val_score, 4),
                     validation_accuracy=round(hv["accuracy"], 4), learning_rate=round(lr, 6),
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

    # Threshold for "take the signal" chosen on VAL only, then applied once to TEST.
    metrics = {"note": "Measured on past data. Not a forecast and not a guarantee of future results."}
    metrics["horizon"] = {"val": eval_horizon(model, hx[h_va], hy[h_va]),
                          "test": eval_horizon(model, hx[h_te], hy[h_te])}
    if s_va.any() and s_te.any():
        pv, pt = predict_signal(model, sx[s_va], sf[s_va]), predict_signal(model, sx[s_te], sf[s_te])
        rv, rt = r_per_trade(sy[s_va], sr[s_va], scost[s_va]), r_per_trade(sy[s_te], sr[s_te], scost[s_te])
        thr_best, r_best = None, float(rv.mean())
        for thr in np.linspace(0.3, 0.8, 26):
            keep = pv >= thr
            if keep.sum() >= cfg.training.min_filtered_trades and rv[keep].mean() > r_best:
                thr_best, r_best = float(thr), float(rv[keep].mean())
        keep_t = pt >= thr_best if thr_best is not None else np.ones_like(pt, dtype=bool)
        metrics["signal"] = {
            "val": signal_metrics(sy[s_va], pv), "test": signal_metrics(sy[s_te], pt),
            "threshold_from_val": thr_best,
            "test_all_signals_mean_R_after_spread": float(rt.mean()),
            "test_filtered_mean_R_after_spread": float(rt[keep_t].mean()) if keep_t.any() else None,
            "test_filtered_trades": int(keep_t.sum()), "test_all_trades": int(len(rt)),
        }
    else:
        metrics["signal"] = {"note": "Not enough signals with outcomes in val/test yet."}
    print(json.dumps(metrics, indent=2))

    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["best_val_score"] = best
    rel = exp_dir / "release"
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_release(rel, model, manifest)
    tel.emit("training_completed", model_id=cfg.model_id, experiment_id=exp_id, best_val_loss=best)
    print(f"Weights: {rel / 'model.safetensors'}")


if __name__ == "__main__":
    main()
