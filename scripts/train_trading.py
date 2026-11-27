"""Train NICS Trading (nics-trading-*) on ALL our trading data.

Data: a snapshot directory (data/market/<date>/), see src/nics/market/data.py:
  - every candle series: Vantage MT5 broker candles on every timeframe (1min..4h)
    for every instrument, plus public D1/W1 history (unless --no-public). Each bar
    carries price action in ATR units, indicators (RSI, EMA 20/50/200, MACD,
    Bollinger, stochastic, ADX, channel/level distances) and candle patterns
    (engulfing, pin bars, doji, inside/outside bars).
  - every trade setup with entry / SL / TP: all scanner signals (every mode, strong
    and weak, with the scanner's own confidence / opportunity / technical scores and
    strategy version), shadow strategies, filled EA signals, our own MT5 trades, and
    setups generated on history by our trading methodologies (src/nics/market/methods.py).
    Outcomes are re-derived from broker candles the same way for every source.

Split by wall-clock time into train / val / test (boundaries from our broker data),
keeping a sample only when its label is known before the next split starts. The
test split is touched once, at the end.

From random weights (default) or warm-started from earlier weights (--init-from,
used by the self-learning cycle). Produces, under experiments/trading/<model_id>/<exp>/:
  checkpoint-step-000000/, checkpoints/, events.jsonl, manifest.json,
  release/ (best-on-val weights + checksums + metrics.json)

metrics.json is a research measurement on past data, not a forecast and not a
promise of future results.

Usage:
  python scripts/train_trading.py --config configs/trading/nano.yaml --snapshot data/market/<date>
  python scripts/train_trading.py ... --no-public              (our data only)
  python scripts/train_trading.py ... --init-from <release dir> --max-steps 1500
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
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.common.seed import set_seed  # noqa: E402
from nics.market import data as market  # noqa: E402
from nics.telemetry.events import TelemetryWriter  # noqa: E402
from nics.training import trading as T  # noqa: E402
from nics.training.checkpoint import export_release, save_trainer_state, sha256_of  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def lr_at(step, cfg, max_steps, base_lr):
    t = cfg.training
    warm = min(t.warmup_steps, max(1, max_steps // 10))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, max_steps - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def subsample(mask: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    out = np.zeros_like(mask)
    out[idx] = True
    return out


def by(values, mask, names=None):
    out = {}
    for v in (np.unique(values[mask]) if mask.any() else []):
        out[str(names[int(v)] if names else v)] = int(((values == v) & mask).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True, help="directory written by scripts/fetch_*.py")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--no-public", action="store_true", help="train on our own data only")
    ap.add_argument("--init-from", default=None, help="release/ dir to warm-start from (same feature schema)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--trend-context", choices=["config", "on", "off"], default="config",
                    help="daily trend features (trend_context.py); a warm start keeps the schema of its weights")
    ap.add_argument("--add-ons", choices=["config", "on", "off"], default="config",
                    help="add-on setups after +1 R (methods.add_on_winners, trial); a warm start keeps its schema")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    snapshot = Path(args.snapshot)
    use_public = not args.no_public and bool(cfg.data.get("use_public", True))
    max_steps = args.max_steps or cfg.training.max_steps
    base_lr = cfg.training.learning_rate * args.lr_scale
    if args.init_from:  # warm start: same features as those weights
        T.lock_schema_from(json.loads((Path(args.init_from) / "training-manifest.json").read_text(encoding="utf-8")))
    else:
        if args.trend_context != "config":
            market.use_trend_context(args.trend_context == "on", lock=True)
        if args.add_ons != "config":
            market.use_add_ons(args.add_ons == "on", lock=True)
    t_load = time.time()
    series, H, S, info = T.build_samples(cfg, snapshot, use_public)
    print(f"Loaded {len(series)} candle series, {len(H['y'])} horizon samples, {len(S['y'])} setups "
          f"in {time.time() - t_load:.0f}s")
    if len(H["y"]) == 0:
        sys.exit("No horizon samples: snapshot has too few candles for the configured window.")
    win = T.Windows(series, cfg.data.window)
    splits = T.make_splits(H, S, cfg.data.split)
    (h_tr, h_va, h_te), (s_tr, s_va, s_te), (q1, q2), own = splits

    split_info = {
        "train_until": T.iso(q1), "test_from": T.iso(q2),
        "use_public_history": use_public,
        "horizon_samples": {"train": int(h_tr.sum()), "val": int(h_va.sum()), "test": int(h_te.sum()),
                            "train_by_timeframe": by(H["iv"], h_tr, market.INTERVALS),
                            "train_public": int((h_tr & ~own).sum())},
        "setup_samples": {"train": int(s_tr.sum()), "val": int(s_va.sum()), "test": int(s_te.sum()),
                          "public_train": int((s_tr & S["public"].astype(bool)).sum()) if len(S["y"]) else 0,
                          "public_in_val_or_test": int(((s_va | s_te) & S["public"].astype(bool)).sum())
                          if len(S["y"]) else 0,
                          "all_by_kind": by(S["kind"], np.ones(len(S["kind"]), bool)) if len(S["kind"]) else {}},
        "setups_skipped": info["skipped"],
        "label_agreement_with_tracker": info["tracker_agreement"],
        "series": info["inventory"],
    }
    print(json.dumps({k: v for k, v in split_info.items() if k != "series"}, indent=2))

    model = T.new_model(cfg)
    init = "random"
    if args.init_from:
        rel = Path(args.init_from)
        man = json.loads((rel / "training-manifest.json").read_text(encoding="utf-8"))
        if T.schema_matches(man):
            model.load_state_dict(load_file(str(rel / "model.safetensors")))
            init = f"warm-start from {man.get('experiment_id')}"
        else:
            print("Feature schema changed since those weights: training from random weights instead.")
    print(f"Model initialised ({init}): {model.num_parameters():,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.training.weight_decay)

    exp_id = args.experiment_id or datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%SZ")
    exp_dir = ROOT / cfg.paths.experiments_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    snap_manifest = snapshot / "manifest.json"
    manifest = {
        "model_id": cfg.model_id, "experiment_id": exp_id, "git_commit": git_commit(),
        "seed": cfg.seed, "config": dict(cfg), "parameter_count": model.num_parameters(),
        "init": init, "max_steps": max_steps, "learning_rate": base_lr,
        "bar_features": market.BAR_FEATURES, "signal_features": market.SIGNAL_FEATURES,
        "dataset": {
            "snapshot": snapshot.name,
            "sha256": {p.name: sha256_of(p) for p in sorted(snapshot.glob("*.csv"))},
            "exporter_manifest": json.loads(snap_manifest.read_text()) if snap_manifest.exists() else None,
        },
        "split": split_info,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    save_trainer_state(exp_dir / "checkpoint-step-000000" / "trainer-state.pt", model, opt, 0,
                       float("inf"), dict(cfg))

    tel = TelemetryWriter(exp_dir / "events.jsonl")
    tr_h, tr_s = np.flatnonzero(h_tr), np.flatnonzero(s_tr)
    s_pub = S["public"].astype(bool) if len(S["y"]) else np.zeros(0, bool)
    # public-history setups can outnumber ours 50:1: every signal batch is half ours, half public
    tr_s_own, tr_s_pub = np.flatnonzero(s_tr & ~s_pub), np.flatnonzero(s_tr & s_pub)
    if len(tr_h) == 0:
        sys.exit("No training samples before the split boundary.")
    h_va_eval = subsample(h_va, int(cfg.training.get("eval_max_samples", 20000)), rng)
    use_signal = len(tr_s) >= cfg.training.min_signal_samples
    bs, best, best_state, bad_evals = cfg.training.batch_size, float("inf"), None, 0
    t0 = time.time()
    last_step = 0
    for step in range(max_steps):
        last_step = step
        lr = lr_at(step, cfg, max_steps, base_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        ix = tr_h[rng.integers(0, len(tr_h), bs)]
        xb = torch.from_numpy(win(H["sid"][ix], H["end"][ix]))
        _, loss = model.forward_horizon(xb, torch.from_numpy(H["y"][ix].astype(np.int64)))
        if use_signal:
            if len(tr_s_own) and len(tr_s_pub):
                half = bs // 2
                jx = np.concatenate([tr_s_own[rng.integers(0, len(tr_s_own), half)],
                                     tr_s_pub[rng.integers(0, len(tr_s_pub), bs - half)]])
            else:
                jx = tr_s[rng.integers(0, len(tr_s), min(bs, len(tr_s)))]
            _, sloss = model.forward_signal(torch.from_numpy(win(S["sid"][jx], S["end"][jx])),
                                            torch.from_numpy(S["f"][jx]),
                                            torch.from_numpy(S["y"][jx].astype(np.int64)))
            loss = loss + cfg.training.signal_loss_weight * sloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        opt.step()

        if step % cfg.training.eval_interval == 0 or step == max_steps - 1:
            hv = T.eval_horizon(model, win, H, h_va_eval)
            sv = (T.signal_metrics(S["y"][s_va].astype(np.int64), T.predict_signal(model, win, S, s_va))
                  if s_va.any() else {"n": 0})
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

    S["y"] = S["y"].astype(np.int64)
    metrics = T.final_metrics(model, win, H, S, splits, cfg)
    metrics["label_agreement_with_tracker"] = info["tracker_agreement"]
    print(json.dumps(metrics, indent=2))

    manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["best_val_score"] = best
    manifest["final_step"] = last_step
    manifest["val_score"] = metrics["val_score"]
    manifest["test_score"] = metrics["test_score"]
    rel = exp_dir / "release"
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_release(rel, model, manifest)
    tel.emit("training_completed", model_id=cfg.model_id, experiment_id=exp_id,
             final_step=last_step, best_val_loss=best)
    print(f"Weights: {rel / 'model.safetensors'}")
    print(f"RELEASE_DIR={rel}")


if __name__ == "__main__":
    main()
