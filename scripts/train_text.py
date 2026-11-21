"""Train NICS Text from random initialisation.

Produces, under experiments/text/<model_id>/<experiment_id>/:
  - checkpoint-step-000000/trainer-state.pt   (proof of random init)
  - checkpoints/step-XXXXXX/trainer-state.pt  (periodic resume checkpoints)
  - events.jsonl                              (telemetry, same schema the API reads)
  - manifest.json                             (provenance: seed, git commit, config, dataset)
  - release/ (on completion)                  (model.safetensors + checksums)

The corpus is the token files written by scripts/encode_corpus.py with our own
tokenizer (src/nics/text/tokenizer.py). Validation documents are never in the
training file.

Usage:
  python scripts/train_text.py --config configs/text/micro.yaml
  python scripts/train_text.py --config configs/text/micro.yaml --resume <experiment_dir> --max-minutes 290
"""
import argparse
import json
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
from nics.models.text.model import NicsTextConfig, NicsTextModel  # noqa: E402
from nics.training.checkpoint import (  # noqa: E402
    export_release,
    save_trainer_state,
    load_trainer_state,
    sha256_of,
)
from nics.telemetry.events import TelemetryWriter  # noqa: E402
from nics.text.tokenizer import NicsTokenizer  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_tokenized_data(cfg) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Memory-map the encoded corpus. Returns (train, val, vocab_size, corpus manifest)."""
    if "tokens_dir" not in cfg.data:
        sys.exit(
            f"{cfg.model_id}: this config has no data.tokens_dir. It is an archived pipeline check, "
            "not a trainable NICS Language model. Use configs/text/micro.yaml."
        )
    tokens_dir = ROOT / cfg.data.tokens_dir
    manifest_path = tokens_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No encoded corpus in {tokens_dir}. Run scripts/encode_corpus.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tok_path = ROOT / cfg.data.tokenizer
    if sha256_of(tok_path) != manifest["tokenizer_sha256"]:
        sys.exit("The corpus was encoded with a different tokenizer file. Re-run scripts/encode_corpus.py.")
    vocab_size = NicsTokenizer.load(tok_path).vocab_size
    train = np.memmap(tokens_dir / "train.bin", dtype="<u2", mode="r")
    val = np.memmap(tokens_dir / "val.bin", dtype="<u2", mode="r")
    return train, val, vocab_size, manifest


def get_batch(data: np.ndarray, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def estimate_loss(model, data, cfg, block_size: int) -> float:
    """Loss on FIXED validation windows, so numbers of different runs and steps are comparable."""
    model.eval()
    losses = []
    state = torch.get_rng_state()
    torch.manual_seed(1234)
    batches = [get_batch(data, cfg.training.batch_size, block_size) for _ in range(cfg.training.eval_iters)]
    torch.set_rng_state(state)
    for x, y in batches:
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def lr_at(step: int, cfg) -> float:
    warmup = cfg.training.warmup_steps
    base_lr = cfg.training.learning_rate
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.training.max_steps - warmup)
    import math

    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="path to an existing experiment dir")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop cleanly with a checkpoint after this long (nightly runs); resume later")
    parser.add_argument("--max-steps", type=int, default=None, help="override training.max_steps")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: PyTorch's choice)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(cfg.seed)

    train_data, val_data, vocab_size, corpus = load_tokenized_data(cfg)
    print(f"Corpus: {len(train_data):,} train tokens, {len(val_data):,} val tokens, vocab={vocab_size}")

    model_cfg = NicsTextConfig(
        vocab_size=vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        pos_encoding=cfg.model.get("pos_encoding", "learned"),
    )
    model = NicsTextModel(model_cfg)
    print(f"Model initialised (random weights): {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    if args.resume:
        exp_dir = Path(args.resume)
        saved = sorted((exp_dir / "checkpoints").glob("step-*"))
        latest = saved[-1] if saved else exp_dir / "checkpoint-step-000000"
        state = load_trainer_state(latest / "trainer-state.pt", model, optimizer)
        # a periodic checkpoint is saved after its step's update; step 0 is saved before any update
        start_step = state["step"] + 1 if saved else 0
        best_val_loss = state["best_val_loss"]
        print(f"Resumed from {latest} at step {start_step}, best_val_loss={best_val_loss:.4f}")
    else:
        experiment_id = args.experiment_id or datetime.now(timezone.utc).strftime("exp-%Y%m%dT%H%M%SZ")
        exp_dir = ROOT / cfg.paths.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        start_step = 0
        best_val_loss = float("inf")

        manifest = {
            "model_id": cfg.model_id,
            "experiment_id": experiment_id,
            "git_commit": git_commit(),
            "seed": cfg.seed,
            "config": dict(cfg),
            "vocab_size": vocab_size,
            "pretrained_weights_used": None,
            "corpus": {k: corpus[k] for k in ("tokenizer_sha256", "tokens", "documents", "tokens_per_language")},
            "parameter_count": model.num_parameters(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        # Step-0 checkpoint: proof the model started from this exact random init.
        step0_dir = exp_dir / "checkpoint-step-000000"
        save_trainer_state(step0_dir / "trainer-state.pt", model, optimizer, 0, best_val_loss, dict(cfg))
        print(f"Saved step-0 (random init) checkpoint to {step0_dir}")

    telemetry = TelemetryWriter(exp_dir / "events.jsonl")
    block_size = cfg.training.get("block_size", cfg.model.max_seq_len)
    t0 = time.time()
    tokens_seen = 0
    stopped_early = False
    last_step = start_step - 1

    for step in range(start_step, cfg.training.max_steps):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, cfg.training.batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        tokens_seen += x.numel()
        last_step = step
        out_of_time = args.max_minutes is not None and (time.time() - t0) > args.max_minutes * 60
        is_last = step == cfg.training.max_steps - 1

        if step % cfg.training.eval_interval == 0 or is_last or out_of_time:
            val_loss = estimate_loss(model, val_data, cfg, block_size)
            best_val_loss = min(best_val_loss, val_loss)
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            print(
                f"step {step:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                f"| lr {lr:.6f} | tok/s {tok_per_sec:.0f}"
            )
            telemetry.emit(
                "training_metric",
                model_id=cfg.model_id,
                experiment_id=exp_dir.name,
                step=step,
                train_loss=round(loss.item(), 4),
                validation_loss=round(val_loss, 4),
                learning_rate=round(lr, 6),
                tokens_processed=tokens_seen,
                tokens_per_second=round(tok_per_sec, 1),
                gpu_utilisation_percent=None,
                checkpoint_status="not_due",
            )

        if step > 0 and (step % cfg.training.checkpoint_interval == 0 or is_last or out_of_time):
            ckpt_dir = exp_dir / "checkpoints" / f"step-{step:06d}"
            save_trainer_state(ckpt_dir / "trainer-state.pt", model, optimizer, step, best_val_loss, dict(cfg))
            print(f"  checkpoint saved: {ckpt_dir}")
            telemetry.emit(
                "checkpoint_saved",
                model_id=cfg.model_id,
                experiment_id=exp_dir.name,
                step=step,
                path=ckpt_dir.relative_to(exp_dir).as_posix(),
            )

        if out_of_time and not is_last:
            stopped_early = True
            print(f"Time budget of {args.max_minutes} min used at step {step}. Resume with --resume {exp_dir}")
            break

    if stopped_early or last_step < cfg.training.max_steps - 1:
        return

    # Final export: distribution weights.
    release_dir = exp_dir / "release"
    manifest_path = exp_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["final_step"] = cfg.training.max_steps - 1
    manifest["best_val_loss"] = best_val_loss
    export_release(release_dir, model, manifest)
    print(f"Exported release weights to {release_dir}")
    for f in sorted(release_dir.iterdir()):
        print(f"  {f.name}: sha256={sha256_of(f)}")

    telemetry.emit(
        "training_completed",
        model_id=cfg.model_id,
        experiment_id=exp_dir.name,
        final_step=cfg.training.max_steps - 1,
        best_val_loss=best_val_loss,
    )
    print(f"Experiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
