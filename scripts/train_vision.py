"""Train NICS Vision from random initialisation on Fashion-MNIST.

Produces, under experiments/vision/<model_id>/<experiment_id>/:
  - checkpoint-step-000000/trainer-state.pt   (proof of random init)
  - checkpoints/step-XXXXXX/trainer-state.pt  (periodic resume checkpoints)
  - events.jsonl                              (telemetry, same schema the API reads)
  - manifest.json                             (provenance: seed, git commit, config, dataset)
  - release/ (on completion)                  (model.safetensors + checksums)

Usage:
  python scripts/train_vision.py --config configs/vision/nano.yaml
  python scripts/train_vision.py --config configs/vision/nano.yaml --resume <experiment_dir>
"""
import argparse
import gzip
import json
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.common.seed import set_seed  # noqa: E402
from nics.models.vision.model import NicsVisionConfig, NicsVisionModel  # noqa: E402
from nics.training.checkpoint import (  # noqa: E402
    export_release,
    save_trainer_state,
    load_trainer_state,
    sha256_of,
)
from nics.telemetry.events import TelemetryWriter  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic number for images: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data


def load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic number for labels: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_dataset(cfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_dir = ROOT / cfg.data.raw_dir
    train_images = load_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = load_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")

    def to_tensor(images: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(images).float() / 255.0  # (N, 28, 28) in [0, 1]
        x = x.unsqueeze(1)  # (N, 1, 28, 28)
        pad = (cfg.model.image_size - x.shape[-1]) // 2
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad))
        return x

    return (
        to_tensor(train_images),
        torch.from_numpy(train_labels).long(),
        to_tensor(test_images),
        torch.from_numpy(test_labels).long(),
    )


def get_batch(
    x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(x), (batch_size,))
    return x[ix], y[ix]


@torch.no_grad()
def estimate_loss(model, x, y, cfg) -> tuple[float, float]:
    model.eval()
    losses, accs = [], []
    for _ in range(cfg.training.eval_iters):
        xb, yb = get_batch(x, y, cfg.training.batch_size)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        losses.append(loss.item())
        accs.append((logits.argmax(dim=-1) == yb).float().mean().item())
    model.train()
    return sum(losses) / len(losses), sum(accs) / len(accs)


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    train_x, train_y, val_x, val_y = load_dataset(cfg)
    print(
        f"Dataset loaded: {len(train_x)} train images, {len(val_x)} val images, "
        f"{cfg.model.image_size}x{cfg.model.image_size}x{cfg.model.in_channels}"
    )

    model_cfg = NicsVisionConfig(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        image_size=cfg.model.image_size,
        base_channels=cfg.model.base_channels,
        dropout=cfg.model.dropout,
    )
    model = NicsVisionModel(model_cfg)
    print(f"Model initialised (random weights): {model.num_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    if args.resume:
        exp_dir = Path(args.resume)
        latest = sorted((exp_dir / "checkpoints").glob("step-*"))[-1]
        state = load_trainer_state(latest / "trainer-state.pt", model, optimizer)
        start_step = state["step"]
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
            "parameter_count": model.num_parameters(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        # Step-0 checkpoint: proof the model started from this exact random init.
        step0_dir = exp_dir / "checkpoint-step-000000"
        save_trainer_state(step0_dir / "trainer-state.pt", model, optimizer, 0, best_val_loss, dict(cfg))
        print(f"Saved step-0 (random init) checkpoint to {step0_dir}")

    telemetry = TelemetryWriter(exp_dir / "events.jsonl")
    t0 = time.time()
    images_seen = 0

    for step in range(start_step, cfg.training.max_steps):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xb, yb = get_batch(train_x, train_y, cfg.training.batch_size)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        images_seen += xb.shape[0]

        if step % cfg.training.eval_interval == 0 or step == cfg.training.max_steps - 1:
            val_loss, val_acc = estimate_loss(model, val_x, val_y, cfg)
            best_val_loss = min(best_val_loss, val_loss)
            elapsed = time.time() - t0
            img_per_sec = images_seen / max(elapsed, 1e-6)
            print(
                f"step {step:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                f"| val_acc {val_acc:.4f} | lr {lr:.6f} | img/s {img_per_sec:.0f}"
            )
            telemetry.emit(
                "training_metric",
                model_id=cfg.model_id,
                experiment_id=exp_dir.name,
                step=step,
                train_loss=round(loss.item(), 4),
                validation_loss=round(val_loss, 4),
                validation_accuracy=round(val_acc, 4),
                learning_rate=round(lr, 6),
                images_processed=images_seen,
                images_per_second=round(img_per_sec, 1),
                gpu_utilisation_percent=None,
                checkpoint_status="not_due",
            )

        if step > 0 and (step % cfg.training.checkpoint_interval == 0 or step == cfg.training.max_steps - 1):
            ckpt_dir = exp_dir / "checkpoints" / f"step-{step:06d}"
            save_trainer_state(ckpt_dir / "trainer-state.pt", model, optimizer, step, best_val_loss, dict(cfg))
            print(f"  checkpoint saved: {ckpt_dir}")
            telemetry.emit(
                "checkpoint_saved",
                model_id=cfg.model_id,
                experiment_id=exp_dir.name,
                step=step,
                path=str(ckpt_dir.relative_to(ROOT)),
            )

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
