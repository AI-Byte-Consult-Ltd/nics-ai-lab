"""Checkpoint save/resume utilities.

Two distinct artifact types, per docs/architecture.md:
  - trainer-state.pt   -- full resume checkpoint (model+optimizer+scheduler+step)
  - model.safetensors  -- distribution weights only, for export/inference
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def save_trainer_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "config": config,
            "rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_trainer_state(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"])
    return state


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_release(
    out_dir: Path,
    model: torch.nn.Module,
    manifest: dict,
) -> None:
    """Export distribution weights: safetensors + config + manifest + checksums."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # safetensors requires contiguous tensors and no shared-storage aliases;
    # the text model ties lm_head.weight to tok_emb.weight, so clone before saving.
    state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out_dir / "model.safetensors"))

    (out_dir / "training-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    checksums = {}
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            checksums[f.name] = sha256_of(f)
    (out_dir / "sha256sums.txt").write_text(
        "\n".join(f"{h}  {name}" for name, h in checksums.items()) + "\n",
        encoding="utf-8",
    )
