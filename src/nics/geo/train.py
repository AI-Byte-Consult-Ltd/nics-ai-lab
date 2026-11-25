"""Training, calibration and evaluation for NICS Vision Geo (CPU).

Choices and why:
  * class-balanced sampling (weight 1/sqrt(n) per class): museum sets are very uneven
    (thousands of meteorite slices, a few dozen pyrites); plain sampling would teach
    the model to say "meteorite";
  * augmentation that keeps colour honest: crops, flips, 90-degree turns, small
    brightness/contrast/saturation changes, NO hue shift (colour is diagnostic:
    malachite is green, azurite is blue);
  * the checkpoint kept is the one with the best balanced accuracy on VALIDATION;
    the test split is evaluated once at the end and only reported;
  * temperature scaling fitted on validation, so "70 %" means right about 7 times in 10
    (expected calibration error is reported before and after).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import taxonomy
from .model import GeoNet, GeoNetConfig, count_params


def stats(images: np.ndarray, idx: np.ndarray, sample: int = 2000, seed: int = 0) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(sample, len(idx)), replace=False) if len(idx) else idx
    x = images[np.sort(pick)].astype(np.float32) / 255.0
    return x.mean(axis=(0, 1, 2)).tolist(), (x.std(axis=(0, 1, 2)) + 1e-6).tolist()


def to_tensor(batch: np.ndarray, mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.array(batch, dtype=np.uint8)).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def augment(x: torch.Tensor, out: int, gen: torch.Generator, min_scale: float = 0.45) -> torch.Tensor:
    """x: [B,3,S,S] normalised. Random resized crop to `out`, flips, rot90, mild photometric jitter."""
    b, _, s, _ = x.shape
    res = []
    for i in range(b):
        area = float(torch.empty(1).uniform_(min_scale, 1.0, generator=gen))
        side = max(8, int(round(s * math.sqrt(area))))
        t = int(torch.randint(0, s - side + 1, (1,), generator=gen))
        l = int(torch.randint(0, s - side + 1, (1,), generator=gen))
        res.append(F.interpolate(x[i:i + 1, :, t:t + side, l:l + side], size=(out, out), mode="bilinear",
                                 align_corners=False))
    y = torch.cat(res)
    flip_h = torch.rand(b, generator=gen) < 0.5
    flip_v = torch.rand(b, generator=gen) < 0.5
    y[flip_h] = y[flip_h].flip(3)
    y[flip_v] = y[flip_v].flip(2)
    k = int(torch.randint(0, 4, (1,), generator=gen))
    y = torch.rot90(y, k, dims=(2, 3))
    bright = torch.empty(b, 1, 1, 1).uniform_(-0.25, 0.25, generator=gen)
    contrast = torch.empty(b, 1, 1, 1).uniform_(0.8, 1.2, generator=gen)
    sat = torch.empty(b, 1, 1, 1).uniform_(0.85, 1.15, generator=gen)
    grey = y.mean(1, keepdim=True)
    y = grey + (y - grey) * sat
    m = y.mean((1, 2, 3), keepdim=True)
    return (y - m) * contrast + m + bright


def eval_view(x: torch.Tensor, out: int) -> torch.Tensor:
    return F.interpolate(x, size=(out, out), mode="bilinear", align_corners=False) if x.shape[-1] != out else x


@torch.no_grad()
def predict_logits(model: GeoNet, images: np.ndarray, idx: np.ndarray, mean, std, size: int, batch: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        part = idx[i:i + batch]
        x = eval_view(to_tensor(images[np.sort(part)][np.argsort(np.argsort(part))], mean, std), size)
        out.append(model(x)[0])
    return torch.cat(out) if out else torch.zeros(0, model.cfg.num_classes)


def metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int, temperature: float = 1.0) -> dict:
    if len(y) == 0:
        return {"n": 0}
    p = F.softmax(logits / temperature, dim=1)
    top1 = p.argmax(1)
    top3 = p.topk(min(3, num_classes), dim=1).indices
    recall = []
    for c in range(num_classes):
        m = y == c
        if m.any():
            recall.append(float((top1[m] == c).float().mean()))
    return {
        "n": int(len(y)),
        "top1": float((top1 == y).float().mean()),
        "top3": float((top3 == y[:, None]).any(1).float().mean()),
        "balanced_accuracy": float(np.mean(recall)) if recall else 0.0,
        "nll": float(F.nll_loss(torch.log(p + 1e-12), y)),
        "ece": ece(p, y),
    }


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    conf, pred = p.max(1)
    acc = (pred == y).float()
    total = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1.0 / bins)
        if m.any():
            total += float(m.float().mean()) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return total


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    if len(y) == 0:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def per_class_recall(logits: torch.Tensor, y: torch.Tensor, classes: list[str]) -> dict:
    pred = logits.argmax(1)
    out = {}
    for i, c in enumerate(classes):
        m = y == i
        if m.any():
            out[c] = {"n": int(m.sum()), "recall": round(float((pred[m] == i).float().mean()), 3)}
    return out


def confusions(logits: torch.Tensor, y: torch.Tensor, classes: list[str], top: int = 10) -> list:
    pred = logits.argmax(1)
    pairs: dict[tuple[int, int], int] = {}
    for a, b in zip(y.tolist(), pred.tolist()):
        if a != b:
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    return [{"true": classes[a], "predicted": classes[b], "count": n}
            for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:top]]


def train(images: np.ndarray, meta: list[dict], summary: dict, cfg: dict, out_dir: Path,
          log: Callable[[str], None] = print, max_minutes: float | None = None,
          init_state: dict | None = None, init_from: str | None = None) -> dict:
    """Train GeoNet on a built set. Writes model.pt (state + config), model_meta.json and report.json.

    With init_state (weights of an earlier run on the same classes) training continues from them
    instead of random weights. The starting weights are scored on validation first and kept as the
    result unless a later epoch beats them, so a continued run is never worse on validation."""
    torch.manual_seed(int(cfg.get("seed", 42)))
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    classes: list[str] = summary["classes"]
    groups = sorted({taxonomy.BY_KEY[c].group for c in classes})
    ci = {c: i for i, c in enumerate(classes)}
    gi = {g: i for i, g in enumerate(groups)}
    y_all = torch.tensor([ci[m["class"]] for m in meta])
    g_all = torch.tensor([gi[taxonomy.BY_KEY[m["class"]].group] for m in meta])
    split = np.array([m["split"] for m in meta])
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    size = int(cfg.get("image_size", 160))
    mean, std = stats(images, tr)

    mc = GeoNetConfig(num_classes=len(classes), num_groups=len(groups), **(cfg.get("model") or {}))
    model = GeoNet(mc)
    if init_state is not None:
        model.load_state_dict(init_state)
    n_params = count_params(model)
    log(f"GeoNet: {n_params:,} parameters, {len(classes)} classes, {len(groups)} groups; "
        f"train {len(tr)} / val {len(va)} / test {len(te)}")

    counts = np.bincount(y_all[tr].numpy(), minlength=len(classes)).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    sample_w = torch.tensor(w[y_all[tr].numpy()], dtype=torch.double)

    epochs = int(cfg.get("epochs", 40))
    bs = int(cfg.get("batch_size", 64))
    steps_per_epoch = max(1, int(cfg.get("steps_per_epoch") or math.ceil(len(tr) / bs)))
    total = epochs * steps_per_epoch
    warm = max(1, int(0.05 * total))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-3)), weight_decay=float(cfg.get("weight_decay", 0.05)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    smooth = float(cfg.get("label_smoothing", 0.1))
    gw = float(cfg.get("group_loss_weight", 0.3))

    best = {"balanced_accuracy": -1.0}
    history = []
    start = None
    if init_state is not None:
        vm0 = metrics(predict_logits(model, images, va, mean, std, size), y_all[va], len(classes))
        best = {**vm0, "epoch": 0}
        start = {"from": init_from, **{f"val_{k}": round(v, 4) for k, v in vm0.items() if k != "n"}}
        torch.save({"state": model.state_dict(), "config": mc.__dict__}, out_dir / "model.pt")
        log(json.dumps({"epoch": 0, **start}))
    t0 = time.time()
    step = 0
    stopped_early = False
    for ep in range(epochs):
        model.train()
        pick = torch.multinomial(sample_w, steps_per_epoch * bs, replacement=True, generator=gen).numpy()
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = tr[pick[s * bs:(s + 1) * bs]]
            order = np.argsort(idx)
            x = to_tensor(images[idx[order]], mean, std)
            yb, gb = y_all[idx[order]], g_all[idx[order]]
            x = augment(x, size, gen)
            lc, lg = model(x)
            loss = F.cross_entropy(lc, yb, label_smoothing=smooth) + gw * F.cross_entropy(lg, gb, label_smoothing=smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            loss_sum += float(loss.detach())
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                stopped_early = True
                break
        vl = predict_logits(model, images, va, mean, std, size)
        vm = metrics(vl, y_all[va], len(classes))
        row = {"epoch": ep + 1, "step": step, "train_loss": round(loss_sum / max(1, s + 1), 4),
               **{f"val_{k}": round(v, 4) for k, v in vm.items() if k != "n"}, "minutes": round((time.time() - t0) / 60, 1)}
        history.append(row)
        log(json.dumps(row))
        if vm.get("balanced_accuracy", 0) > best["balanced_accuracy"]:
            best = {**vm, "epoch": ep + 1}
            torch.save({"state": model.state_dict(), "config": mc.__dict__}, out_dir / "model.pt")
        if stopped_early:
            break

    ck = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"])
    vl = predict_logits(model, images, va, mean, std, size)
    temp = fit_temperature(vl, y_all[va])
    tl = predict_logits(model, images, te, mean, std, size)
    report = {
        "model": "nics-vision-geo-micro", "parameters": n_params, "classes": classes, "groups": groups,
        "image_size": size, "mean": mean, "std": std, "temperature": round(temp, 4),
        "epochs_run": len(history), "steps": step, "stopped_early_by_time": stopped_early,
        "minutes": round((time.time() - t0) / 60, 1), "best_epoch": best.get("epoch"),
        "val": {"raw": metrics(vl, y_all[va], len(classes)), "calibrated": metrics(vl, y_all[va], len(classes), temp)},
        "test": {"raw": metrics(tl, y_all[te], len(classes)), "calibrated": metrics(tl, y_all[te], len(classes), temp),
                 "per_class": per_class_recall(tl, y_all[te], classes), "top_confusions": confusions(tl, y_all[te], classes)},
        "chance_top1": round(1.0 / len(classes), 4), "continued": start, "history": history, "data": {k: summary[k] for k in ("images", "per_split", "sources", "licences")},
    }
    (out_dir / "model_meta.json").write_text(json.dumps(
        {k: report[k] for k in ("model", "classes", "groups", "image_size", "mean", "std", "temperature")}
        | {"config": mc.__dict__}, indent=1), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_model(run_dir: Path) -> tuple[GeoNet, dict]:
    meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = GeoNet(GeoNetConfig(**ck["config"]))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, meta


@torch.no_grad()
def classify(model: GeoNet, meta: dict, image: np.ndarray, tta: bool = True) -> dict[str, float]:
    """One RGB uint8 image [S,S,3] (any S) -> {class: probability}, calibrated, 8-view average if tta."""
    x = eval_view(to_tensor(image[None], meta["mean"], meta["std"]), meta["image_size"])
    views = [x]
    if tta:
        views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
    logits = torch.stack([model(v)[0][0] for v in views]).mean(0)
    p = F.softmax(logits / meta["temperature"], dim=0)
    return {c: float(p[i]) for i, c in enumerate(meta["classes"])}
