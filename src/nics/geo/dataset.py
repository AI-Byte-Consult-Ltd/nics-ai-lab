"""Build the NICS Vision Geo training set from a manifest: select, download, resize,
de-duplicate, split by specimen, and store as one uint8 array.

Leakage rules (docs/architecture.md, project rules on train/val/test separation):
  * the split is decided by specimen_id, so all photos of one specimen land in the
    same split (museums photograph one specimen several times);
  * near-duplicate pictures (difference hash within MAX_HAMMING bits) are kept once,
    even across specimens and sources, so a picture re-uploaded elsewhere cannot sit
    in train and in test at the same time;
  * the split is a hash of (seed, specimen_id): adding new images never moves an old
    specimen to another split.

Pixels are re-encoded from scratch, so no EXIF (camera, GPS) survives into the cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageOps

from . import taxonomy

MAX_HAMMING = 4


def split_of(specimen_id: str, seed: int = 42, val: float = 0.1, test: float = 0.1) -> str:
    h = int(hashlib.sha256(f"{seed}:{specimen_id}".encode()).hexdigest()[:12], 16) / float(1 << 48)
    return "test" if h < test else "val" if h < test + val else "train"


def select(rows: Iterable[dict], min_images: int, max_per_class: int, max_per_specimen: int,
           seed: int = 42, exclude_share_alike: bool = False, classes: list[str] | None = None) -> list[dict]:
    """Labelled rows, at most max_per_specimen photos per specimen and max_per_class per class,
    classes with fewer than min_images dropped. Deterministic for a given seed."""
    rng = random.Random(seed)
    by_spec: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for r in rows:
        if not r.get("class") or r["image_id"] in seen:
            continue
        if classes is not None and r["class"] not in classes:
            continue
        if exclude_share_alike and str(r.get("licence", "")).startswith("CC-BY-SA"):
            continue
        seen.add(r["image_id"])
        by_spec[r["specimen_id"]].append(r)
    per_class: dict[str, list[dict]] = defaultdict(list)
    for spec in sorted(by_spec):
        imgs = sorted(by_spec[spec], key=lambda r: r["image_id"])
        rng.shuffle(imgs)
        for r in imgs[:max_per_specimen]:
            per_class[r["class"]].append(r)
    out = []
    for key in sorted(per_class):
        items = per_class[key]
        if len(items) < min_images and classes is None:
            continue
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    return out


def to_square(data: bytes, size: int) -> np.ndarray:
    """Any image -> RGB uint8 [size, size, 3]: EXIF orientation applied, short side scaled, centre crop."""
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)


def dhash(img: np.ndarray) -> int:
    g = np.asarray(Image.fromarray(img).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


class NearDupIndex:
    def __init__(self, max_hamming: int = MAX_HAMMING):
        self.max = max_hamming
        self.h = np.zeros(1024, dtype=np.uint64)
        self.n = 0

    def add_if_new(self, h: int) -> bool:
        hv = np.uint64(h)
        if self.n:
            if int(_popcount64(self.h[: self.n] ^ hv).min()) <= self.max:
                return False
        if self.n == len(self.h):
            self.h = np.concatenate([self.h, np.zeros_like(self.h)])
        self.h[self.n] = hv
        self.n += 1
        return True


def cache_path(cache_dir: Path, image_id: str, size: int) -> Path:
    d = hashlib.sha1(image_id.encode()).hexdigest()
    return cache_dir / str(size) / d[:2] / f"{d}.png"


def load_one(row: dict, cache_dir: Path, size: int, fetch: Callable[[str], bytes]) -> np.ndarray | None:
    p = cache_path(cache_dir, row["image_id"], size)
    if p.exists():
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    try:
        arr = to_square(fetch(row["url"]), size)
    except Exception:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(p)      # PNG, no metadata
    return arr


def build(rows: list[dict], out_dir: Path, cache_dir: Path, size: int, fetch: Callable[[str], bytes],
          seed: int = 42, workers: int = 8, log: Callable[[str], None] = print) -> dict:
    """Download (or read from cache) every selected row, drop failures and near-duplicates,
    write images.npy + meta.jsonl + summary.json into out_dir. Returns the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    meta: list[dict] = []
    near = NearDupIndex()
    failed = dup = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (row, arr) in enumerate(zip(rows, ex.map(lambda r: load_one(r, cache_dir, size, fetch), rows))):
            if arr is None:
                failed += 1
                continue
            if not near.add_if_new(dhash(arr)):
                dup += 1
                continue
            arrays.append(arr)
            meta.append({k: row.get(k) for k in ("image_id", "specimen_id", "source", "class", "licence",
                                                  "attribution", "record_link", "raw_name")} |
                        {"split": split_of(row["specimen_id"], seed)})
            if (i + 1) % 500 == 0:
                log(f"build: {i + 1}/{len(rows)} (failed {failed}, near-duplicates {dup})")
    classes = sorted({m["class"] for m in meta})
    np.save(out_dir / "images.npy", np.stack(arrays) if arrays else np.zeros((0, size, size, 3), np.uint8))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    counts = {s: dict(sorted(Counter(m["class"] for m in meta if m["split"] == s).items())) for s in ("train", "val", "test")}
    summary = {
        "size": size, "images": len(meta), "failed_downloads": failed, "near_duplicates_dropped": dup,
        "classes": classes, "groups": {c: taxonomy.BY_KEY[c].group for c in classes},
        "per_split": {s: sum(v.values()) for s, v in counts.items()}, "per_class": counts,
        "sources": dict(Counter(m["source"] for m in meta)), "licences": dict(Counter(m["licence"] for m in meta)),
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_built(out_dir: Path) -> tuple[np.ndarray, list[dict], dict]:
    images = np.load(out_dir / "images.npy", mmap_mode="r")
    meta = [json.loads(l) for l in open(out_dir / "meta.jsonl", encoding="utf-8")]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return images, meta, summary
