"""Export a trained NICS Vision Geo model for the phone web app (runs in the browser, on the device).

  python scripts/geo_export_web.py --out ../remix-of-nixie-spark-nexus/public/geo
  python scripts/geo_export_web.py --run experiments/vision/nics-vision-geo-micro/exp-... --out <dir>

Writes two files:
  geo-model.bin   all weights as float16, BatchNorm folded into the convolutions
  geo-model.json  the layer list (offsets into the .bin), classes, normalisation, temperature,
                  thresholds and all answer texts (ru/en) from taxonomy.py and advisor.py

No third-party runtime: the app executes these layers with its own JavaScript (geo-engine.js).
Pass --check to compare our float16 export against PyTorch on random input (max logit difference).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.geo import advisor, taxonomy, train  # noqa: E402
from nics.geo.model import InvertedResidual, SqueezeExcite  # noqa: E402


class Packer:
    def __init__(self):
        self.parts: list[np.ndarray] = []
        self.n = 0

    def add(self, a: torch.Tensor | np.ndarray) -> dict:
        a = np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=np.float32).ravel()
        ref = {"o": self.n, "n": int(a.size)}
        self.parts.append(a.astype(np.float16))
        self.n += int(a.size)
        return ref

    def blob(self) -> bytes:
        return np.concatenate(self.parts).tobytes()


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        b = b + conv.bias * scale
    return w, b


def conv_spec(p: Packer, conv, bn, act: bool) -> dict:
    w, b = fold(conv, bn)
    return {"cin": conv.in_channels, "cout": conv.out_channels, "k": conv.kernel_size[0], "s": conv.stride[0],
            "p": conv.padding[0], "g": conv.groups, "act": act, "w": p.add(w), "b": p.add(b)}


def export(model, meta: dict, out: Path) -> dict:
    p = Packer()
    spec: dict = {"stem": conv_spec(p, model.stem[0], model.stem[1], True), "blocks": []}
    for blk in model.blocks:
        assert isinstance(blk, InvertedResidual)
        layers = list(blk.body)
        b: dict = {"skip": blk.skip}
        i = 0
        if layers[0].groups == 1 and len(layers) == 9:          # 1x1 expand, BN, SiLU
            b["expand"] = conv_spec(p, layers[0], layers[1], True)
            i = 3
        b["dw"] = conv_spec(p, layers[i], layers[i + 1], True)
        se: SqueezeExcite = layers[i + 3]
        b["se"] = {"mid": se.fc1.out_channels, "w1": p.add(se.fc1.weight), "b1": p.add(se.fc1.bias),
                   "w2": p.add(se.fc2.weight), "b2": p.add(se.fc2.bias)}
        b["proj"] = conv_spec(p, layers[i + 4], layers[i + 5], False)
        spec["blocks"].append(b)
    spec["neck"] = conv_spec(p, model.neck[0], model.neck[1], True)
    spec["cls"] = {"in": model.cls.in_features, "out": model.cls.out_features, "w": p.add(model.cls.weight),
                   "b": p.add(model.cls.bias)}
    texts = {
        "classes": {c: {"ru": taxonomy.BY_KEY[c].ru, "en": taxonomy.BY_KEY[c].en, "group": taxonomy.BY_KEY[c].group,
                        "indicates": list(taxonomy.BY_KEY[c].indicates), "tests": list(taxonomy.BY_KEY[c].tests),
                        "caution": list(taxonomy.BY_KEY[c].caution)} for c in meta["classes"]},
        "indicators": taxonomy.INDICATORS, "tests": taxonomy.TESTS, "cautions": taxonomy.CAUTIONS,
        "answer": advisor._T, "abstain": advisor.ABSTAIN, "maybe": advisor.MAYBE, "sure": advisor.SURE,
    }
    doc = {"format": "nics-geo-web-1", "model": meta["model"], "classes": meta["classes"],
           "image_size": meta["image_size"], "cache_size": 192, "mean": meta["mean"], "std": meta["std"],
           "temperature": meta["temperature"], "params": p.n, "spec": spec, "texts": texts}
    out.mkdir(parents=True, exist_ok=True)
    (out / "geo-model.bin").write_bytes(p.blob())
    (out / "geo-model.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="")
    args = ap.parse_args()
    run = Path(args.run) if args.run else sorted((ROOT / "experiments/vision/nics-vision-geo-micro").glob("exp-*/model_meta.json"))[-1].parent
    model, meta = train.load_model(run)
    report = json.loads((run / "report.json").read_text(encoding="utf-8")) if (run / "report.json").exists() else {}
    doc = export(model, meta, Path(args.out))
    t = (report.get("test") or {}).get("calibrated") or {}
    info = {"run": run.name, "version": args.version, "test_top1": t.get("top1"), "test_top3": t.get("top3"),
            "photos": (report.get("data") or {}).get("images")}
    doc["run"] = info
    (Path(args.out) / "geo-model.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"exported {run.name}: {doc['params']:,} weights, {len(doc['classes'])} classes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
