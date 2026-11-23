"""NICS Vision Geo: fair comparison of two of our models on the same test photos.

  python scripts/geo_compare.py                      # latest local run vs the model in the phone app
  python scripts/geo_compare.py --web <dir or URL>   # where geo-model.json / geo-model.bin are

Model A is the latest local training run. Model B is a web export (scripts/geo_export_web.py), by
default the one the phone app uses. Both answer the same test photos of the local built set, split
into museum photos and photos from the open media archive, and only over the classes both models
know. The split is by specimen with a fixed seed, so a museum test photo here was not a training
photo of either model. Prints a table; nothing is trained, nothing is sent anywhere.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.geo import dataset, train  # noqa: E402

APP = "https://aibyteconsult.com/geo"


def read(where: str, name: str) -> bytes:
    if where.startswith("http"):
        req = urllib.request.Request(f"{where.rstrip('/')}/{name}", headers={"User-Agent": "nics-ai-lab"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    return (Path(where) / name).read_bytes()


class WebModel:
    """Runs a nics-geo-web-1 export (float16 weights, BatchNorm folded) with torch."""

    def __init__(self, doc: dict, blob: bytes):
        self.doc, self.spec = doc, doc["spec"]
        self.w = torch.from_numpy(np.frombuffer(blob, dtype=np.float16).astype(np.float32))

    def t(self, ref: dict) -> torch.Tensor:
        return self.w[ref["o"]:ref["o"] + ref["n"]]

    def conv(self, x: torch.Tensor, L: dict) -> torch.Tensor:
        w = self.t(L["w"]).view(L["cout"], L["cin"] // L["g"], L["k"], L["k"])
        y = F.conv2d(x, w, self.t(L["b"]), stride=L["s"], padding=L["p"], groups=L["g"])
        return F.silu(y) if L["act"] else y

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, self.spec["stem"])
        for B in self.spec["blocks"]:
            y = self.conv(x, B["expand"]) if "expand" in B else x
            y = self.conv(y, B["dw"])
            S, c = B["se"], y.shape[1]
            z = F.silu(F.linear(y.mean((2, 3)), self.t(S["w1"]).view(S["mid"], c), self.t(S["b1"])))
            y = y * torch.sigmoid(F.linear(z, self.t(S["w2"]).view(c, S["mid"]), self.t(S["b2"])))[:, :, None, None]
            y = self.conv(y, B["proj"])
            x = x + y if B["skip"] else y
        f = self.conv(x, self.spec["neck"]).mean((2, 3))
        L = self.spec["cls"]
        return F.linear(f, self.t(L["w"]).view(L["out"], L["in"]), self.t(L["b"]))


def logits_of(fn, images, idx, mean, std, size, batch=32) -> torch.Tensor:
    out = []
    for i in range(0, len(idx), batch):
        part = np.sort(idx[i:i + batch])
        with torch.no_grad():
            out.append(fn(train.eval_view(train.to_tensor(images[part], mean, std), size)))
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/vision/geo_micro.yaml"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--web", default=APP)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(args.config)
    runs = sorted((ROOT / "experiments" / "vision" / cfg["model_id"]).glob("exp-*/model_meta.json"))
    run = Path(args.run) if args.run else runs[-1].parent
    a, am = train.load_model(run)
    doc = json.loads(read(args.web, "geo-model.json").decode("utf-8"))
    b = WebModel(doc, read(args.web, "geo-model.bin"))
    images, meta, _ = dataset.load_built(ROOT / cfg["data"]["built_dir"])
    common = [c for c in am["classes"] if c in doc["classes"]]
    ia = torch.tensor([am["classes"].index(c) for c in common])
    ib = torch.tensor([doc["classes"].index(c) for c in common])
    museum = ("smithsonian", "nhm")
    print(f"A = local run {run.name}: {len(am['classes'])} classes")
    print(f"B = web export {(doc.get('run') or {}).get('run', '?')}: {len(doc['classes'])} classes")
    print(f"common classes: {len(common)}. Test photos only, one view, answers limited to the common classes.")
    print(f"{'photos':<28}{'n':>6}{'A top1':>9}{'A top3':>9}{'A bal':>8}{'B top1':>9}{'B top3':>9}{'B bal':>8}")
    result = {}
    for name, keep in (("museum", lambda s: s.startswith(museum)), ("open media archive", lambda s: not s.startswith(museum)),
                       ("all", lambda s: True)):
        idx = np.array(sorted(i for i, m in enumerate(meta)
                              if m["split"] == "test" and m["class"] in common and keep(m["source"])), dtype=np.int64)
        if len(idx) == 0:
            print(f"{name:<28}{0:>6}")
            continue
        y = torch.tensor([common.index(meta[i]["class"]) for i in idx])
        la = logits_of(lambda x: a(x)[0], images, idx, am["mean"], am["std"], am["image_size"])[:, ia]
        lb = logits_of(b, images, idx, doc["mean"], doc["std"], doc["image_size"])[:, ib]
        ma, mb = train.metrics(la, y, len(common)), train.metrics(lb, y, len(common))
        result[name] = {"n": len(idx), "A": ma, "B": mb}
        p = lambda v: f"{v * 100:.1f} %"  # noqa: E731
        print(f"{name:<28}{len(idx):>6}{p(ma['top1']):>9}{p(ma['top3']):>9}{p(ma['balanced_accuracy']):>8}"
              f"{p(mb['top1']):>9}{p(mb['top3']):>9}{p(mb['balanced_accuracy']):>8}")
    extra = [c for c in am["classes"] if c not in common]
    if extra:
        print(f"only A knows ({len(extra)}): {', '.join(extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
