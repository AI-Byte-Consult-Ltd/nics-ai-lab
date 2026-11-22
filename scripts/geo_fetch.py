"""NICS Vision Geo, step 1: collect image metadata (no pixels) into the manifest.

  python scripts/geo_fetch.py                         # sources from the config (smithsonian, nhm)
  python scripts/geo_fetch.py --sources commons       # adds Wikimedia Commons (run on the laptop)

Rows of the same source are replaced, other sources are kept, so sources can be
fetched at different times. Prints how many labelled photos each class has.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.geo import sources  # noqa: E402

SOURCE_IDS = {"smithsonian": "smithsonian-nmnh-minsci-v1", "nhm": "nhm-london-mineralogy-v1",
              "commons": "wikimedia-commons-geo-v1"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/vision/geo_micro.yaml"))
    ap.add_argument("--sources", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = [s for s in (args.sources or ",".join(cfg["data"]["sources"])).split(",") if s]
    manifest = ROOT / cfg["data"]["manifest"]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if manifest.exists():
        drop = {SOURCE_IDS[n] for n in names}
        keep = [l for l in manifest.read_text(encoding="utf-8").splitlines() if l and json.loads(l)["source"] not in drop]
    fresh = []
    for n in names:
        got = list(sources.SOURCES[n]())
        print(f"{n}: {len(got)} images with an allowed licence, {sum(1 for r in got if r['class'])} labelled")
        fresh += [json.dumps(r, ensure_ascii=False) for r in got]
    tmp = manifest.with_suffix(".tmp")
    tmp.write_text("\n".join(keep + fresh) + "\n", encoding="utf-8")
    tmp.replace(manifest)
    rows = [json.loads(l) for l in keep + fresh]
    c = Counter(r["class"] for r in rows if r["class"])
    print(f"manifest: {len(rows)} rows, {sum(c.values())} labelled, {len(c)} classes")
    for k, v in c.most_common():
        print(f"  {k:24s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
