"""NICS Vision Geo, step 2: select, download, resize, de-duplicate and split.

  python scripts/geo_build.py

Images are downloaded once into data/geo/cache (resized PNG, no EXIF) and re-used by
later builds. Output: data/geo/built/{images.npy, meta.jsonl, summary.json}.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.geo import dataset, sources  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/vision/geo_micro.yaml"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache-only", action="store_true", help="use only photos already downloaded (no network)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    rows = [json.loads(l) for l in (ROOT / d["manifest"]).read_text(encoding="utf-8").splitlines() if l]
    chosen = dataset.select(rows, int(d["min_images"]), int(d["max_per_class"]), int(d["max_per_specimen"]),
                            seed=int(cfg["seed"]), exclude_share_alike=bool(d.get("exclude_share_alike")))
    print(f"selected {len(chosen)} of {len(rows)} manifest rows")
    if args.cache_only:
        cache = ROOT / d["cache_dir"]
        chosen = [r for r in chosen if dataset.cache_path(cache, r["image_id"], int(d["cache_size"])).exists()]
        print(f"cache only: {len(chosen)} already downloaded")
    summary = dataset.build(chosen, ROOT / d["built_dir"], ROOT / d["cache_dir"], int(d["cache_size"]),
                            sources.http_get_quick, seed=int(cfg["seed"]), workers=args.workers)
    print(json.dumps({k: summary[k] for k in ("images", "failed_downloads", "near_duplicates_dropped", "per_split",
                                              "sources", "licences")}, indent=1))
    print("classes:", len(summary["classes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
