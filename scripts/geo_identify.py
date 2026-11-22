"""Identify one photo with a trained NICS Vision Geo model (local test, nothing is sent anywhere).

  python scripts/geo_identify.py photo.jpg                 # newest run, Russian answer
  python scripts/geo_identify.py photo.jpg --lang en --run experiments/vision/nics-vision-geo-micro/exp-...
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.geo import advisor, dataset, train  # noqa: E402


def newest_run() -> Path:
    runs = sorted((ROOT / "experiments/vision/nics-vision-geo-micro").glob("exp-*/model_meta.json"))
    if not runs:
        sys.exit("no trained run yet: python scripts/train_geo.py")
    return runs[-1].parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--run", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    model, meta = train.load_model(Path(args.run) if args.run else newest_run())
    img = dataset.to_square(Path(args.photo).read_bytes(), 192)
    res = advisor.answer(train.classify(model, meta, img), args.lang)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1) if args.json else res["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
