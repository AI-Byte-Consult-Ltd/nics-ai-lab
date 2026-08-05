"""Download and register the Fashion-MNIST dataset for nics-vision-nano.

Usage: python scripts/prepare_vision_data.py

Raw files are NOT committed to git (see .gitignore) -- this script makes
acquisition reproducible instead of vendoring the data. Source is the
zalandoresearch/fashion-mnist GitHub repo itself (MIT license, Zalando SE
owns the underlying images), fetched over https.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = "fashion-mnist-v1"
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fashion-mnist"
REGISTRY = Path(__file__).resolve().parent.parent / "data_registry" / "datasets.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    checksums = {}
    sizes = {}
    for filename in FILES:
        dest = DEST_DIR / filename
        url = BASE_URL + filename
        print(f"Downloading {url} -> {dest}")
        subprocess.run(["curl", "-sS", "-m", "60", "-o", str(dest), url], check=True)
        checksums[filename] = sha256_of(dest)
        sizes[filename] = dest.stat().st_size

    entry = {
        "dataset_id": DATASET_ID,
        "name": "Fashion-MNIST",
        "source": "Zalando Research (zalandoresearch/fashion-mnist)",
        "source_url": [BASE_URL + f for f in FILES],
        "version": "1",
        "acquisition_date": datetime.now(timezone.utc).isoformat(),
        "license": "MIT (Zalando SE, 2017)",
        "permitted_use": "training, evaluation, redistribution",
        "commercial_use": True,
        "personal_data_risk": "none",
        "copyright_risk": (
            "none -- images are Zalando's own product catalog photographs, "
            "MIT-licensed by the copyright holder (not a third-party scrape)"
        ),
        "checksum_sha256": checksums,
        "size_bytes": sum(sizes.values()),
        "notes": (
            "10-class grayscale clothing image classification dataset, "
            "28x28px, 60,000 train / 10,000 test images. Used to train "
            "nics-vision-nano end to end. checksum_sha256 and size_bytes "
            "are per-file (4 idx-ubyte.gz files); size_bytes total is the "
            "sum of all four."
        ),
    }

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry = []
    if REGISTRY.exists():
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        registry = [d for d in registry if d["dataset_id"] != DATASET_ID]
    registry.append(entry)
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Registered dataset '{DATASET_ID}' in {REGISTRY}")
    for filename in FILES:
        print(f"  {filename}: {sizes[filename]} bytes, sha256={checksums[filename]}")


if __name__ == "__main__":
    main()
