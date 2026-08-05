"""Download and register the first NICS Text dataset.

Usage: python scripts/prepare_text_data.py

The raw text is NOT committed to git (see .gitignore) -- this script makes
acquisition reproducible instead of vendoring the file.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = "tinyshakespeare-v1"
SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "tinyshakespeare.txt"
REGISTRY = Path(__file__).resolve().parent.parent / "data_registry" / "datasets.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL} -> {DEST}")
    subprocess.run(
        ["curl", "-sS", "-m", "60", "-o", str(DEST), SOURCE_URL],
        check=True,
    )
    checksum = sha256_of(DEST)
    size = DEST.stat().st_size

    entry = {
        "dataset_id": DATASET_ID,
        "name": "Tiny Shakespeare",
        "source": "karpathy/char-rnn (GitHub mirror of Shakespeare's complete works)",
        "source_url": SOURCE_URL,
        "version": "1",
        "acquisition_date": datetime.now(timezone.utc).isoformat(),
        "license": "Public Domain (William Shakespeare, d. 1616)",
        "permitted_use": "training, evaluation, redistribution",
        "commercial_use": True,
        "personal_data_risk": "none",
        "copyright_risk": "none -- public domain source text",
        "checksum_sha256": checksum,
        "size_bytes": size,
        "notes": (
            "First NICS Text dataset: small (~1.1MB), unambiguously licensed, "
            "used to validate the from-scratch tokenizer + training pipeline "
            "end to end (nics-text-nano). Not intended to produce a broadly "
            "capable model on its own."
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
    print(f"  size: {size} bytes, sha256: {checksum}")


if __name__ == "__main__":
    main()
