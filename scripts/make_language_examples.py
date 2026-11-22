"""Write our own "facts in, text out" examples for NICS Language tasks.

The examples come from the phrase bank in src/nics/text/examples.py (written by us).
No output of any other model is used. Train and validation use different random
seeds; the files are read by scripts/encode_corpus.py.

  data/processed/language/tasks/journal/train.jsonl   one record per line (a JSON string)
  data/processed/language/tasks/journal/val.jsonl

Usage:
  python scripts/make_language_examples.py --train 45000 --val 2000
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.text import examples  # noqa: E402

OUT = ROOT / "data" / "processed" / "language" / "tasks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=45000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    folder = Path(args.out) / "journal"
    folder.mkdir(parents=True, exist_ok=True)
    for name, count, seed in (("train", args.train, 20261122), ("val", args.val, 20261123)):
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in itertools.islice(examples.stream(seed), count):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"journal {name}: {count} records -> {folder / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
