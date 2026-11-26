"""Write our own "facts in, text out" examples for NICS Language tasks.

The examples come from the phrase bank in src/nics/text/examples.py (written by us).
No output of any other model is used. Hand-written product examples from
data_registry/language_tasks/*.jsonl are added as tasks of their own. Train and validation use different random
seeds; the files are read by scripts/encode_corpus.py.

  data/processed/language/tasks/journal/train.jsonl   one record per line (a JSON string)
  data/processed/language/tasks/journal/val.jsonl

Usage:
  python scripts/make_language_examples.py --train 45000 --val 2000
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.text import examples  # noqa: E402

OUT = ROOT / "data" / "processed" / "language" / "tasks"
TASK_FILES = ROOT / "data_registry" / "language_tasks"
# one ready record: task, language, input fields, answer; only the control tokens of this shape
RECORD = re.compile(r"<\|task:[a-z]+\|><\|lang:[a-z]{2}\|><\|in\|>[^<]*(<\|sep\|>[^<]*)*<\|out\|>[^<]+<\|end\|>")


def ready_task(path: Path, out: Path, repeat: int, val_percent: int = 10) -> tuple[int, int]:
    """Examples written by hand for a product (one JSON object per line with "text").

    About val_percent of the DISTINCT records go to validation (chosen by a hash of the
    text, so the split never changes) and are not trained on; the rest is repeated
    `repeat` times and shuffled, so that a few hundred records are not lost among
    millions of corpus tokens. Returns (train records written, validation records)."""
    texts = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        if not RECORD.fullmatch(text):
            raise ValueError(f"{path.name} line {number}: not a task record")
        texts.append(text)
    texts = sorted(set(texts))
    val = [t for t in texts if int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % 100 < val_percent]
    held = set(val)
    train = [t for t in texts if t not in held] * repeat
    random.Random(20261126).shuffle(train)
    folder = out / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train), ("val", val)):
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(train), len(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=45000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--repeat", type=int, default=200, help="copies of each hand-written product example")
    args = parser.parse_args()

    folder = Path(args.out) / "journal"
    folder.mkdir(parents=True, exist_ok=True)
    for name, count, seed in (("train", args.train, 20261122), ("val", args.val, 20261123)):
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in itertools.islice(examples.stream(seed), count):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"journal {name}: {count} records -> {folder / (name + '.jsonl')}")
    for path in sorted(TASK_FILES.glob("*.jsonl")) if TASK_FILES.is_dir() else []:
        n_train, n_val = ready_task(path, Path(args.out), args.repeat)
        print(f"{path.stem}: {n_train} train records ({args.repeat} copies each), {n_val} held out for validation")


if __name__ == "__main__":
    main()
