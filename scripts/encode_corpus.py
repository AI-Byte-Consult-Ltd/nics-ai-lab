"""Encode the NICS Language corpus into token files for training.

Reads   data/processed/language/text/<lang>/<source_id>/*.txt
Writes  <out>/train.bin, <out>/val.bin   (little-endian uint16 token ids)
        <out>/manifest.json              (tokenizer checksum, counts, split)

The split is by DOCUMENT, not by position: a document is entirely in train or
entirely in validation, so validation text is never seen in training. Which
side a document lands on depends only on its name (stable across runs).

Usage:
  python scripts/encode_corpus.py --tokenizer data/processed/language/tokenizer-24k.json --langs ru,en,es,bg,pt
"""
import argparse
import hashlib
import json
import sys
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.text.tokenizer import NicsTokenizer  # noqa: E402

TEXT_DIR = ROOT / "data" / "processed" / "language" / "text"
TASKS_DIR = ROOT / "data" / "processed" / "language" / "tasks"


def is_validation(doc_key: str, val_percent: int) -> bool:
    return int(hashlib.sha256(doc_key.encode("utf-8")).hexdigest(), 16) % 100 < val_percent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text-dir", default=str(TEXT_DIR))
    parser.add_argument("--out", default=None, help="default: data/processed/language/tokens-<N>k")
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--tasks-dir", default=str(TASKS_DIR),
                        help="our own task examples (scripts/make_language_examples.py); skipped when the folder is missing")
    parser.add_argument("--langs", default=None,
                        help="comma-separated languages to encode (default: every language folder). The tokenizer "
                             "is trained on all 12 languages, the first model only on the first wave.")
    args = parser.parse_args()

    tok = NicsTokenizer.load(args.tokenizer)
    if tok.vocab_size > 65536:
        sys.exit("uint16 token files support vocabularies up to 65536")
    text_dir = Path(args.text_dir)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / "language" / f"tokens-{tok.vocab_size // 1000}k"
    out.mkdir(parents=True, exist_ok=True)

    only = {x.strip() for x in args.langs.split(",") if x.strip()} if args.langs else None
    bos, eos = tok.special_id("<bos>"), tok.special_id("<eos>")
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    per_lang: dict[str, int] = {}
    t0 = time.time()
    with open(out / "train.bin", "wb") as f_train, open(out / "val.bin", "wb") as f_val:
        for path in sorted(text_dir.rglob("*.txt")):
            key = path.relative_to(text_dir).as_posix()
            lang = key.split("/", 1)[0]
            if only and lang not in only:
                continue
            split = "val" if is_validation(key, args.val_percent) else "train"
            ids = array("H", [bos])
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    ids.extend(tok.encode(line))
            ids.append(eos)
            if sys.byteorder != "little":
                ids.byteswap()
            (f_val if split == "val" else f_train).write(ids.tobytes())
            counts[split] += len(ids)
            docs[split] += 1
            per_lang[lang] = per_lang.get(lang, 0) + len(ids)
        # Our own task examples: one record per line, control tokens allowed because the
        # text was produced by our code (text from the corpus above never gets them).
        task_tokens: dict[str, int] = {}
        tasks_dir = Path(args.tasks_dir)
        for folder in sorted(p for p in tasks_dir.iterdir() if p.is_dir()) if tasks_dir.is_dir() else []:
            for split, handle in (("train", f_train), ("val", f_val)):
                path = folder / f"{split}.jsonl"
                if not path.exists():
                    continue
                ids = array("H")
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        ids.append(bos)
                        ids.extend(tok.encode(json.loads(line), allow_special=True))
                        ids.append(eos)
                if sys.byteorder != "little":
                    ids.byteswap()
                handle.write(ids.tobytes())
                counts[split] += len(ids)
                task_tokens[f"{folder.name}/{split}"] = len(ids)
    if docs["val"] == 0 or docs["train"] == 0:
        sys.exit(f"Split is empty ({docs}): add documents or change --val-percent.")

    manifest = {
        "tokenizer": Path(args.tokenizer).name,
        "tokenizer_sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
        "vocab_size": tok.vocab_size,
        "dtype": "uint16-le",
        "split": "by document, sha256(name) % 100 < val_percent",
        "val_percent": args.val_percent,
        "languages": sorted(per_lang),
        "tokens": counts,
        "documents": docs,
        "tokens_per_language": per_lang,
        "task_tokens": task_tokens,
        "encoded_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = counts["train"] + counts["val"]
    print(f"Encoded {total:,} tokens ({counts}) from {docs} documents in {time.time() - t0:.0f}s -> {out}")
    print(f"Tokens per language: {per_lang}")
    print(f"Task example tokens: {task_tokens}")


if __name__ == "__main__":
    main()
