"""Train the NICS Language tokenizer from scratch on our own corpus.

The algorithm is ours (src/nics/text/tokenizer.py): no pretrained tokenizer and
no third-party tokenization library.

The corpus is the cleaned text written by scripts/prepare_language_corpus.py:
  data/processed/language/text/<lang>/<source_id>/*.txt
Every language gets the same byte budget, so the vocabulary is not dominated by
whichever language has the most text.

Usage:
  python scripts/train_tokenizer.py --vocab-size 24576
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.text.tokenizer import MERGE_OFFSET, NicsTokenizer, count_pieces, train_merges  # noqa: E402

TEXT_DIR = ROOT / "data" / "processed" / "language" / "text"


def read_budget(files: list[Path], max_bytes: int):
    """Yield text line by line from files until max_bytes were read."""
    seen = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
                seen += len(line.encode("utf-8"))
                if seen >= max_bytes:
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", default=str(TEXT_DIR))
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-mb-per-lang", type=float, default=200.0)
    parser.add_argument("--out", default=None, help="default: data/processed/language/tokenizer-<N>k.json")
    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    langs = sorted(p.name for p in text_dir.iterdir() if p.is_dir()) if text_dir.is_dir() else []
    if not langs:
        sys.exit(f"No corpus in {text_dir}. Run scripts/prepare_language_corpus.py first.")

    files_by_lang = {lang: sorted((text_dir / lang).rglob("*.txt")) for lang in langs}
    sizes = {lang: sum(f.stat().st_size for f in fs) for lang, fs in files_by_lang.items()}
    budget = int(min(min(sizes.values()), args.max_mb_per_lang * 1024 * 1024))
    if budget <= 0:
        sys.exit(f"A language folder is empty: {sizes}")
    print(f"Languages: {sizes} bytes; using {budget:,} bytes of each")

    t0 = time.time()
    counts: Counter = Counter()
    for lang in langs:
        counts.update(count_pieces(read_budget(files_by_lang[lang], budget)))
    print(f"Unique pieces: {len(counts):,} ({time.time() - t0:.0f}s)")

    merges = train_merges(counts, args.vocab_size, args.min_frequency, progress_every=1000)
    tok = NicsTokenizer(merges)
    out = Path(args.out) if args.out else ROOT / "data" / "processed" / "language" / f"tokenizer-{args.vocab_size // 1000}k.json"
    tok.save(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(".manifest.json")).write_text(
        json.dumps(
            {
                "format": "nics-bpe-v1",
                "vocab_size": tok.vocab_size,
                "requested_vocab_size": args.vocab_size,
                "merges": len(merges),
                "languages": langs,
                "bytes_per_language": budget,
                "min_frequency": args.min_frequency,
                "sha256": sha,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "pretrained_tokenizer_used": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Tokenizer trained from scratch: vocab={tok.vocab_size} (merges {len(merges)}, base {MERGE_OFFSET})")
    print(f"Saved {out} sha256={sha} in {time.time() - t0:.0f}s")
    if tok.vocab_size < args.vocab_size:
        print("Note: the corpus is too small to fill the requested vocabulary; the model must use the size above.")


if __name__ == "__main__":
    main()
