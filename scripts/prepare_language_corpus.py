"""Build the NICS Language text corpus from registered, licence-checked sources.

Sources are listed in data_registry/language_sources.json. A source is used only
when its entry states a licence, allows commercial use and carries the date the
licence was actually read (license_checked). Anything else is refused: the model
must not be trained on text whose licence nobody checked.

Output (not tracked in git):
  data/processed/language/text/<lang>/<source_id>/<doc>.txt   cleaned text
and one provenance entry per source in data/processed/language/datasets.json with
the checksum of every document. That file lives next to the corpus and is not tracked
by git, so building the corpus on the laptop never blocks the nightly "git pull".

Usage:
  python scripts/prepare_language_corpus.py                 # every usable source
  python scripts/prepare_language_corpus.py --source nics-own-texts
  python scripts/prepare_language_corpus.py --list
  python scripts/prepare_language_corpus.py --force         # rebuild sources that are already built

Source kinds:
  local      our own texts under data/raw/language/<source>/<lang>/*.txt
  gutenberg  Project Gutenberg books by id (header and footer removed)
  opus_mono  one gzip file with one sentence per line (United Nations Parallel Corpus,
             Europarl). The file is read as a stream and reading stops at max_text_bytes,
             so only the needed part is downloaded and nothing raw is kept on disk.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.text.tokenizer import LANGUAGES as LANGS  # noqa: E402  (one list of languages for the whole lab)

SOURCES = ROOT / "data_registry" / "language_sources.json"
REGISTRY = ROOT / "data" / "processed" / "language" / "datasets.json"
RAW = ROOT / "data" / "raw" / "language"
TEXT = ROOT / "data" / "processed" / "language" / "text"
GUTENBERG_LANG = {
    "russian": "ru", "english": "en", "spanish": "es", "bulgarian": "bg", "portuguese": "pt", "german": "de",
    "french": "fr", "italian": "it", "polish": "pl", "turkish": "tr", "arabic": "ar", "chinese": "zh",
}

_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)


def refusal(source: dict) -> str | None:
    """Why this source may not be used, or None when it may."""
    if not source.get("license"):
        return "no licence stated"
    if source.get("commercial_use") is not True:
        return "commercial use is not allowed or not stated"
    if not source.get("license_checked"):
        return "licence not checked yet (license_checked is empty)"
    return None


def clean_text(text: str) -> str:
    """Normalise without changing meaning: NFC, unix newlines, no control characters."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", ""))
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"


def strip_gutenberg(text: str) -> tuple[str, str | None]:
    """Return (book body without the Project Gutenberg header/footer, language from the header)."""
    start, end = _START.search(text), _END.search(text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Project Gutenberg start/end markers not found")
    lang = re.search(r"^Language:\s*(.+)$", text[: start.start()], re.M)
    return text[start.end() : end.start()], (lang.group(1).strip().lower() if lang else None)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-builder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def docs_gutenberg(source: dict, delay: float):
    for book_id in source["items"]:
        raw = RAW / source["source_id"] / f"pg{book_id}.txt"
        if not raw.exists():
            fetch(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt", raw)
            time.sleep(delay)  # be polite to the public server
        body, header_lang = strip_gutenberg(raw.read_text(encoding="utf-8", errors="replace"))
        lang = GUTENBERG_LANG.get(header_lang or "")
        if lang != source["lang"]:
            print(f"  skipped pg{book_id}: header language {header_lang!r}, source expects {source['lang']!r}")
            continue
        yield lang, f"pg{book_id}", body


def sample_lines(lines, sample_every: int = 1, block_lines: int = 2000, min_chars: int = 25):
    """Yield usable lines: blocks of block_lines, one block kept out of every sample_every.

    Taking whole blocks keeps neighbouring sentences together (the model sees connected text),
    skipping blocks spreads the sample over more documents. Short lines and exact repeats
    (headers, formulas of procedure) are dropped."""
    seen: set[bytes] = set()
    for number, line in enumerate(lines):
        if (number // block_lines) % sample_every:
            continue
        line = line.strip()
        if len(line) < min_chars:
            continue
        key = hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()
        if key in seen:
            continue
        seen.add(key)
        yield line


def docs_from_lines(lines, lang: str, max_text_bytes: int, doc_bytes: int = 4_000_000):
    """Group lines into documents of about doc_bytes until max_text_bytes is reached."""
    part, size, total, number = [], 0, 0, 0
    for line in lines:
        part.append(line)
        size += len(line.encode("utf-8")) + 1
        if size >= doc_bytes or total + size >= max_text_bytes:
            number += 1
            total += size
            yield lang, f"part-{number:04d}", "\n".join(part)
            part, size = [], 0
            if total >= max_text_bytes:
                return
    if part:
        yield lang, f"part-{number + 1:04d}", "\n".join(part)


def docs_opus_mono(source: dict):
    req = urllib.request.Request(source["url"], headers={"User-Agent": "corpus-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, gzip.open(resp, "rt", encoding="utf-8", errors="replace") as lines:
        picked = sample_lines(lines, int(source.get("sample_every", 1)), min_chars=int(source.get("min_chars", 25)))
        yield from docs_from_lines(picked, source["lang"], int(source["max_text_bytes"]))


def is_built(source: dict, registry: list) -> bool:
    entry = next((d for d in registry if d["dataset_id"] == f"language-{source['source_id']}"), None)
    return bool(entry) and all((TEXT / f["lang"] / source["source_id"] / f"{f['doc']}.txt").exists() for f in entry["files"])


def docs_local(source: dict):
    base = ROOT / source["path"]
    for lang in LANGS:
        for path in sorted((base / lang).glob("*.txt")) if (base / lang).is_dir() else []:
            yield lang, path.stem, path.read_text(encoding="utf-8")


def build_source(source: dict, delay: float) -> dict | None:
    sid = source["source_id"]
    kind = source["kind"]
    if kind == "gutenberg":
        docs = docs_gutenberg(source, delay)
    elif kind == "opus_mono":
        docs = docs_opus_mono(source)
    elif kind == "local":
        docs = docs_local(source)
    else:
        sys.exit(f"{sid}: unknown source kind {kind!r}")
    files, total = [], 0
    for lang, name, body in docs:
        text = clean_text(body)
        if len(text) < 200:
            continue
        out = TEXT / lang / sid / f"{name}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        size = len(text.encode("utf-8"))
        total += size
        files.append({"lang": lang, "doc": name, "size_bytes": size, "sha256": sha256_text(text)})
    if not files:
        print(f"  {sid}: no documents found")
        return None
    return {
        "dataset_id": f"language-{sid}",
        "name": source["name"],
        "source": sid,
        "source_url": source.get("url") or source.get("license_url"),
        "license_url": source.get("license_url"),
        "attribution": source.get("attribution"),
        "acquisition_date": datetime.now(timezone.utc).isoformat(),
        "license": source["license"],
        "license_checked": source["license_checked"],
        "commercial_use": True,
        "permitted_use": "training, evaluation",
        "documents": len(files),
        "size_bytes": total,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", help="source_id to build (repeatable); default: all usable")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between downloads")
    parser.add_argument("--force", action="store_true", help="rebuild downloaded sources that are already built")
    args = parser.parse_args()

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    if args.list:
        for s in sources:
            print(f"{s['source_id']:28s} {s['lang']:10s} {'OK' if not refusal(s) else 'REFUSED: ' + refusal(s)}")
        return

    wanted = set(args.source or [])
    unknown = wanted - {s["source_id"] for s in sources}
    if unknown:
        sys.exit(f"Unknown source(s): {sorted(unknown)}")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else []
    built = 0
    for source in sources:
        sid = source["source_id"]
        if wanted and sid not in wanted:
            continue
        why = refusal(source)
        if why:
            print(f"REFUSED {sid}: {why}")
            if wanted:
                sys.exit(1)
            continue
        if source["kind"] == "opus_mono" and not args.force and is_built(source, registry):
            print(f"Already built: {sid} (use --force to download it again)")
            continue
        print(f"Building {sid} ...", flush=True)
        try:
            entry = build_source(source, args.delay)
        except OSError as error:  # network trouble on one source must not lose the others
            print(f"  FAILED {sid}: {error}. Run the script again to retry this source.")
            continue
        if entry:
            registry = [d for d in registry if d["dataset_id"] != entry["dataset_id"]] + [entry]
            built += 1
            print(f"  {entry['documents']} documents, {entry['size_bytes']:,} bytes", flush=True)
            REGISTRY.parent.mkdir(parents=True, exist_ok=True)
            REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sources built: {built}. Text is in {TEXT}")


if __name__ == "__main__":
    main()
