"""Build the NICS Language text corpus from registered, licence-checked sources.

Sources are listed in data_registry/language_sources.json. A source is used only
when its entry states a licence, allows commercial use and carries the date the
licence was actually read (license_checked). Anything else is refused: the model
must not be trained on text whose licence nobody checked.

Output (not tracked in git):
  data/processed/language/text/<lang>/<source_id>/<doc>.txt   cleaned text
and one provenance entry per source in data_registry/datasets.json with the
checksum of every document.

Usage:
  python scripts/prepare_language_corpus.py                 # every usable source
  python scripts/prepare_language_corpus.py --source nics-own-texts
  python scripts/prepare_language_corpus.py --list
"""
import argparse
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
REGISTRY = ROOT / "data_registry" / "datasets.json"
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


def docs_local(source: dict):
    base = ROOT / source["path"]
    for lang in LANGS:
        for path in sorted((base / lang).glob("*.txt")) if (base / lang).is_dir() else []:
            yield lang, path.stem, path.read_text(encoding="utf-8")


def build_source(source: dict, delay: float) -> dict | None:
    sid = source["source_id"]
    docs = docs_gutenberg(source, delay) if source["kind"] == "gutenberg" else docs_local(source)
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
        "source_url": source.get("license_url"),
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
        print(f"Building {sid} ...")
        entry = build_source(source, args.delay)
        if entry:
            registry = [d for d in registry if d["dataset_id"] != entry["dataset_id"]] + [entry]
            built += 1
            print(f"  {entry['documents']} documents, {entry['size_bytes']:,} bytes")
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sources built: {built}. Text is in {TEXT}")


if __name__ == "__main__":
    main()
