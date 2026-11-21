"""NICS Language tokenizer -- byte-level BPE written from scratch.

No third-party tokenization library is used: the merge-learning algorithm,
the encoder and the decoder are all in this file (standard library only).

Design
  * Lossless: decode(encode(text)) == text for any valid Unicode string.
  * Byte-level: the base alphabet is the 256 byte values, so there is no
    "unknown" token and every language of the site (Cyrillic, Latin, Arabic, Chinese) shares one vocabulary.
  * Digits never merge: every digit is its own token. Prices and levels are
    therefore never fused into opaque multi-digit tokens.
  * Special tokens (task, language, value slots) live in a reserved id block
    and are produced ONLY when the caller asks for them (allow_special=True).
    Free user text is always encoded with allow_special=False, so a user who
    types "<|task:card|>" gets plain bytes, never a control token.

Id layout (stable across vocabulary sizes)
  0 .. RESERVED-1            special tokens (named ones first, rest unused)
  RESERVED .. RESERVED+255   the 256 byte values
  RESERVED+256 ..            learned merges, in the order they were learned
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

FORMAT = "nics-bpe-v1"
RESERVED = 256
BYTE_OFFSET = RESERVED
MERGE_OFFSET = RESERVED + 256

# A piece is the unit merges are learned inside; merges never cross pieces.
#   " ?letters"  a word with its leading space
#   "digit"      one digit (so digits can never merge with anything)
#   " ?punct"    a run of punctuation with its leading space
#   "spaces"     any other whitespace run
#   "_"          underscore (the one \w character not covered above)
_PIECE_RE = re.compile(r" ?[^\W\d_]+|\d| ?[^\s\w]+|\s+|_")

# Every language of the public site. The first five are the Mini App languages
# and are the first to be trained; the order is part of the id layout: append only.
LANGUAGES: tuple[str, ...] = ("ru", "en", "es", "bg", "pt", "de", "fr", "it", "pl", "tr", "ar", "zh")
APP_LANGUAGES: tuple[str, ...] = LANGUAGES[:5]

DEFAULT_SPECIALS: tuple[str, ...] = (
    "<pad>", "<bos>", "<eos>", "<unk>",
    # layout of one training / inference record
    "<|in|>", "<|out|>", "<|sep|>", "<|end|>",
    # tasks
    "<|task:journal|>", "<|task:card|>", "<|task:digest|>", "<|task:chat|>",
    # languages (see LANGUAGES)
) + tuple(f"<|lang:{lang}|>" for lang in LANGUAGES) + (
    # value slots: the model writes the slot, code substitutes the real value.
    "<|symbol|>", "<|price|>", "<|entry|>", "<|stop|>",
    "<|tp1|>", "<|tp2|>", "<|tp3|>",
) + tuple(f"<|v{i}|>" for i in range(16))


def split_pieces(text: str) -> list[str]:
    """Split text into pieces; ''.join(split_pieces(t)) == t always."""
    return _PIECE_RE.findall(text)


def count_pieces(texts: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(_PIECE_RE.findall(text))
    return counts


def train_merges(
    piece_counts: Counter,
    vocab_size: int,
    min_frequency: int = 2,
    progress_every: int = 0,
) -> list[tuple[int, int]]:
    """Learn BPE merges from piece frequencies.

    Returns the merges as (left_id, right_id) pairs in learned order; merge i
    creates token id MERGE_OFFSET + i. Deterministic: ties are broken by the
    pair ids, never by dict order or hashing.
    """
    n_merges = vocab_size - MERGE_OFFSET
    if n_merges < 0:
        raise ValueError(f"vocab_size must be at least {MERGE_OFFSET}")

    # Unique pieces as mutable symbol lists, sorted for determinism.
    words: list[list[int]] = []
    freqs: list[int] = []
    for piece, freq in sorted(piece_counts.items()):
        raw = piece.encode("utf-8")
        if len(raw) < 2:
            continue  # single byte (every digit lands here): nothing to merge
        words.append([BYTE_OFFSET + b for b in raw])
        freqs.append(freq)

    pair_counts: dict[tuple[int, int], int] = {}
    pair_words: dict[tuple[int, int], set[int]] = {}
    for wi, word in enumerate(words):
        f = freqs[wi]
        for pair in zip(word, word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + f
            pair_words.setdefault(pair, set()).add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        current = pair_counts.get(pair, 0)
        if current != -neg:
            continue  # stale heap entry
        if current < min_frequency:
            break
        new_id = MERGE_OFFSET + len(merges)
        merges.append(pair)
        a, b = pair
        touched: dict[tuple[int, int], None] = {}
        for wi in sorted(pair_words.pop(pair, ())):
            word = words[wi]
            f = freqs[wi]
            # remove this word's contribution
            for p in zip(word, word[1:]):
                pair_counts[p] -= f
                touched[p] = None
            # apply the merge
            merged: list[int] = []
            i = 0
            n = len(word)
            while i < n:
                if i + 1 < n and word[i] == a and word[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[wi] = merged
            # add the contribution back
            for p in zip(merged, merged[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_words.setdefault(p, set()).add(wi)
                touched[p] = None
        pair_counts.pop(pair, None)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]
        if progress_every and len(merges) % progress_every == 0:
            print(f"  merges learned: {len(merges)}/{n_merges}", flush=True)
    return merges


class NicsTokenizer:
    """Encoder/decoder for a learned merge table."""

    def __init__(self, merges: list[tuple[int, int]], specials: Iterable[str] = DEFAULT_SPECIALS):
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials)
        if len(self.specials) > RESERVED:
            raise ValueError(f"at most {RESERVED} special tokens")
        if len(set(self.specials)) != len(self.specials):
            raise ValueError("duplicate special token")
        self.special_to_id = {s: i for i, s in enumerate(self.specials)}
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        # bytes of every non-special token
        self._bytes: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a - BYTE_OFFSET] + self._bytes[b - BYTE_OFFSET])
        self._cache: dict[str, tuple[int, ...]] = {}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")")
            if self.specials
            else None
        )

    # ---- construction -------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        specials: Iterable[str] = DEFAULT_SPECIALS,
        progress_every: int = 0,
    ) -> "NicsTokenizer":
        counts = count_pieces(texts)
        return cls(train_merges(counts, vocab_size, min_frequency, progress_every), specials)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT,
            "reserved": RESERVED,
            "specials": self.specials,
            "merges": [list(m) for m in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NicsTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT or payload.get("reserved") != RESERVED:
            raise ValueError(f"{path}: not a {FORMAT} tokenizer file")
        return cls([tuple(m) for m in payload["merges"]], payload["specials"])

    # ---- properties ---------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return MERGE_OFFSET + len(self.merges)

    def special_id(self, token: str) -> int:
        return self.special_to_id[token]

    # ---- encoding -----------------------------------------------------
    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = [BYTE_OFFSET + b for b in piece.encode("utf-8")]
        rank = self._rank
        while len(ids) > 1:
            best = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                r = rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id = MERGE_OFFSET + best_rank
            a, b = best
            merged: list[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        result = tuple(ids)
        if len(self._cache) > 1_000_000:
            self._cache.clear()
        self._cache[piece] = result
        return result

    def _encode_plain(self, text: str, out: list[int]) -> None:
        for piece in _PIECE_RE.findall(text):
            out.extend(self._encode_piece(piece))

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text. Control tokens are recognised only with allow_special=True.

        Text that came from a user must be encoded with allow_special=False.
        """
        out: list[int] = []
        if allow_special and self._special_re is not None:
            for part in self._special_re.split(text):
                if not part:
                    continue
                sid = self.special_to_id.get(part)
                if sid is not None:
                    out.append(sid)
                else:
                    self._encode_plain(part, out)
        else:
            self._encode_plain(text, out)
        return out

    def count(self, text: str, allow_special: bool = False) -> int:
        """Number of tokens in text: the unit usage is metered in."""
        return len(self.encode(text, allow_special))

    def iter_encode(self, texts: Iterable[str], allow_special: bool = False) -> Iterator[list[int]]:
        for text in texts:
            yield self.encode(text, allow_special)

    # ---- decoding -----------------------------------------------------
    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        parts: list[str] = []
        buf = bytearray()
        for i in ids:
            if i >= BYTE_OFFSET:
                buf += self._bytes[i - BYTE_OFFSET]
                continue
            if buf:
                parts.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            if not skip_special and i < len(self.specials):
                parts.append(self.specials[i])
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def token_bytes(self, token_id: int) -> bytes:
        if token_id < BYTE_OFFSET:
            name = self.specials[token_id] if token_id < len(self.specials) else ""
            return name.encode("utf-8")
        return self._bytes[token_id - BYTE_OFFSET]
