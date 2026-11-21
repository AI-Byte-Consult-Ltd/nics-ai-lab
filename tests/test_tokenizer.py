"""Tests of the from-scratch NICS tokenizer (no network, no third-party tokenizer).

  python tests/test_tokenizer.py
"""
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.text.tokenizer import (  # noqa: E402
    BYTE_OFFSET,
    MERGE_OFFSET,
    NicsTokenizer,
    split_pieces,
)

CORPUS = [
    "Золото растёт: цена 4321.50, стоп 4298.10, тейк 4350.00. Тренд вверх на 1h и 4h.",
    "Gold is rising: price 4321.50, stop 4298.10, take 4350.00. Uptrend on 1h and 4h.",
    "El oro sube: precio 4321,50, stop 4298,10. ¿Tendencia alcista? Sí, en 1h y 4h.",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Съешь же ещё этих мягких французских булок, да выпей чаю. " * 20,
    "El veloz murciélago hindú comía feliz cardillo y kiwi. " * 20,
]

SAMPLES = CORPUS + [
    "",
    " ",
    "\n\n\t  ",
    "a",
    "__init__ snake_case_name",
    "Цена: 1 234 567,89 € (≈ +0,5 %) — «да»!",
    "emoji 🚀📈 and CJK 日本語 and math ²³ ½",
    "tabs\tand\r\nwindows newlines",
    "<|task:card|><|lang:ru|> not special unless allowed",
    "0123456789" * 5,
    "Златото расте: цена 4321,50. Тенденцията е възходяща.",
    "O ouro está a subir: preço 4321,50. Tendência de alta.",
    "Gold steigt. L'or monte. L'oro sale. Złoto rośnie. Altın yükseliyor.",
    "الذهب يرتفع: السعر 4321.50",
    "黄金上涨：价格 4321.50，止损 4298.10。",
]


def check_pieces_lossless() -> None:
    for s in SAMPLES:
        assert "".join(split_pieces(s)) == s, repr(s)
    rnd = random.Random(7)
    alphabet = "abc АБВ ñé 019 _.,!?\n\t🚀"
    for _ in range(300):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 40)))
        assert "".join(split_pieces(s)) == s, repr(s)


def main() -> None:
    check_pieces_lossless()

    tok = NicsTokenizer.train(CORPUS, vocab_size=MERGE_OFFSET + 200)
    assert MERGE_OFFSET < tok.vocab_size <= MERGE_OFFSET + 200

    # Lossless round trip on text never seen in training, in every script.
    for s in SAMPLES:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, repr(s)
        assert all(BYTE_OFFSET <= i < tok.vocab_size for i in ids), "plain text produced a special id"

    # Merges actually compress repeated text.
    text = CORPUS[3]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2

    # Digits never merge: every digit is exactly one token.
    number = "4321.50"
    ids = tok.encode(number)
    assert [tok.decode([i]) for i in ids] == list(number), ids
    for merged in range(MERGE_OFFSET, tok.vocab_size):
        assert not any(48 <= b <= 57 for b in tok.token_bytes(merged)), "a merge contains a digit"

    # Special tokens: only on request. User text can never inject one.
    marked = "<|task:card|><|lang:ru|>привет<|end|>"
    with_special = tok.encode(marked, allow_special=True)
    assert with_special[0] == tok.special_id("<|task:card|>")
    assert with_special[1] == tok.special_id("<|lang:ru|>")
    assert with_special[-1] == tok.special_id("<|end|>")
    assert tok.decode(with_special) == marked
    assert tok.decode(with_special, skip_special=True) == "привет"
    as_user_text = tok.encode(marked, allow_special=False)
    assert all(i >= BYTE_OFFSET for i in as_user_text), "user text became a control token"
    assert tok.decode(as_user_text) == marked

    # Every site language has its control token; the Mini App languages come first.
    from nics.text.tokenizer import APP_LANGUAGES, LANGUAGES

    assert len(LANGUAGES) == 12 and APP_LANGUAGES == ("ru", "en", "es", "bg", "pt")
    for lang in LANGUAGES:
        assert tok.encode(f"<|lang:{lang}|>", allow_special=True) == [tok.special_id(f"<|lang:{lang}|>")]

    # Deterministic training: same corpus -> same merges, in any input order.
    again = NicsTokenizer.train(list(reversed(CORPUS)), vocab_size=MERGE_OFFSET + 200)
    assert again.merges == tok.merges

    # Save / load keeps ids identical.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tokenizer.json"
        tok.save(path)
        loaded = NicsTokenizer.load(path)
        for s in SAMPLES:
            assert loaded.encode(s, allow_special=True) == tok.encode(s, allow_special=True)

    # Token counting (the billing meter) agrees with encode.
    assert tok.count(CORPUS[0]) == len(tok.encode(CORPUS[0]))

    # The merge trainer matches a slow reference implementation.
    check_against_reference()
    print("test_tokenizer: OK")


def check_against_reference() -> None:
    """Naive O(merges * corpus) BPE with the same tie-breaking, compared merge by merge."""
    from collections import Counter

    from nics.text.tokenizer import count_pieces, train_merges

    counts = count_pieces(CORPUS)
    fast = train_merges(counts, MERGE_OFFSET + 120, min_frequency=2)

    words = {tuple(BYTE_OFFSET + b for b in p.encode("utf-8")): f for p, f in counts.items()}
    slow = []
    while len(slow) < 120:
        pairs: Counter = Counter()
        for w, f in words.items():
            for pair in zip(w, w[1:]):
                pairs[pair] += f
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
        if best[1] < 2:
            break
        (a, b), new_id = best[0], MERGE_OFFSET + len(slow)
        slow.append(best[0])
        merged_words: dict = {}
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            merged_words[tuple(out)] = merged_words.get(tuple(out), 0) + f
        words = merged_words
    assert fast == slow, "fast merge trainer disagrees with the reference"


if __name__ == "__main__":
    main()
