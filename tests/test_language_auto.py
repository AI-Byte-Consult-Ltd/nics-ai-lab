"""NICS Language corpus sampling and the unattended night run: pure logic, no network."""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import language_auto  # noqa: E402
import language_night  # noqa: E402
import prepare_language_corpus as plc  # noqa: E402
import self_learn  # noqa: E402


def test_sample_lines_blocks_repeats_and_short_lines():
    lines = [f"sentence number {i} is long enough to be kept" for i in range(40)]
    lines[5] = lines[4]          # exact repeat
    lines[6] = "too short"
    kept = list(plc.sample_lines(lines, sample_every=2, block_lines=10))
    # blocks 0 and 2 are kept (lines 0-9 and 20-29), minus one repeat and one short line
    assert len(kept) == 18 and kept[0].endswith("0 is long enough to be kept")
    assert all(not 10 <= int(k.split()[2]) < 20 for k in kept)
    assert len(list(plc.sample_lines(["短句也是完整的句子"], min_chars=8))) == 1


def test_docs_stop_at_the_size_limit():
    lines = ["x" * 99 for _ in range(1000)]          # 100 bytes per line with the newline
    docs = list(plc.docs_from_lines(iter(lines), "en", max_text_bytes=2500, doc_bytes=1000))
    assert [d[1] for d in docs] == ["part-0001", "part-0002", "part-0003"]
    assert sum(len(d[2]) + 1 for d in docs) == 2500 and all(d[0] == "en" for d in docs)
    short = list(plc.docs_from_lines(iter(lines[:3]), "en", max_text_bytes=10_000, doc_bytes=1000))
    assert len(short) == 1 and short[0][2].count("\n") == 2


def test_every_source_states_a_licence_and_known_kind():
    sources = json.loads((ROOT / "data_registry" / "language_sources.json").read_text(encoding="utf-8"))
    first_wave = {s["lang"] for s in sources if s["kind"] == "opus_mono" and not plc.refusal(s)
                  and s["max_text_bytes"] >= 50_000_000}
    assert first_wave == set(language_auto.FIRST_WAVE)
    for s in sources:
        assert s["kind"] in ("local", "gutenberg", "opus_mono") and s["license"]
        if s["kind"] == "opus_mono":
            assert s["url"].startswith("https://") and s["attribution"] and s["license_checked"]


def test_minutes_until_next_clock_time():
    assert language_auto.minutes_until("08:00", datetime(2026, 11, 22, 4, 30)) == 210
    assert language_auto.minutes_until("08:00", datetime(2026, 11, 22, 8, 0)) == 24 * 60


def test_language_night_starts_only_at_night(monkeypatch=None):
    assert self_learn.start_language_night(datetime(2026, 11, 22, 14, 0)) is False   # a run by hand in the day
    assert self_learn.start_language_night(datetime(2026, 11, 22, 7, 30)) is False   # too late to be useful


def test_newest_unfinished_experiment():
    tmp = Path(tempfile.mkdtemp())
    for name, released in (("exp-1", True), ("exp-2", False), ("exp-3", False), ("junk", False)):
        d = tmp / name
        d.mkdir()
        if name != "junk":
            (d / "checkpoint-step-000000").mkdir()
            (d / "checkpoint-step-000000" / "trainer-state.pt").write_bytes(b"x")
        if released:
            (d / "release").mkdir()
    assert language_night.newest_unfinished(tmp).name == "exp-3"
    assert language_night.newest_unfinished(tmp / "missing") is None




def test_journal_examples_are_valid_and_filled_safely():
    import random
    from nics.text import examples as ex
    from nics.text.tokenizer import APP_LANGUAGES, DEFAULT_SPECIALS
    assert set(ex.BANK) == set(APP_LANGUAGES)
    assert all(token in DEFAULT_SPECIALS for token in ex.SLOTS.values())
    rng = random.Random(7)
    for lang in ex.BANK:
        assert set(ex.BANK[lang]) == set(ex.BANK["ru"])
        for direction in ("buy", "sell"):
            for targets in (1, 4):
                for conf in (True, False):
                    for note in (True, False):
                        for _ in range(30):
                            text = ex.compose(lang, direction, targets, conf, note, rng)
                            assert ex.check(text, targets, conf, note, lang, direction) == [], (lang, text)
                            other = "sell" if direction == "buy" else "buy"
                            assert ex.check(text, targets, conf, note, lang, other), (lang, text)
                            assert "{" not in text and "|}" not in text
    text = ex.compose("ru", "buy", 2, True, True, rng)
    # the checker refuses what the model must never do
    assert ex.check(text.replace("<|stop|>", "2350"), 2, True, True)          # wrote a number itself
    assert ex.check(text + " <|tp1|>", 2, True, True)                           # unexpected slot
    assert ex.check(text + " <a href='x'>x</a>", 2, True, True)                 # other markup
    assert ex.check(text, 2, True, False)                                       # note slot without a note
    assert ex.check(text, 2, True, True, "ru") == [] and ex.check(text, 2, True, True, "en")   # wrong language
    spanish = ex.compose("es", "buy", 2, True, True, rng)
    assert ex.check(spanish, 2, True, True, "pt") and ex.check(spanish, 2, True, True, "es") == []
    assert ex.check(text.replace("<|stop|>", "<|stop|> гарантия"), 2, True, True, "ru")           # invented word
    filled = ex.fill(text, {"symbol": "XAUUSD", "entry": "1-2", "stop": "0.5", "targets": "3, 4", "conf": "70",
                            "note": "<b>x</b> & <|stop|>"})
    assert "&lt;b&gt;x&lt;/b&gt; &amp; &lt;|stop|&gt;" in filled and "<|" not in filled
    assert ex.prompt("bg", "sell", 1, False, True) == "<|in|><|task:journal|><|lang:bg|> dir=sell targets=one conf=no note=yes<|out|>"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
