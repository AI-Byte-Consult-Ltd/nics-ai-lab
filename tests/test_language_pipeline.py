"""End-to-end test of the NICS Language pipeline on a tiny SYNTHETIC corpus.

corpus sources -> cleaned text -> own tokenizer -> token files -> training from random
weights -> stop on a time/step budget -> resume -> release weights -> generation.
No network. The numbers mean nothing; only the mechanics are checked.

  python tests/test_language_pipeline.py
"""
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_language_corpus as plc  # noqa: E402

WORDS = {
    "ru": "цена растёт падает золото нефть тренд вверх вниз стоп тейк уровень рынок сегодня выше ниже".split(),
    "en": "price rises falls gold oil trend up down stop take level market today above below".split(),
    "es": "precio sube baja oro petróleo tendencia arriba abajo stop objetivo nivel mercado hoy sobre bajo".split(),
    "bg": "цената расте пада злато петрол тенденция нагоре надолу стоп цел ниво пазар днес над под".split(),
    "pt": "preço sobe desce ouro petróleo tendência cima baixo stop alvo nível mercado hoje acima abaixo".split(),
}


def run(*args: str) -> str:
    res = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"{args}\n{res.stdout}\n{res.stderr}"
    return res.stdout


def main() -> None:
    # --- licence gate and cleaning (pure functions, real registry) -----------
    sources = json.loads((ROOT / "data_registry" / "language_sources.json").read_text(encoding="utf-8"))
    for s in sources:
        assert s["license"] and "commercial_use" in s and "license_checked" in s, s["source_id"]
    assert plc.refusal({"license": "x", "commercial_use": True, "license_checked": None})
    assert plc.refusal({"license": "x", "commercial_use": False, "license_checked": "2026-01-01"})
    assert plc.refusal({"license": "", "commercial_use": True, "license_checked": "2026-01-01"})
    assert plc.refusal({"license": "x", "commercial_use": True, "license_checked": "2026-01-01"}) is None

    book = (
        "Title: Example\nLanguage: Spanish\n\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
        "Texto del libro.\r\n\r\n\r\n\r\n\r\nFin.\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nlicense text"
    )
    body, lang = plc.strip_gutenberg(book)
    assert lang == "spanish" and "PROJECT GUTENBERG" not in body and "license text" not in body
    assert plc.clean_text(body) == "Texto del libro.\n\n\nFin.\n"
    assert plc.clean_text("a\u0000b﻿  \nc") == "ab\nc\n"

    tmp = Path(tempfile.mkdtemp())
    try:
        # --- synthetic corpus in five languages --------------------------------
        rnd = random.Random(3)
        text_dir = tmp / "text"
        for lang, words in WORDS.items():
            for d in range(8):
                doc = text_dir / lang / "synthetic" / f"doc{d}.txt"
                doc.parent.mkdir(parents=True, exist_ok=True)
                lines = [" ".join(rnd.choice(words) for _ in range(12)) + f" {rnd.randint(1000, 9999)}." for _ in range(60)]
                doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tok_path = tmp / "tokenizer.json"
        out = run("scripts/train_tokenizer.py", "--text-dir", str(text_dir), "--vocab-size", "700", "--out", str(tok_path))
        assert "trained from scratch" in out
        tok_manifest = json.loads(tok_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert tok_manifest["pretrained_tokenizer_used"] is None and len(tok_manifest["languages"]) == 5

        tokens_dir = tmp / "tokens"
        run("scripts/encode_corpus.py", "--tokenizer", str(tok_path), "--text-dir", str(text_dir),
            "--out", str(tokens_dir), "--val-percent", "20")
        manifest = json.loads((tokens_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["documents"]["train"] > 0 and manifest["documents"]["val"] > 0
        assert manifest["documents"]["train"] + manifest["documents"]["val"] == 40
        assert set(manifest["tokens_per_language"]) == set(WORDS)
        assert (tokens_dir / "train.bin").stat().st_size == manifest["tokens"]["train"] * 2

        # --- training from random weights, stopped by a step budget, then resumed --
        vocab = manifest["vocab_size"]
        cfg = tmp / "tiny.yaml"
        cfg.write_text(
            f"""model_id: nics-text-test
seed: 1
model: {{vocab_size: {vocab}, d_model: 48, n_layers: 2, n_heads: 4, max_seq_len: 64, dropout: 0.0, pos_encoding: rope}}
data: {{tokenizer: "{tok_path.as_posix()}", tokens_dir: "{tokens_dir.as_posix()}"}}
training: {{batch_size: 8, block_size: 32, max_steps: 120, eval_interval: 20, eval_iters: 4,
  checkpoint_interval: 40, learning_rate: 0.003, warmup_steps: 10, weight_decay: 0.01, grad_clip: 1.0}}
paths: {{experiments_dir: "{(tmp / 'exp').as_posix()}"}}
""",
            encoding="utf-8",
        )
        run("scripts/train_text.py", "--config", str(cfg), "--experiment-id", "t1", "--max-steps", "60", "--threads", "1")
        exp = tmp / "exp" / "t1"
        assert (exp / "checkpoint-step-000000" / "trainer-state.pt").exists(), "no proof of random init"
        events = [json.loads(line) for line in (exp / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        metrics = [e for e in events if e["event_type"] == "training_metric"]
        first = metrics[0]["validation_loss"]
        assert abs(first - math.log(vocab)) < 0.6, f"step-0 loss {first} is not random-init entropy {math.log(vocab):.2f}"
        run_manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
        assert run_manifest["pretrained_weights_used"] is None
        assert run_manifest["corpus"]["tokenizer_sha256"] == manifest["tokenizer_sha256"]

        out = run("scripts/train_text.py", "--config", str(cfg), "--resume", str(exp), "--threads", "1")
        assert "Resumed from" in out and "at step 60" in out, out
        events = [json.loads(line) for line in (exp / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        metrics = [e for e in events if e["event_type"] == "training_metric"]
        assert metrics[-1]["step"] == 119
        assert metrics[-1]["validation_loss"] < first - 1.0, "the model did not learn"
        assert any(e["event_type"] == "training_completed" for e in events)
        assert (exp / "release" / "model.safetensors").exists()

        # --- the trained weights generate text through the cache ---------------
        import torch
        from safetensors.torch import load_file

        from nics.models.text.model import NicsTextConfig, NicsTextModel
        from nics.text.tokenizer import NicsTokenizer

        tok = NicsTokenizer.load(tok_path)
        model = NicsTextModel(NicsTextConfig(vocab_size=vocab, d_model=48, n_layers=2, n_heads=4, max_seq_len=64, pos_encoding="rope"))
        model.load_state_dict(load_file(str(exp / "release" / "model.safetensors")))
        prompt = torch.tensor([[tok.special_id("<bos>")] + tok.encode("price")])
        ids = model.generate(prompt, 20, temperature=0)[0].tolist()
        assert len(ids) == prompt.shape[1] + 20 and isinstance(tok.decode(ids), str)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_language_pipeline: OK")


if __name__ == "__main__":
    main()
