"""Check what the NICS Language model writes for the journal task, by code.

Loads the newest checkpoint of an experiment, asks for posts over all languages and fact
combinations and runs every answer through nics.text.examples.check(): exact value
slots, no digits written by the model, allowed markup, length, and only words of the
requested language that our own phrases use. The share of answers that
pass is the number that decides whether the model may ever replace the current text
source; a post that fails is never published (the template is used instead).

  python scripts/eval_journal.py --experiment experiments/text/nics-text-micro/<id> --samples 40
"""
import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.models.text.model import NicsTextConfig, NicsTextModel  # noqa: E402
from nics.text import examples  # noqa: E402
from nics.text.tokenizer import NicsTokenizer  # noqa: E402
from nics.training.checkpoint import load_trainer_state  # noqa: E402


def evaluate(model, tok, samples: int, seed: int = 11, temperature: float = 0.7, top_k: int = 40,
             max_new_tokens: int = 260) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    combos = list(itertools.product(sorted(examples.BANK), ("buy", "sell"), (1, 3), (True, False), (True, False)))
    rng.shuffle(combos)
    end_id = tok.special_id("<|end|>")
    passed, by_lang, failures, example, t0, new_tokens = 0, {}, [], None, time.time(), 0
    for lang, direction, targets, conf, note in itertools.islice(itertools.cycle(combos), samples):
        ids = [tok.special_id("<bos>")] + tok.encode(examples.prompt(lang, direction, targets, conf, note), allow_special=True)
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new_tokens, temperature=temperature, top_k=top_k, eos_id=end_id)
        answer = out[0, len(ids):].tolist()
        new_tokens += len(answer)
        ended = bool(answer) and answer[-1] == end_id
        text = tok.decode(answer[:-1] if ended else answer)
        problems = examples.check(text, targets, conf, note, lang) + ([] if ended else ["no end marker"])
        stat = by_lang.setdefault(lang, {"passed": 0, "total": 0})
        stat["total"] += 1
        if not problems:
            passed += 1
            stat["passed"] += 1
            example = example or {"lang": lang, "text": text}
        elif len(failures) < 5:
            failures.append({"lang": lang, "problems": problems[:3], "text": text[:300]})
    seconds = time.time() - t0
    return {"samples": samples, "passed": passed, "pass_percent": round(100.0 * passed / max(samples, 1), 1),
            "by_language": by_lang, "tokens_per_second": round(new_tokens / max(seconds, 1e-6), 1),
            "example": example, "failures": failures}


def load_model(experiment: Path, config_path: str):
    cfg = load_config(config_path)
    tok = NicsTokenizer.load(str(ROOT / cfg["data"]["tokenizer"]))
    model = NicsTextModel(NicsTextConfig(
        vocab_size=tok.vocab_size, d_model=cfg.model.d_model, n_layers=cfg.model.n_layers, n_heads=cfg.model.n_heads,
        max_seq_len=cfg.model.max_seq_len, dropout=0.0, pos_encoding=cfg.model.get("pos_encoding", "learned")))
    saved = sorted((experiment / "checkpoints").glob("step-*"))
    latest = saved[-1] if saved else experiment / "checkpoint-step-000000"
    state = load_trainer_state(latest / "trainer-state.pt", model)
    return model, tok, int(state["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "text" / "micro.yaml"))
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--out", default=None, help="write the result as JSON to this file")
    args = parser.parse_args()
    model, tok, step = load_model(Path(args.experiment), args.config)
    result = {"step": step, **evaluate(model, tok, args.samples)}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2))  # ASCII only: a Windows log file may not take Cyrillic


if __name__ == "__main__":
    main()
