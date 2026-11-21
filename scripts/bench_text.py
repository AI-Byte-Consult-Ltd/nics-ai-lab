"""Measure how fast THIS machine trains and runs NICS Language models.

Stage 0 of the NICS Language plan: every time estimate in the plan was calculated,
not measured. This script measures it. It uses random tokens and random weights,
needs no data and no network, and writes nothing but its own report.

Usage:
  python scripts/bench_text.py                 # nano-size check, micro, alpha (about 3-4 minutes)
  python scripts/bench_text.py --seconds 60    # longer, steadier measurement
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.models.text.model import NicsTextConfig, NicsTextModel  # noqa: E402


def build(cfg) -> NicsTextModel:
    m = cfg.model
    return NicsTextModel(
        NicsTextConfig(
            vocab_size=m.vocab_size, d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
            max_seq_len=m.max_seq_len, dropout=0.0, pos_encoding=m.get("pos_encoding", "learned"),
        )
    )


def bench_training(model, cfg, seconds: float) -> float:
    """Training tokens per second: forward + backward + optimiser step."""
    batch, block = cfg.training.batch_size, cfg.training.get("block_size", cfg.model.max_seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    def step() -> None:
        x = torch.randint(0, cfg.model.vocab_size, (batch, block))
        _, loss = model(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    step()  # warm-up, not timed
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        step()
        n += 1
    return n * batch * block / (time.perf_counter() - t0)


@torch.no_grad()
def bench_generation(model, cfg, new_tokens: int = 200, prompt_len: int = 300) -> tuple[float, float]:
    """(seconds to read the prompt, generated tokens per second) for one request with the cache."""
    prompt_len = min(prompt_len, cfg.model.max_seq_len - new_tokens - 1)
    prompt = torch.randint(0, cfg.model.vocab_size, (1, prompt_len))
    model.eval()
    cache = model.new_cache()
    t0 = time.perf_counter()
    logits, _ = model(prompt, cache=cache, start_pos=0)
    t_prompt = time.perf_counter() - t0
    nxt = logits[:, -1:].argmax(-1)
    t0 = time.perf_counter()
    for i in range(new_tokens):
        logits, _ = model(nxt, cache=cache, start_pos=prompt_len + i)
        nxt = logits[:, -1:].argmax(-1)
    return t_prompt, new_tokens / (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0, help="training measurement time per model")
    parser.add_argument("--models", nargs="+", default=["micro", "alpha"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "logs" / "bench_text.json"))
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "machine": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "models": [],
    }
    print(f"Machine: {report['machine']} | logical CPUs {report['cpu_count']} | torch {report['torch']} "
          f"| threads {report['torch_threads']}")
    for name in args.models:
        cfg = load_config(ROOT / "configs" / "text" / f"{name}.yaml")
        torch.manual_seed(0)
        model = build(cfg)
        params = model.num_parameters()
        print(f"{cfg.model_id}: {params:,} parameters, measuring ...", flush=True)
        train_tps = bench_training(model, cfg, args.seconds)
        t_prompt, gen_tps = bench_generation(model, cfg)
        row = {
            "model_id": cfg.model_id,
            "parameters": params,
            "train_tokens_per_second": round(train_tps, 1),
            "train_million_tokens_per_5h_night": round(train_tps * 5 * 3600 / 1e6, 2),
            "prompt_300_tokens_seconds": round(t_prompt, 3),
            "generate_tokens_per_second": round(gen_tps, 1),
            "answer_200_tokens_seconds": round(t_prompt + 200 / gen_tps, 2),
        }
        report["models"].append(row)
        print(f"  training:   {row['train_tokens_per_second']:,.0f} tokens/s "
              f"= {row['train_million_tokens_per_5h_night']} M tokens per 5-hour night")
        print(f"  generation: {row['generate_tokens_per_second']:,.0f} tokens/s, "
              f"a 200-token answer to a 300-token request takes {row['answer_200_tokens_seconds']} s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out}. Send this file (or the text above) to Claude.")


if __name__ == "__main__":
    main()
