"""Tests of the NICS Language model on random weights (no data, no network).

  python tests/test_text_model.py
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.common.config import load_config  # noqa: E402
from nics.models.text.model import NicsTextConfig, NicsTextModel  # noqa: E402


def build(pos: str, vocab: int = 300) -> NicsTextModel:
    torch.manual_seed(0)
    return NicsTextModel(
        NicsTextConfig(vocab_size=vocab, d_model=64, n_layers=3, n_heads=4, max_seq_len=48, pos_encoding=pos)
    ).eval()


def main() -> None:
    for pos in ("learned", "rope"):
        model = build(pos)
        idx = torch.randint(0, 300, (2, 20))

        # Random-init loss is close to ln(vocab): evidence of no pretrained weights.
        _, loss = model(idx[:, :-1], idx[:, 1:])
        assert abs(loss.item() - math.log(300)) < 0.5, (pos, loss.item())

        # Causality: changing a later token never changes earlier logits.
        full, _ = model(idx)
        changed = idx.clone()
        changed[:, 12:] = (changed[:, 12:] + 1) % 300
        other, _ = model(changed)
        assert torch.allclose(full[:, :12], other[:, :12], atol=1e-5), pos
        assert not torch.allclose(full[:, 12:], other[:, 12:], atol=1e-5), pos

        # The key/value cache gives the same logits as a full pass:
        # prompt in one piece, then a multi-token chunk, then single tokens.
        cache = model.new_cache()
        parts = [model(idx[:, :7], cache=cache, start_pos=0)[0]]
        parts.append(model(idx[:, 7:11], cache=cache, start_pos=7)[0])
        for t in range(11, 20):
            parts.append(model(idx[:, t : t + 1], cache=cache, start_pos=t)[0])
        assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-4), pos

        # Greedy generation is identical with and without the cache.
        prompt = idx[:1, :5]
        a = model.generate(prompt, 20, temperature=0, use_cache=True)
        b = model.generate(prompt, 20, temperature=0, use_cache=False)
        assert torch.equal(a, b), pos
        assert a.shape[1] == 25

        # The cache never runs past the context; eos stops generation.
        assert model.generate(idx[:1, :40], 100, temperature=0).shape[1] == 48
        eos = a[0, 5].item()
        assert model.generate(prompt, 20, temperature=0, eos_id=eos).shape[1] == 6

    # Rotary positions add no parameters and nothing to the checkpoint.
    learned, rope = build("learned"), build("rope")
    assert learned.num_parameters() - rope.num_parameters() == 48 * 64
    assert not any("rope" in k for k in rope.state_dict())
    assert "pos_emb.weight" in learned.state_dict()  # nics-text-nano checkpoints still load

    # Shipped configs build and have the documented size.
    for name, lo, hi in (("nano", 0.8e6, 1.0e6), ("micro", 22e6, 25e6), ("alpha", 45e6, 56e6)):
        cfg = load_config(ROOT / "configs" / "text" / f"{name}.yaml")
        m = cfg.model
        vocab = m.get("vocab_size", 512)
        model = NicsTextModel(
            NicsTextConfig(
                vocab_size=vocab, d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads,
                max_seq_len=m.max_seq_len, dropout=m.dropout, pos_encoding=m.get("pos_encoding", "learned"),
            )
        )
        n = model.num_parameters()
        assert lo <= n <= hi, (name, n)
        print(f"  {name}: {n:,} parameters")
    print("test_text_model: OK")


if __name__ == "__main__":
    main()
