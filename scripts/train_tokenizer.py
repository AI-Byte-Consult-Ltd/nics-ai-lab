"""Train a byte-level BPE tokenizer from scratch on the NICS Text corpus.

This does NOT load any pretrained tokenizer (e.g. GPT-2's). The vocabulary
and merge rules are learned entirely from the local dataset.
"""
import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(ROOT / "data" / "raw" / "tinyshakespeare.txt"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "tokenizer"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[args.corpus],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.save_model(str(out_dir))
    tokenizer.save(str(out_dir / "tokenizer.json"))

    encoded = tokenizer.encode("First Citizen:\nBefore we proceed")
    print(f"Tokenizer trained. vocab_size={tokenizer.get_vocab_size()}")
    print(f"Saved to {out_dir}")
    print(f"Sanity check tokens: {encoded.tokens[:10]}")


if __name__ == "__main__":
    main()
