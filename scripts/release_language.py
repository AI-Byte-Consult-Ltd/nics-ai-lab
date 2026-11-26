"""Make a versioned NICS Language release folder from the newest training checkpoint.

The lab is the single source of truth for the weights: training happens only here, other
products (AquaTerra and so on) receive a release and use it as it is, and send their
dialogues back as data. A release is one folder:

  releases/language/<model_id>-step-XXXXXX/
    model.safetensors        weights only (no optimizer state)
    tokenizer.json           the vocabulary these weights need
    config.yaml              model sizes
    training-manifest.json   what this is: step, validation loss, lab version, text sources
    README.txt               how to load it
    sha256sums.txt           checksums of every file above

Training is not touched: the experiment keeps running from its checkpoints.

  python scripts/release_language.py [--experiment experiments/text/nics-text-micro/<id>]
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nics.common.config import load_config  # noqa: E402
from nics.training.checkpoint import export_release  # noqa: E402

README = """NICS Language release {name}

Weights of the NICS Language model, trained from scratch in the NICS AI Lab.
The lab is the only place where this model is trained. Use this folder as it is;
do not continue training from it elsewhere: send dialogues back to the lab instead.

Load (needs the lab code, repository nics-ai-lab, same version or newer):

    from safetensors.torch import load_file
    from nics.models.text.model import NicsTextConfig, NicsTextModel
    from nics.text.tokenizer import NicsTokenizer

    tok = NicsTokenizer.load("tokenizer.json")
    model = NicsTextModel(NicsTextConfig(vocab_size=tok.vocab_size, d_model={d_model}, n_layers={n_layers},
                                         n_heads={n_heads}, max_seq_len={max_seq_len}, dropout=0.0,
                                         pos_encoding="{pos}"))
    model.load_state_dict(load_file("model.safetensors"))
    model.eval()

State at this release: step {step} of {max_steps}. See training-manifest.json.
Check the files with sha256sums.txt before use.
"""


def newest_experiment(experiments_dir: Path) -> Path | None:
    """The experiment whose latest checkpoint has the highest step (newest name wins a tie)."""
    best = None
    for d in sorted(experiments_dir.iterdir()) if experiments_dir.is_dir() else []:
        saved = sorted((d / "checkpoints").glob("step-*/trainer-state.pt"))
        if saved:
            key = (int(saved[-1].parent.name.split("-")[1]), d.name)
            if best is None or key > best[0]:
                best = (key, d)
    return best[1] if best else None


def text_sources() -> list[dict]:
    registry = ROOT / "data" / "processed" / "language" / "datasets.json"
    if not registry.exists():
        return []
    return [{"source": d.get("source"), "license": d.get("license"), "size_bytes": d.get("size_bytes")}
            for d in json.loads(registry.read_text(encoding="utf-8"))]


def build(experiment: Path, config_path: str, out_root: Path) -> Path:
    from eval_journal import load_model
    cfg = load_config(config_path)
    model, _tok, step = load_model(experiment, config_path)
    name = f"{cfg['model_id']}-step-{step:06d}"
    out = out_root / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(ROOT / cfg["data"]["tokenizer"], out / "tokenizer.json")
    shutil.copyfile(config_path, out / "config.yaml")
    m = cfg["model"]
    (out / "README.txt").write_text(README.format(
        name=name, d_model=m["d_model"], n_layers=m["n_layers"], n_heads=m["n_heads"], max_seq_len=m["max_seq_len"],
        pos=m.get("pos_encoding", "learned"), step=step, max_steps=cfg["training"]["max_steps"]), encoding="utf-8")
    last = {}
    last_file = ROOT / "logs" / "language_last.json"
    if last_file.exists():
        try:
            last = json.loads(last_file.read_text(encoding="utf-8"))
        except ValueError:
            last = {}
    same_step = last.get("step") == step
    manifest = {
        "release": name,
        "model_id": cfg["model_id"],
        "experiment_id": experiment.name,
        "step": step,
        "max_steps": cfg["training"]["max_steps"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation_loss": last.get("metric") if same_step else None,
        "journal_check": last.get("journal_check") if same_step else None,
        "lab_version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "released_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pretrained_weights_used": None,
        "source_of_truth": "nics-ai-lab; do not train this release further elsewhere",
        "text_sources": text_sources(),
    }
    export_release(out, model, manifest)  # writes the weights, the manifest and checksums of every file
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "text" / "micro.yaml"))
    parser.add_argument("--out", default=str(ROOT / "releases" / "language"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    experiment = Path(args.experiment) if args.experiment else newest_experiment(ROOT / cfg["paths"]["experiments_dir"])
    if experiment is None or not experiment.is_dir():
        sys.exit("No NICS Language experiment with checkpoints found. Run a training night first.")
    out = build(experiment, args.config, Path(args.out))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"Release folder: {out}")
    print(f"Files: {', '.join(sorted(f.name for f in out.iterdir()))}")
    print(f"Size: {size:.0f} MB. Give this whole folder to the other project.")


if __name__ == "__main__":
    main()
