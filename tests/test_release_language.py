import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from release_language import newest_experiment  # noqa: E402


def touch(root: Path, exp: str, step: int) -> None:
    d = root / exp / "checkpoints" / f"step-{step:06d}"
    d.mkdir(parents=True)
    (d / "trainer-state.pt").write_bytes(b"x")


def test_newest_experiment_is_the_most_trained_one():
    root = Path(tempfile.mkdtemp())
    assert newest_experiment(root) is None
    touch(root, "exp-a", 500)
    touch(root, "exp-a", 3000)
    touch(root, "exp-b", 1000)
    (root / "exp-c").mkdir()
    assert newest_experiment(root).name == "exp-a"
    touch(root, "exp-b", 3000)
    assert newest_experiment(root).name == "exp-b"


if __name__ == "__main__":
    test_newest_experiment_is_the_most_trained_one()
    print("ok test_newest_experiment_is_the_most_trained_one")
