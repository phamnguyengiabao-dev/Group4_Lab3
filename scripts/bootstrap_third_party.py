from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
UPSTREAM_DIR = THIRD_PARTY_DIR / "Unsupervised-Classification"
UPSTREAM_URL = "https://github.com/wvangansbeke/Unsupervised-Classification.git"
UPSTREAM_COMMIT = "952ec31eee2c38e7233d8ad3ac0de39bb031877a"


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], check=True, cwd=cwd or PROJECT_ROOT)


def ensure_upstream_repo() -> None:
    required_paths = [
        UPSTREAM_DIR / "data" / "cifar.py",
        UPSTREAM_DIR / "data" / "stl.py",
        UPSTREAM_DIR / "losses" / "losses.py",
        UPSTREAM_DIR / "models" / "resnet.py",
    ]
    if all(path.exists() for path in required_paths):
        print(f"[third_party] ready: {UPSTREAM_DIR}")
        return

    THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)
    if UPSTREAM_DIR.exists() and not (UPSTREAM_DIR / ".git").exists():
        raise RuntimeError(
            f"Existing directory is not a git repo: {UPSTREAM_DIR}. "
            "Please remove it or rename it before bootstrapping."
        )

    if not UPSTREAM_DIR.exists():
        if shutil.which("git") is None:
            raise RuntimeError("git is required to bootstrap third_party/Unsupervised-Classification.")
        print(f"[third_party] cloning {UPSTREAM_URL} into {UPSTREAM_DIR}")
        _run_git(["clone", UPSTREAM_URL, str(UPSTREAM_DIR)])
    else:
        print(f"[third_party] fetching updates in {UPSTREAM_DIR}")
        _run_git(["fetch", "--all", "--tags"], cwd=UPSTREAM_DIR)

    print(f"[third_party] checking out commit {UPSTREAM_COMMIT}")
    _run_git(["checkout", UPSTREAM_COMMIT], cwd=UPSTREAM_DIR)
    print(f"[third_party] pinned upstream ready at {UPSTREAM_COMMIT}")


def main() -> None:
    ensure_upstream_repo()


if __name__ == "__main__":
    main()
