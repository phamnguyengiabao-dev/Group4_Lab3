from __future__ import annotations

import re
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_PRETEXT_DIR = PROJECT_ROOT / "data" / "scan_pretext"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"


CHECKPOINT_SPECS = [
    {
        "name": "cifar10_pretext.pth.tar",
        "path": SCAN_PRETEXT_DIR / "cifar10_pretext.pth.tar",
        "type": "gdrive",
        "file_id": "1Cl5oAcJKoNE5FSTZsBSAKLcyA5jXGgTT",
    },
    {
        "name": "cifar20_pretext.pth.tar",
        "path": SCAN_PRETEXT_DIR / "cifar20_pretext.pth.tar",
        "type": "gdrive",
        "file_id": "1huW-ChBVvKcx7t8HyDaWTQB5Li1Fht9x",
    },
    {
        "name": "stl10_pretext.pth.tar",
        "path": SCAN_PRETEXT_DIR / "stl10_pretext.pth.tar",
        "type": "gdrive",
        "file_id": "1261NDFfXuKR2Dh4RWHYYhcicdcPag9NZ",
    },
    {
        "name": "moco_v2_800ep_pretrain.pth.tar",
        "path": CHECKPOINT_DIR / "moco_v2_800ep_pretrain.pth.tar",
        "type": "http",
        "url": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep_pretrain.pth.tar",
    },
]


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _download_gdrive(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cj = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    base_url = "https://drive.google.com/uc?export=download"
    first_url = f"{base_url}&id={urllib.parse.quote(file_id)}"
    with opener.open(first_url) as response:
        content_type = response.headers.get("Content-Type", "")
        if "application/octet-stream" in content_type:
            with destination.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        html = response.read().decode("utf-8", errors="ignore")

    token_match = re.search(r'confirm=([0-9A-Za-z_]+)', html)
    if not token_match:
        token_match = re.search(r'name="confirm" value="([0-9A-Za-z_]+)"', html)
    if not token_match:
        raise RuntimeError(f"Could not resolve Google Drive confirmation token for file id {file_id}.")

    confirm = token_match.group(1)
    second_url = f"{base_url}&confirm={urllib.parse.quote(confirm)}&id={urllib.parse.quote(file_id)}"
    with opener.open(second_url) as response, destination.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def ensure_checkpoints() -> None:
    for spec in CHECKPOINT_SPECS:
        path = spec["path"]
        if path.exists() and path.stat().st_size > 0:
            print(f"[checkpoint] ready: {path}")
            continue

        print(f"[checkpoint] downloading {spec['name']} -> {path}")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            if spec["type"] == "gdrive":
                _download_gdrive(spec["file_id"], tmp_path)
            else:
                _download_url(spec["url"], tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"[checkpoint] saved: {path}")


def main() -> None:
    ensure_checkpoints()


if __name__ == "__main__":
    main()
