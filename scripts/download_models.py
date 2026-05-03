"""Fetch detector weights for VisionFlow.

Downloads the IISc UVH-26 YOLOv11-S model (14 South Asian traffic classes,
including Three-wheeler / auto rickshaw) into ``models/uvh26.pt``.

Run once:

    python scripts/download_models.py

The pipeline will pick the file up via the path in your config's
``detector.weights`` field. The COCO ``yolov8n.pt`` model used by the
optional secondary detector is downloaded automatically by ultralytics
the first time it runs, so no extra step is needed for it.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

UVH26_URL = (
    "https://huggingface.co/iisc-aim/UVH-26/resolve/main/"
    "weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TARGET = MODELS_DIR / "uvh26.pt"


def _report(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded * 100 / total_size
    mb_done = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  {pct:5.1f}%  ({mb_done:6.1f} / {mb_total:6.1f} MiB)")
    sys.stdout.flush()


def download(force: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and not force:
        print(f"[skip] {TARGET} already exists ({TARGET.stat().st_size / 1e6:.1f} MB)")
        return TARGET

    print(f"[fetch] {UVH26_URL}")
    print(f"   -> {TARGET}")
    urllib.request.urlretrieve(UVH26_URL, TARGET, reporthook=_report)
    print()
    print(f"[done] {TARGET.stat().st_size / 1e6:.1f} MB")
    return TARGET


if __name__ == "__main__":
    force = "--force" in sys.argv[1:]
    download(force=force)
