"""Run the real visionflow pipeline on a real video and capture frame stills.

Output: docs/img/tracking.png, heatmap.png, demo.gif — all frames produced by
the actual pipeline (UVH-26 vehicle detector + COCO YOLOv8n for pedestrians
+ IoU tracker + analytics).

The default source is data/taxilla_traffic.mp4 (a 13s busy-street clip from
Taxila, Pakistan; mixed people / cars / trucks / rickshaws / bikes). If
that file is missing the script falls back to the public Roboflow vehicles
clip and downscales it to 720p.

Run scripts/download_models.py first to fetch the UVH-26 weights.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FALLBACK_URL = (
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"
)

from visionflow.config import (  # noqa: E402
    DetectorConfig,
    HeatmapConfig,
    LineConfig,
    OutputConfig,
    PipelineConfig,
    TrackerConfig,
)
from visionflow.pipeline import Pipeline  # noqa: E402
from visionflow.visualization import draw_hud  # noqa: E402

VIDEO = ROOT / "data" / "taxilla_traffic.mp4"
FALLBACK_RAW = ROOT / "data" / "traffic.mp4"
FALLBACK_DOWNSCALED = ROOT / "data" / "traffic_720.mp4"
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def build_config(heatmap: bool) -> PipelineConfig:
    return PipelineConfig(
        source=str(VIDEO),
        detector=DetectorConfig(
            weights="models/uvh26.pt",
            conf=0.30,
            iou=0.45,
            classes=None,
            imgsz=640,
            device="cpu",
        ),
        secondary_detector=DetectorConfig(
            weights="yolov8n.pt",
            conf=0.30,
            iou=0.45,
            classes=[0],
            imgsz=640,
            device="cpu",
        ),
        tracker=TrackerConfig(iou_threshold=0.30, max_age=20, min_hits=2),
        lines=[LineConfig(name="counts", start=(0, 760), end=(576, 760))],
        heatmap=HeatmapConfig(enabled=heatmap, decay=0.99, radius=22, alpha=0.55),
        output=OutputConfig(show=False),
    )


SCREENSHOT_W = 1280


def _resize_for_web(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= SCREENSHOT_W:
        return frame
    new_h = int(h * SCREENSHOT_W / w)
    return cv2.resize(frame, (SCREENSHOT_W, new_h), interpolation=cv2.INTER_AREA)


def capture_pass(label: str, target_frame: int, save_as: Path,
                 heatmap: bool) -> None:
    print(f"[{label}] running until frame {target_frame}...")
    pipeline = Pipeline(build_config(heatmap=heatmap))
    last_frame = None
    for idx, frame, tracks, _speeds in pipeline.stream():
        last_frame = (idx, frame, tracks)
        if idx == target_frame:
            extras = []
            for c in pipeline.line_counters:
                extras.append(f"{c.config.name}: in={c.in_count} out={c.out_count}")
            annotated = draw_hud(frame, pipeline._fps_ema, len(tracks),  # noqa: SLF001
                                 extra=extras)
            cv2.imwrite(str(save_as), _resize_for_web(annotated))
            print(f"[{label}] saved {save_as.name}  tracks={len(tracks)}  "
                  f"fps_ema={pipeline._fps_ema:.1f}")  # noqa: SLF001
            return
    if last_frame is not None:
        idx, frame, tracks = last_frame
        annotated = draw_hud(frame, pipeline._fps_ema, len(tracks))  # noqa: SLF001
        cv2.imwrite(str(save_as), _resize_for_web(annotated))
        print(f"[{label}] video ended at frame {idx}; saved {save_as.name}")


def ensure_video() -> None:
    if VIDEO.exists():
        return
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print(f"{VIDEO.name} not found; falling back to public Roboflow clip ...")
    if not FALLBACK_RAW.exists():
        urllib.request.urlretrieve(FALLBACK_URL, FALLBACK_RAW)
    if not FALLBACK_DOWNSCALED.exists():
        src = cv2.VideoCapture(str(FALLBACK_RAW))
        w, h = int(src.get(3)), int(src.get(4))
        fps = src.get(5)
        new_w = 1280
        new_h = int(h * new_w / w)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        dst = cv2.VideoWriter(str(FALLBACK_DOWNSCALED), fourcc, fps, (new_w, new_h))
        while True:
            ok, fr = src.read()
            if not ok:
                break
            dst.write(cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_AREA))
        src.release()
        dst.release()
    raise SystemExit(
        f"Place a video at {VIDEO} or wire FALLBACK_DOWNSCALED into the "
        f"capture targets. Sample fallback ready at {FALLBACK_DOWNSCALED}."
    )


def render_gif(out_path: Path, start_frame: int, end_frame: int,
               every_n: int = 2, gif_width: int = 720) -> None:
    """Run the pipeline and save selected frames as an animated GIF."""
    from PIL import Image  # local import; only needed for GIF

    print(f"[gif] capturing frames {start_frame}..{end_frame} every {every_n}")
    pipeline = Pipeline(build_config(heatmap=False))
    images: list[Image.Image] = []
    for idx, frame, tracks, _speeds in pipeline.stream():
        if idx < start_frame:
            continue
        if idx > end_frame:
            break
        if (idx - start_frame) % every_n != 0:
            continue
        extras = []
        for c in pipeline.line_counters:
            extras.append(f"{c.config.name}: in={c.in_count} out={c.out_count}")
        annotated = draw_hud(frame, pipeline._fps_ema, len(tracks), extra=extras)  # noqa: SLF001
        h, w = annotated.shape[:2]
        new_h = int(h * gif_width / w)
        small = cv2.resize(annotated, (gif_width, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    if not images:
        print("[gif] no frames captured")
        return
    palette = images[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    size_kb = out_path.stat().st_size // 1024
    print(f"[gif] saved {out_path.name} ({len(images)} frames, {size_kb} KB)")


def main() -> None:
    ensure_video()

    capture_pass("tracking", target_frame=184,
                 save_as=OUT / "tracking.png", heatmap=False)
    capture_pass("heatmap",  target_frame=300,
                 save_as=OUT / "heatmap.png",  heatmap=True)

    render_gif(OUT / "demo.gif", start_frame=20, end_frame=320,
               every_n=6, gif_width=380)


if __name__ == "__main__":
    main()
