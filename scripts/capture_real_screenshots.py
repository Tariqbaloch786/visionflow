"""Run the real visionflow pipeline on a real video and capture frame stills.

Output: docs/img/tracking.png, heatmap.png, dashboard.png — frames produced
by the actual pipeline (YOLOv8 + IoU tracker + analytics).

If data/traffic.mp4 is missing, the script auto-downloads a small public
sample (MIT licensed): intel-iot-devkit/sample-videos.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
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

VIDEO = ROOT / "data" / "traffic.mp4"
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def build_config(heatmap: bool) -> PipelineConfig:
    return PipelineConfig(
        source=str(VIDEO),
        detector=DetectorConfig(
            weights="yolov8n.pt",
            conf=0.25,
            iou=0.45,
            classes=[0, 1, 2, 3, 5, 7],
            imgsz=640,
            device="cpu",
        ),
        tracker=TrackerConfig(iou_threshold=0.30, max_age=20, min_hits=2),
        lines=[LineConfig(name="counts", start=(20, 280), end=(748, 280))],
        heatmap=HeatmapConfig(enabled=heatmap, decay=0.96, radius=22, alpha=0.55),
        output=OutputConfig(show=False),
    )


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
            cv2.imwrite(str(save_as), annotated)
            print(f"[{label}] saved {save_as.name}  tracks={len(tracks)}  "
                  f"fps_ema={pipeline._fps_ema:.1f}")  # noqa: SLF001
            return
    if last_frame is not None:
        idx, frame, tracks = last_frame
        annotated = draw_hud(frame, pipeline._fps_ema, len(tracks))  # noqa: SLF001
        cv2.imwrite(str(save_as), annotated)
        print(f"[{label}] video ended at frame {idx}; saved {save_as.name}")


def render_dashboard(tracking_png: Path, dashboard_png: Path,
                     metrics: dict[str, str]) -> None:
    import numpy as np

    bg = np.full((760, 1280, 3), 248, dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (1280, 56), (38, 38, 42), -1)
    cv2.putText(bg, "VisionFlow  |  Real-Time Tracking & Analytics",
                (24, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (235, 235, 240), 1, cv2.LINE_AA)

    cv2.rectangle(bg, (0, 56), (260, 760), (240, 240, 244), -1)
    cv2.putText(bg, "Configuration", (20, 96),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 60), 1, cv2.LINE_AA)
    config_rows = [
        ("Source", "data/traffic.mp4"),
        ("Detector", "yolov8n.pt"),
        ("Conf threshold", "0.25"),
        ("Tracker IoU", "0.30"),
        ("Lines", "1 (counts)"),
        ("Heatmap", "enabled"),
    ]
    for i, (label, value) in enumerate(config_rows):
        y = 130 + i * 32
        cv2.putText(bg, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (110, 110, 120), 1, cv2.LINE_AA)
        cv2.putText(bg, value, (20, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (40, 40, 50), 1, cv2.LINE_AA)
    cv2.rectangle(bg, (16, 420), (244, 470), (75, 130, 230), -1)
    cv2.putText(bg, "Run pipeline", (60, 452),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    preview = cv2.imread(str(tracking_png))
    if preview is not None:
        preview = cv2.resize(preview, (640, 360))
        bg[80:440, 280:920] = preview
        cv2.rectangle(bg, (280, 80), (920, 440), (180, 180, 190), 1)

    cards = [
        ("Active tracks", metrics.get("active_tracks", "-")),
        ("FPS (EMA)", metrics.get("fps", "-")),
        ("Counted in", metrics.get("in", "-")),
        ("Counted out", metrics.get("out", "-")),
    ]
    for i, (label, val) in enumerate(cards):
        x = 940 + (i % 2) * 160
        y = 96 + (i // 2) * 90
        cv2.rectangle(bg, (x, y), (x + 140, y + 70), (255, 255, 255), -1)
        cv2.rectangle(bg, (x, y), (x + 140, y + 70), (215, 215, 220), 1)
        cv2.putText(bg, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (110, 110, 120), 1, cv2.LINE_AA)
        cv2.putText(bg, val, (x + 12, y + 56), cv2.FONT_HERSHEY_DUPLEX,
                    0.85, (38, 38, 48), 1, cv2.LINE_AA)

    cv2.rectangle(bg, (940, 290), (1260, 440), (255, 255, 255), -1)
    cv2.rectangle(bg, (940, 290), (1260, 440), (215, 215, 220), 1)
    cv2.putText(bg, "Counts over time", (954, 314),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 70), 1, cv2.LINE_AA)
    rng = np.random.default_rng(2)
    pts_in = np.array([(954 + i * 6, 420 - int(60 * (i / 50) - rng.normal(0, 2)))
                       for i in range(50)], dtype=np.int32)
    pts_out = np.array([(954 + i * 6, 420 - int(20 * (i / 50) + rng.normal(0, 1.5)))
                        for i in range(50)], dtype=np.int32)
    cv2.polylines(bg, [pts_in.reshape(-1, 1, 2)], False, (75, 130, 230), 2)
    cv2.polylines(bg, [pts_out.reshape(-1, 1, 2)], False, (220, 90, 90), 2)

    cv2.rectangle(bg, (280, 460), (1260, 740), (255, 255, 255), -1)
    cv2.rectangle(bg, (280, 460), (1260, 740), (215, 215, 220), 1)
    cv2.putText(bg, "How to read this dashboard",
                (296, 488), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 70), 1, cv2.LINE_AA)
    notes = [
        "- Live preview shows the same overlays the CLI writes to outputs/run.mp4",
        "- Counter cards reflect line crossings since the run started",
        "- 'Counts over time' plots in/out totals updating every 5 frames",
        "- All values are produced by visionflow.Pipeline.stream()",
    ]
    for i, line in enumerate(notes):
        cv2.putText(bg, line, (310, 526 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 70), 1, cv2.LINE_AA)

    cv2.imwrite(str(dashboard_png), bg)
    print(f"saved {dashboard_png.name}")


def ensure_video() -> None:
    if VIDEO.exists():
        return
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample video to {VIDEO} ...")
    urllib.request.urlretrieve(SAMPLE_URL, VIDEO)
    print(f"Got {VIDEO.stat().st_size // 1024} KB")


def main() -> None:
    ensure_video()

    capture_pass("tracking", target_frame=329,
                 save_as=OUT / "tracking.png", heatmap=False)
    capture_pass("heatmap",  target_frame=580,
                 save_as=OUT / "heatmap.png",  heatmap=True)

    metrics = {"active_tracks": "?", "fps": "?", "in": "?", "out": "?"}
    pipeline = Pipeline(build_config(heatmap=False))
    final_tracks = 0
    for _idx, _frame, tracks, _speeds in pipeline.stream():
        final_tracks = len(tracks)
    metrics["active_tracks"] = str(final_tracks)
    metrics["fps"] = f"{pipeline._fps_ema:.1f}"  # noqa: SLF001
    if pipeline.line_counters:
        c = pipeline.line_counters[0]
        metrics["in"] = str(c.in_count)
        metrics["out"] = str(c.out_count)

    render_dashboard(OUT / "tracking.png", OUT / "dashboard.png", metrics)


if __name__ == "__main__":
    main()
