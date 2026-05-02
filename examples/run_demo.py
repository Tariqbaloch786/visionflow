"""Programmatic demo: build a config in code and run the pipeline.

Usage:
    python examples/run_demo.py path/to/video.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

from visionflow.config import (
    DetectorConfig,
    HeatmapConfig,
    LineConfig,
    OutputConfig,
    PipelineConfig,
    TrackerConfig,
)
from visionflow.pipeline import Pipeline


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    source = sys.argv[1]

    cfg = PipelineConfig(
        source=source,
        detector=DetectorConfig(weights="yolov8n.pt", conf=0.3, classes=[0, 2, 5, 7]),
        tracker=TrackerConfig(iou_threshold=0.3, min_hits=3, max_age=30),
        lines=[LineConfig(name="gate", start=(0, 400), end=(1280, 400))],
        heatmap=HeatmapConfig(enabled=True),
        output=OutputConfig(video="outputs/demo.mp4", csv="outputs/demo.csv", show=False),
    )
    Path("outputs").mkdir(exist_ok=True)
    summary = Pipeline(cfg).run()
    print("Done:", summary)


if __name__ == "__main__":
    main()
