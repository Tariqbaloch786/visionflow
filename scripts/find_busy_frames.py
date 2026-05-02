"""Scan video, log detection count per frame, return the busiest ones."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from visionflow.config import DetectorConfig  # noqa: E402
from visionflow.detector import Detector  # noqa: E402

VIDEO = ROOT / "data" / "traffic.mp4"


def main() -> None:
    detector = Detector(DetectorConfig(
        weights="yolov8n.pt", conf=0.25,
        classes=[0, 1, 2, 3, 5, 7], device="cpu",
    ))
    cap = cv2.VideoCapture(str(VIDEO))
    counts = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        dets = detector(frame)
        counts.append((idx, len(dets)))
    cap.release()
    counts.sort(key=lambda x: -x[1])
    print("Top 10 busiest frames:")
    for fi, n in counts[:10]:
        print(f"  frame {fi}: {n} detections")
    print(f"\nTotal frames: {idx}")
    print(f"Frames with 0 detections: {sum(1 for _, n in counts if n == 0)}")


if __name__ == "__main__":
    main()
