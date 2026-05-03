"""Scan video, log detection count + class breakdown, return busiest frames."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from visionflow.config import DetectorConfig
from visionflow.detector import Detector

VIDEO = ROOT / "data" / "taxilla_traffic.mp4"


def main() -> None:
    detector = Detector(DetectorConfig(
        weights="models/uvh26.pt", conf=0.25,
        classes=None, device="cpu",
    ))
    cap = cv2.VideoCapture(str(VIDEO))
    counts: list[tuple[int, int]] = []
    cls_totals: Counter[str] = Counter()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        dets = detector(frame)
        counts.append((idx, len(dets)))
        for d in dets:
            cls_totals[d.class_name] += 1
    cap.release()
    counts.sort(key=lambda x: -x[1])
    print("Top 12 busiest frames:")
    for fi, n in counts[:12]:
        print(f"  frame {fi}: {n} detections")
    print(f"\nTotal frames: {idx}")
    print(f"Frames with 0 detections: {sum(1 for _, n in counts if n == 0)}")
    print("\nClass totals across all frames:")
    for name, total in cls_totals.most_common():
        print(f"  {name:12s} {total}")


if __name__ == "__main__":
    main()
