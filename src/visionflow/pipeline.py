"""End-to-end pipeline: video source → detector → tracker → analytics → output."""
from __future__ import annotations

import csv
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from visionflow.analytics import HeatmapAccumulator, LineCounter, SpeedEstimator
from visionflow.config import PipelineConfig
from visionflow.detector import Detector
from visionflow.tracker import IoUTracker
from visionflow.visualization import draw_hud, draw_lines, draw_tracks

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)


def _open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


SECONDARY_CLASS_ID_OFFSET = 1000

# A person riding a two-wheeler is picked up by BOTH detectors (UVH-26 sees the
# vehicle, COCO sees the rider). For traffic analytics a rider+bike is one object,
# so we drop the person box when it sits mostly inside a two-wheeler/bicycle box.
# A standalone pedestrian's box is not contained in any vehicle, so it survives.
RIDER_VEHICLE_CLASSES = {"Two-wheeler", "Bicycle"}
RIDER_CONTAINMENT = 0.5


def _containment(inner: tuple[float, float, float, float],
                 outer: tuple[float, float, float, float]) -> float:
    """Fraction of the `inner` box's area that lies inside `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = Detector(config.detector)
        self.secondary_detector = (
            Detector(config.secondary_detector) if config.secondary_detector is not None else None
        )
        self.tracker = IoUTracker(config.tracker)
        self.line_counters = [LineCounter(c) for c in config.lines]
        self.speed = SpeedEstimator(config.speed)
        self._heatmap: HeatmapAccumulator | None = None
        self._csv_writer: Any = None
        self._frame_idx = 0
        self._fps_ema = 0.0

    def _init_heatmap(self, shape: tuple[int, int]) -> None:
        if self.config.heatmap.enabled and self._heatmap is None:
            self._heatmap = HeatmapAccumulator(self.config.heatmap, shape)

    def _process_frame(self, frame):  # noqa: ANN001
        detections = self.detector(frame)
        if self.secondary_detector is not None:
            riders = [d.bbox for d in detections if d.class_name in RIDER_VEHICLE_CLASSES]
            for d in self.secondary_detector(frame):
                if any(_containment(d.bbox, v) >= RIDER_CONTAINMENT for v in riders):
                    continue  # rider on a two-wheeler — keep the vehicle, drop the person
                d.class_id += SECONDARY_CLASS_ID_OFFSET
                detections.append(d)
        tracks = self.tracker.update(detections)
        speeds = self.speed.update(tracks)
        for c in self.line_counters:
            c.update(tracks)

        self._init_heatmap(frame.shape[:2])
        if self._heatmap is not None:
            self._heatmap.add([t.center for t in tracks])
            frame = self._heatmap.overlay(frame)

        frame = draw_tracks(frame, tracks, speeds=speeds)
        if self.line_counters:
            frame = draw_lines(frame, self.line_counters)
        return frame, tracks, speeds

    def stream(self) -> Iterator[tuple[int, cv2.typing.MatLike, list, dict[int, float]]]:
        cap = _open_capture(self.config.source)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t0 = time.perf_counter()
                frame, tracks, speeds = self._process_frame(frame)
                dt = time.perf_counter() - t0
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                self._fps_ema = inst_fps if self._fps_ema == 0 else 0.9 * self._fps_ema + 0.1 * inst_fps
                self._frame_idx += 1
                yield self._frame_idx, frame, tracks, speeds
        finally:
            cap.release()

    def run(self) -> dict[str, int | float]:
        out_cfg = self.config.output
        writer: cv2.VideoWriter | None = None

        with ExitStack() as stack:
            csv_file = None
            if out_cfg.csv:
                Path(out_cfg.csv).parent.mkdir(parents=True, exist_ok=True)
                csv_file = stack.enter_context(open(out_cfg.csv, "w", newline="", encoding="utf-8"))
                self._csv_writer = csv.writer(csv_file)
                self._csv_writer.writerow(
                    ["frame", "track_id", "class", "x1", "y1", "x2", "y2", "speed_kmh"]
                )

            for idx, frame, tracks, speeds in self.stream():
                extras = []
                for c in self.line_counters:
                    extras.append(f"{c.config.name}: {c.in_count}/{c.out_count}")
                frame = draw_hud(frame, self._fps_ema, len(tracks), extra=extras)

                if writer is None and out_cfg.video:
                    Path(out_cfg.video).parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_cfg.video, fourcc, 30.0, (w, h))
                if writer is not None:
                    writer.write(frame)

                if self._csv_writer is not None:
                    for t in tracks:
                        x1, y1, x2, y2 = t.bbox
                        self._csv_writer.writerow(
                            [idx, t.track_id, t.class_name,
                             f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                             f"{speeds.get(t.track_id, 0.0):.1f}"]
                        )

                if out_cfg.show:
                    cv2.imshow("VisionFlow", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if writer is not None:
                writer.release()
            if out_cfg.show:
                cv2.destroyAllWindows()

        return {
            "frames": self._frame_idx,
            "fps_avg": round(self._fps_ema, 2),
            **{f"line_{c.config.name}_in": c.in_count for c in self.line_counters},
            **{f"line_{c.config.name}_out": c.out_count for c in self.line_counters},
        }
