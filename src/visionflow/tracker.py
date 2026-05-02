"""Lightweight IoU-based multi-object tracker.

Greedy Hungarian-free assignment: sort candidate (track, det) pairs by descending IoU
and accept the highest-IoU non-conflicting pair. Adequate for portfolio demos and
avoids extra dependencies; swap in DeepSORT / ByteTrack for production.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING

from visionflow.config import TrackerConfig

if TYPE_CHECKING:
    from visionflow.detector import Detection

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(b: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    class_id: int
    class_name: str
    score: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
    confirmed: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return _center(self.bbox)

    def predict(self) -> None:
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection) -> None:
        self.bbox = det.bbox
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.score = det.score
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.center)


class IoUTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: dict[int, Track] = {}
        self._ids = count(1)

    @property
    def tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.confirmed]

    def update(self, detections: list[Detection]) -> list[Track]:
        for t in self._tracks.values():
            t.predict()

        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(self._tracks.keys())

        if self._tracks and detections:
            pairs: list[tuple[float, int, int]] = []
            for tid, t in self._tracks.items():
                for di, d in enumerate(detections):
                    if t.class_id != d.class_id:
                        continue
                    score = iou(t.bbox, d.bbox)
                    if score >= self.config.iou_threshold:
                        pairs.append((score, tid, di))

            pairs.sort(reverse=True)
            for _, tid, di in pairs:
                if tid in unmatched_tracks and di in unmatched_dets:
                    self._tracks[tid].update(detections[di])
                    unmatched_tracks.discard(tid)
                    unmatched_dets.discard(di)

        for di in unmatched_dets:
            d = detections[di]
            tid = next(self._ids)
            track = Track(
                track_id=tid,
                bbox=d.bbox,
                class_id=d.class_id,
                class_name=d.class_name,
                score=d.score,
            )
            track.history.append(track.center)
            self._tracks[tid] = track

        for tid in list(self._tracks.keys()):
            t = self._tracks[tid]
            if not t.confirmed and t.hits >= self.config.min_hits:
                t.confirmed = True
            if t.time_since_update > self.config.max_age:
                del self._tracks[tid]

        return self.tracks
