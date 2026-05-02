from __future__ import annotations

from collections import deque

import numpy as np

from visionflow.analytics import HeatmapAccumulator, LineCounter, SpeedEstimator
from visionflow.config import HeatmapConfig, LineConfig, SpeedConfig
from visionflow.tracker import Track


def _track(tid: int, x: float, y: float, history: list[tuple[float, float]] | None = None) -> Track:
    bbox = (x - 5, y - 5, x + 5, y + 5)
    t = Track(track_id=tid, bbox=bbox, class_id=0, class_name="obj", score=0.9)
    if history:
        t.history = deque(history, maxlen=64)
    return t


def test_line_counter_in_then_out():
    counter = LineCounter(LineConfig(name="gate", start=(0, 50), end=(100, 50)))
    counter.update([_track(1, 50, 10)])
    counter.update([_track(1, 50, 90)])
    assert counter.in_count + counter.out_count == 1


def test_line_counter_no_double_counting():
    counter = LineCounter(LineConfig(name="gate", start=(0, 50), end=(100, 50)))
    counter.update([_track(1, 50, 10)])
    counter.update([_track(1, 50, 90)])
    counter.update([_track(1, 50, 10)])
    counter.update([_track(1, 50, 90)])
    assert counter.in_count + counter.out_count == 1


def test_line_counter_independent_tracks():
    counter = LineCounter(LineConfig(name="gate", start=(0, 50), end=(100, 50)))
    counter.update([_track(1, 50, 10), _track(2, 30, 90)])
    counter.update([_track(1, 50, 90), _track(2, 30, 10)])
    assert counter.in_count == 1
    assert counter.out_count == 1


def test_heatmap_accumulator_decays_and_overlays():
    cfg = HeatmapConfig(enabled=True, decay=0.5, radius=5, alpha=0.5)
    hm = HeatmapAccumulator(cfg, shape=(60, 60))
    hm.add([(30.0, 30.0)])
    peak1 = hm._buf.max()  # noqa: SLF001
    hm.add([])
    peak2 = hm._buf.max()  # noqa: SLF001
    assert peak2 < peak1
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    out = hm.overlay(frame)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_speed_estimator_disabled_returns_empty():
    estimator = SpeedEstimator(SpeedConfig(enabled=False))
    assert estimator.update([_track(1, 100, 100)]) == {}


def test_speed_estimator_with_identity_quad():
    cfg = SpeedConfig(
        enabled=True,
        image_quad=[(0, 0), (100, 0), (100, 100), (0, 100)],
        world_size_m=(100.0, 100.0),
        fps=10.0,
        smoothing=2,
    )
    estimator = SpeedEstimator(cfg)
    estimator.update([_track(1, 0, 0)])
    speeds = estimator.update([_track(1, 10, 0)])
    assert 1 in speeds
    assert speeds[1] > 0
