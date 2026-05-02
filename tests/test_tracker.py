from __future__ import annotations

from visionflow.config import TrackerConfig
from visionflow.detector import Detection
from visionflow.tracker import IoUTracker, iou


def make_det(x1: float, y1: float, x2: float, y2: float, cls: int = 0) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), score=0.9, class_id=cls, class_name="obj")


def test_iou_perfect_overlap():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial():
    score = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.14 < score < 0.15


def test_track_id_persists_across_frames():
    tracker = IoUTracker(TrackerConfig(min_hits=1, max_age=5, iou_threshold=0.3))
    t1 = tracker.update([make_det(0, 0, 50, 50)])
    t2 = tracker.update([make_det(2, 2, 52, 52)])
    t3 = tracker.update([make_det(4, 4, 54, 54)])
    assert len(t1) == len(t2) == len(t3) == 1
    assert t1[0].track_id == t2[0].track_id == t3[0].track_id


def test_lost_track_expires_after_max_age():
    tracker = IoUTracker(TrackerConfig(min_hits=1, max_age=2, iou_threshold=0.3))
    tracker.update([make_det(0, 0, 50, 50)])
    for _ in range(5):
        tracker.update([])
    assert tracker.tracks == []


def test_min_hits_delays_confirmation():
    tracker = IoUTracker(TrackerConfig(min_hits=3, max_age=5, iou_threshold=0.3))
    assert tracker.update([make_det(0, 0, 50, 50)]) == []
    assert tracker.update([make_det(1, 1, 51, 51)]) == []
    confirmed = tracker.update([make_det(2, 2, 52, 52)])
    assert len(confirmed) == 1


def test_separate_objects_get_separate_ids():
    tracker = IoUTracker(TrackerConfig(min_hits=1, max_age=5, iou_threshold=0.3))
    tracks = tracker.update([make_det(0, 0, 50, 50), make_det(200, 200, 250, 250)])
    assert len({t.track_id for t in tracks}) == 2


def test_class_change_does_not_match():
    tracker = IoUTracker(TrackerConfig(min_hits=1, max_age=5, iou_threshold=0.3))
    a = tracker.update([make_det(0, 0, 50, 50, cls=0)])
    b = tracker.update([make_det(0, 0, 50, 50, cls=1)])
    original_id = a[0].track_id
    new_class_tracks = [t for t in b if t.class_id == 1]
    assert len(new_class_tracks) == 1
    assert new_class_tracks[0].track_id != original_id
