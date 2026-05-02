from __future__ import annotations

from pathlib import Path

import pytest

from visionflow.config import PipelineConfig, SpeedConfig


def test_default_config_round_trips_through_yaml(tmp_path: Path):
    cfg = PipelineConfig()
    out = tmp_path / "cfg.yaml"
    cfg.to_yaml(out)
    loaded = PipelineConfig.from_yaml(out)
    assert loaded.detector.weights == cfg.detector.weights
    assert loaded.tracker.iou_threshold == cfg.tracker.iou_threshold


def test_speed_quad_must_have_four_points():
    with pytest.raises(ValueError, match="exactly 4 points"):
        SpeedConfig(enabled=True, image_quad=[(0, 0), (1, 1)])


def test_full_config_loads(tmp_path: Path):
    yaml = """
source: "data/sample.mp4"
detector:
  weights: yolov8n.pt
  conf: 0.3
  classes: [2, 5, 7]
tracker:
  iou_threshold: 0.4
  min_hits: 2
lines:
  - name: north
    start: [0, 400]
    end: [1280, 400]
heatmap:
  enabled: true
  decay: 0.92
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml, encoding="utf-8")
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.detector.classes == [2, 5, 7]
    assert cfg.lines[0].name == "north"
    assert cfg.heatmap.enabled is True
