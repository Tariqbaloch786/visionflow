"""Typed configuration for the analytics pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DetectorConfig(BaseModel):
    weights: str = "yolov8n.pt"
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    classes: list[int] | None = None
    imgsz: int = 640
    device: str = "cpu"


class TrackerConfig(BaseModel):
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_age: int = 30
    min_hits: int = 3


class LineConfig(BaseModel):
    name: str
    start: tuple[int, int]
    end: tuple[int, int]


class SpeedConfig(BaseModel):
    enabled: bool = False
    image_quad: list[tuple[int, int]] = Field(default_factory=list)
    world_size_m: tuple[float, float] = (10.0, 20.0)
    fps: float = 30.0
    smoothing: int = 5

    @field_validator("image_quad")
    @classmethod
    def _validate_quad(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if v and len(v) != 4:
            raise ValueError("image_quad must contain exactly 4 points")
        return v


class HeatmapConfig(BaseModel):
    enabled: bool = False
    decay: float = Field(0.95, ge=0.0, le=1.0)
    radius: int = 25
    alpha: float = Field(0.45, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    video: str | None = None
    csv: str | None = None
    show: bool = True


class PipelineConfig(BaseModel):
    source: str = "0"
    detector: DetectorConfig = DetectorConfig()
    secondary_detector: DetectorConfig | None = None
    tracker: TrackerConfig = TrackerConfig()
    lines: list[LineConfig] = Field(default_factory=list)
    speed: SpeedConfig = SpeedConfig()
    heatmap: HeatmapConfig = HeatmapConfig()
    output: OutputConfig = OutputConfig()
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
