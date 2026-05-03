"""YOLO-based detector with a thin, framework-agnostic data interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from visionflow.config import DetectorConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str = ""

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.bbox

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


class Detector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._names: dict[int, str] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.config.weights)
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            self._names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            self._names = {i: str(n) for i, n in enumerate(names)}

    def class_name(self, class_id: int) -> str:
        return self._names.get(int(class_id), str(class_id))

    def __call__(self, frame: NDArray[np.uint8]) -> list[Detection]:
        self._ensure_loaded()
        results = self._model.predict(
            source=frame,
            conf=self.config.conf,
            iou=self.config.iou,
            classes=self.config.classes,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, c in zip(xyxy, confs, cls, strict=False):
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    class_id=int(c),
                    class_name=self.class_name(int(c)),
                )
            )
        return detections
