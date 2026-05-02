"""Analytics modules: line counters, homography-based speed, motion heatmap."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from visionflow.config import HeatmapConfig, LineConfig, SpeedConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from visionflow.tracker import Track


def _side(p: tuple[float, float], a: tuple[int, int], b: tuple[int, int]) -> float:
    """Signed area sign — which side of segment AB point p lies on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass
class LineCounter:
    config: LineConfig
    in_count: int = 0
    out_count: int = 0
    _last_side: dict[int, float] = field(default_factory=dict)
    _counted: set[int] = field(default_factory=set)

    def update(self, tracks: list[Track]) -> None:
        a, b = self.config.start, self.config.end
        for t in tracks:
            tid = t.track_id
            side = _side(t.center, a, b)
            prev = self._last_side.get(tid)
            self._last_side[tid] = side
            if prev is None or tid in self._counted:
                continue
            if prev == 0 or side == 0:
                continue
            if (prev < 0) != (side < 0):
                if side > 0:
                    self.in_count += 1
                else:
                    self.out_count += 1
                self._counted.add(tid)

    @property
    def total(self) -> int:
        return self.in_count + self.out_count


class SpeedEstimator:
    """Estimate ground-plane speed (km/h) via image→world homography."""

    def __init__(self, config: SpeedConfig) -> None:
        self.config = config
        self._H: NDArray[np.float64] | None = None
        self._history: dict[int, deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=max(2, config.smoothing))
        )
        self._frame_idx = 0
        if config.enabled and config.image_quad:
            self._H = self._build_homography()

    def _build_homography(self) -> NDArray[np.float64]:
        src = np.array(self.config.image_quad, dtype=np.float32)
        w, h = self.config.world_size_m
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise ValueError("Failed to compute speed homography from provided quad.")
        return H

    def _project(self, pt: tuple[float, float]) -> tuple[float, float]:
        assert self._H is not None
        v = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        x, y, w = self._H @ v
        return (float(x / w), float(y / w)) if w != 0 else (0.0, 0.0)

    def update(self, tracks: list[Track]) -> dict[int, float]:
        if not self.config.enabled or self._H is None:
            return {}
        self._frame_idx += 1
        t_now = self._frame_idx / self.config.fps
        speeds: dict[int, float] = {}
        for tr in tracks:
            wx, wy = self._project(tr.center)
            buf = self._history[tr.track_id]
            buf.append((t_now, wx, wy))
            if len(buf) >= 2:
                t0, x0, y0 = buf[0]
                t1, x1, y1 = buf[-1]
                dt = t1 - t0
                if dt > 1e-6:
                    dist = float(np.hypot(x1 - x0, y1 - y0))
                    speeds[tr.track_id] = (dist / dt) * 3.6
        return speeds


class HeatmapAccumulator:
    def __init__(self, config: HeatmapConfig, shape: tuple[int, int]) -> None:
        self.config = config
        h, w = shape
        self._buf = np.zeros((h, w), dtype=np.float32)
        self._kernel = self._gaussian_kernel(config.radius)

    @staticmethod
    def _gaussian_kernel(radius: int) -> NDArray[np.float32]:
        size = max(3, radius * 2 + 1)
        k = cv2.getGaussianKernel(size, sigma=radius / 2.0)
        kernel = (k @ k.T).astype(np.float32)
        return kernel / kernel.max()

    def add(self, points: list[tuple[float, float]]) -> None:
        if not self.config.enabled:
            return
        self._buf *= self.config.decay
        if not points:
            return
        h, w = self._buf.shape
        ks = self._kernel.shape[0]
        half = ks // 2
        for px, py in points:
            cx, cy = int(px), int(py)
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half + 1), min(h, cy + half + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            kx1, ky1 = x1 - (cx - half), y1 - (cy - half)
            kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)
            self._buf[y1:y2, x1:x2] += self._kernel[ky1:ky2, kx1:kx2]

    def overlay(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not self.config.enabled:
            return frame
        norm = self._buf
        peak = float(norm.max())
        if peak <= 1e-6:
            return frame
        scaled = np.clip(norm / peak * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1.0 - self.config.alpha, colored, self.config.alpha, 0)
