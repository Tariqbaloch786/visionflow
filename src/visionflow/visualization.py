"""Frame annotation helpers: boxes, trails, lines, HUD."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from visionflow.analytics import LineCounter
    from visionflow.tracker import Track


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 7)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))  # type: ignore[return-value]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    speeds: dict[int, float] | None = None,
    show_trail: bool = True,
) -> np.ndarray:
    speeds = speeds or {}
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        color = _color_for_id(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{t.track_id} {t.class_name}"
        if t.track_id in speeds:
            label += f" {speeds[t.track_id]:.0f} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

        if show_trail and len(t.history) >= 2:
            pts = np.array(t.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
    return frame


def draw_lines(frame: np.ndarray, counters: list[LineCounter]) -> np.ndarray:
    for c in counters:
        a, b = c.config.start, c.config.end
        cv2.line(frame, a, b, (0, 200, 255), 2)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        text = f"{c.config.name}: in={c.in_count} out={c.out_count}"
        cv2.putText(
            frame, text, (mid[0] + 6, mid[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_tracks: int,
    extra: list[str] | None = None,
) -> np.ndarray:
    h, _ = frame.shape[:2]
    lines = [f"FPS: {fps:5.1f}", f"Active tracks: {n_tracks}"]
    if extra:
        lines.extend(extra)
    pad = 8
    box_h = 22 * len(lines) + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - box_h), (260, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        y = h - box_h + pad + 18 + i * 22
        cv2.putText(
            frame, line, (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame
