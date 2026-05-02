"""Generate demo screenshots from the visionflow visualization code.

These are not real footage — they're synthetic scenes drawn with cv2 so the
README can show the visualization style without shipping a video. Replace
docs/img/*.png with real captures once you record some.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionflow.analytics import HeatmapAccumulator, LineCounter
from visionflow.config import HeatmapConfig, LineConfig
from visionflow.tracker import Track
from visionflow.visualization import draw_hud, draw_lines, draw_tracks


W, H = 1280, 720
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def road_background(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 78, dtype=np.uint8)
    noise = rng.normal(0, 10, (H, W)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)

    cv2.rectangle(img, (0, 0), (W, 80), (95, 110, 70), -1)
    cv2.rectangle(img, (0, 80), (W, 110), (50, 50, 50), -1)
    cv2.rectangle(img, (0, H - 80), (W, H), (95, 110, 70), -1)
    cv2.rectangle(img, (0, H - 110), (W, H - 80), (50, 50, 50), -1)

    for x in (W // 4, W // 2, 3 * W // 4):
        for y in range(120, H - 120, 60):
            cv2.rectangle(img, (x - 4, y), (x + 4, y + 30), (235, 235, 200), -1)

    cv2.line(img, (0, 110), (W, 110), (240, 240, 240), 2)
    cv2.line(img, (0, H - 110), (W, H - 110), (240, 240, 240), 2)
    return img


def make_track(tid: int, cx: int, cy: int, w: int, h: int,
               trail: list[tuple[int, int]], cls_id: int = 2,
               cls_name: str = "car") -> Track:
    bbox = (float(cx - w // 2), float(cy - h // 2), float(cx + w // 2), float(cy + h // 2))
    t = Track(track_id=tid, bbox=bbox, class_id=cls_id, class_name=cls_name, score=0.92)
    t.confirmed = True
    t.history = deque(trail, maxlen=64)
    return t


def scene_a_tracks() -> list[Track]:
    return [
        make_track(7,  330, 240, 110, 60,
                   [(330, y) for y in range(140, 240, 6)]),
        make_track(12, 660, 380, 130, 70,
                   [(660, y) for y in range(180, 380, 6)]),
        make_track(18, 980, 510, 140, 80, [(980, y) for y in range(280, 510, 6)],
                   cls_id=7, cls_name="truck"),
        make_track(23, 200, 555, 90, 50,
                   [(200, y) for y in range(610, 555, -4)]),
        make_track(31, 850, 175, 95, 55,
                   [(x, 175) for x in range(720, 850, 6)], cls_id=2, cls_name="car"),
    ]


def render_tracking() -> None:
    img = road_background()
    tracks = scene_a_tracks()
    speeds = {7: 42.0, 12: 38.0, 18: 51.0, 23: 27.0, 31: 33.0}
    img = draw_tracks(img, tracks, speeds=speeds)

    line = LineConfig(name="counts", start=(60, 460), end=(W - 60, 460))
    counter = LineCounter(line)
    counter.in_count, counter.out_count = 38, 14
    img = draw_lines(img, [counter])

    img = draw_hud(img, fps=28.4, n_tracks=len(tracks),
                   extra=["Detector: yolov8n  conf=0.30",
                          "Source: traffic_cam_03.mp4",
                          "Counts (in/out): 38 / 14"])
    cv2.imwrite(str(OUT / "tracking.png"), img)


def render_heatmap() -> None:
    img = road_background(seed=11)
    cfg = HeatmapConfig(enabled=True, decay=0.97, radius=28, alpha=0.55)
    hm = HeatmapAccumulator(cfg, shape=(H, W))

    rng = np.random.default_rng(7)
    for _ in range(120):
        x = int(rng.normal(W * 0.5, 90))
        y = int(rng.normal(H * 0.55, 140))
        hm.add([(x, y)])
    for x in range(120, W - 120, 18):
        hm.add([(x, 250 + int(40 * np.sin(x / 80)))])
    for _ in range(40):
        hm.add([(int(rng.normal(900, 50)), int(rng.normal(380, 30)))])

    img = hm.overlay(img)

    tracks = [
        make_track(12, 660, 380, 130, 70,
                   [(660, y) for y in range(220, 380, 6)]),
        make_track(18, 980, 410, 140, 80,
                   [(980, y) for y in range(300, 410, 6)],
                   cls_id=7, cls_name="truck"),
    ]
    img = draw_tracks(img, tracks, speeds={12: 38.0, 18: 47.0})
    img = draw_hud(img, fps=27.9, n_tracks=2,
                   extra=["Heatmap: decay=0.97  radius=28",
                          "Hotspot: lane 2 @ (900, 380)"])
    cv2.imwrite(str(OUT / "heatmap.png"), img)


def render_dashboard_mock() -> None:
    bg = np.full((760, 1280, 3), 248, dtype=np.uint8)
    cv2.rectangle(bg, (0, 0), (1280, 56), (38, 38, 42), -1)
    cv2.putText(bg, "VisionFlow  |  Real-Time Tracking & Analytics",
                (24, 36), cv2.FONT_HERSHEY_DUPLEX, 0.8, (235, 235, 240), 1, cv2.LINE_AA)

    cv2.rectangle(bg, (0, 56), (260, 760), (240, 240, 244), -1)
    cv2.putText(bg, "Configuration", (20, 96),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 60), 1, cv2.LINE_AA)
    for i, (label, value) in enumerate([
        ("Source", "traffic_cam_03.mp4"),
        ("Detector", "yolov8n.pt"),
        ("Conf threshold", "0.30"),
        ("Tracker IoU", "0.30"),
        ("Lines", "1 (counts)"),
        ("Heatmap", "enabled"),
    ]):
        y = 130 + i * 32
        cv2.putText(bg, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (110, 110, 120), 1, cv2.LINE_AA)
        cv2.putText(bg, value, (20, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (40, 40, 50), 1, cv2.LINE_AA)

    cv2.rectangle(bg, (16, 420), (244, 470), (75, 130, 230), -1)
    cv2.putText(bg, "Run pipeline", (60, 452),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    preview = road_background(seed=3)
    preview = cv2.resize(preview, (640, 360))
    tracks = scene_a_tracks()
    scaled = []
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        nb = (x1 / 2.0, y1 / 2.0, x2 / 2.0, y2 / 2.0)
        nt = Track(track_id=t.track_id, bbox=nb, class_id=t.class_id,
                   class_name=t.class_name, score=t.score)
        nt.confirmed = True
        nt.history = deque([(p[0] / 2.0, p[1] / 2.0) for p in t.history], maxlen=64)
        scaled.append(nt)
    preview = draw_tracks(preview, scaled, speeds={7: 42, 12: 38, 18: 51, 23: 27, 31: 33})
    bg[80:440, 280:920] = preview
    cv2.rectangle(bg, (280, 80), (920, 440), (180, 180, 190), 1)

    metrics = [
        ("Active tracks", "5"),
        ("FPS (EMA)", "28.4"),
        ("Counted in", "38"),
        ("Counted out", "14"),
    ]
    for i, (label, val) in enumerate(metrics):
        x = 940 + (i % 2) * 160
        y = 96 + (i // 2) * 90
        cv2.rectangle(bg, (x, y), (x + 140, y + 70), (255, 255, 255), -1)
        cv2.rectangle(bg, (x, y), (x + 140, y + 70), (215, 215, 220), 1)
        cv2.putText(bg, label, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (110, 110, 120), 1, cv2.LINE_AA)
        cv2.putText(bg, val, (x + 12, y + 56), cv2.FONT_HERSHEY_DUPLEX,
                    0.85, (38, 38, 48), 1, cv2.LINE_AA)

    cv2.rectangle(bg, (940, 290), (1260, 440), (255, 255, 255), -1)
    cv2.rectangle(bg, (940, 290), (1260, 440), (215, 215, 220), 1)
    cv2.putText(bg, "Counts over time", (954, 314),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 70), 1, cv2.LINE_AA)
    rng = np.random.default_rng(2)
    pts_in = np.array([(954 + i * 6, 420 - int(60 * (i / 50) - rng.normal(0, 2)))
                       for i in range(50)], dtype=np.int32)
    pts_out = np.array([(954 + i * 6, 420 - int(20 * (i / 50) + rng.normal(0, 1.5)))
                        for i in range(50)], dtype=np.int32)
    cv2.polylines(bg, [pts_in.reshape(-1, 1, 2)], False, (75, 130, 230), 2)
    cv2.polylines(bg, [pts_out.reshape(-1, 1, 2)], False, (220, 90, 90), 2)

    cv2.rectangle(bg, (280, 460), (1260, 740), (255, 255, 255), -1)
    cv2.rectangle(bg, (280, 460), (1260, 740), (215, 215, 220), 1)
    cv2.putText(bg, "Recent tracks", (296, 488),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 70), 1, cv2.LINE_AA)
    cols = ["frame", "id", "class", "x1,y1,x2,y2", "speed (km/h)"]
    for i, c in enumerate(cols):
        cv2.putText(bg, c, (310 + i * 190, 522), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (110, 110, 120), 1, cv2.LINE_AA)
    rows = [
        ("412", "7",  "car",   "275,210,385,270", "42.1"),
        ("412", "12", "car",   "595,345,725,415", "38.0"),
        ("412", "18", "truck", "910,470,1050,550", "51.4"),
        ("412", "23", "car",   "155,530,245,580", "26.7"),
        ("412", "31", "car",   "802,148,898,202", "33.2"),
    ]
    for r, row in enumerate(rows):
        for i, val in enumerate(row):
            cv2.putText(bg, val, (310 + i * 190, 558 + r * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 50), 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT / "dashboard.png"), bg)


def main() -> None:
    render_tracking()
    render_heatmap()
    render_dashboard_mock()
    print("Wrote screenshots to", OUT)


if __name__ == "__main__":
    main()
