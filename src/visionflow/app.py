"""Streamlit dashboard: live preview + counter & track-count time series."""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from visionflow.config import PipelineConfig
from visionflow.pipeline import Pipeline


def _load_config(uploaded, fallback_path: str) -> PipelineConfig:
    if uploaded is not None:
        tmp = Path(".streamlit_cfg.yaml")
        tmp.write_bytes(uploaded.getvalue())
        return PipelineConfig.from_yaml(tmp)
    return PipelineConfig.from_yaml(fallback_path)


def main() -> None:
    st.set_page_config(page_title="VisionFlow Dashboard", layout="wide")
    st.title("VisionFlow — Real-Time Tracking & Analytics")

    with st.sidebar:
        st.header("Configuration")
        cfg_path = st.text_input("Config YAML", value="examples/sample_config.yaml")
        uploaded = st.file_uploader("…or upload a config", type=["yaml", "yml"])
        source_override = st.text_input("Source override (optional)", value="")
        run_btn = st.button("Run pipeline", type="primary")

    if not run_btn:
        st.info("Configure the pipeline in the sidebar and click **Run pipeline** to start.")
        return

    try:
        cfg = _load_config(uploaded, cfg_path)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load config: {e}")
        return

    if source_override.strip():
        cfg.source = source_override.strip()
    cfg.output.show = False

    pipeline = Pipeline(cfg)

    col_video, col_metrics = st.columns([2, 1])
    frame_slot = col_video.empty()
    metric_slot = col_metrics.empty()
    chart_slot = col_metrics.empty()
    fps_slot = col_metrics.empty()

    history: list[dict[str, float | int]] = []

    for idx, frame, tracks, _speeds in pipeline.stream():
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(rgb, use_container_width=True)

        metric_slot.metric("Active tracks", len(tracks))
        fps_slot.metric("FPS (EMA)", f"{pipeline._fps_ema:.1f}")  # noqa: SLF001

        row: dict[str, float | int] = {"frame": idx, "tracks": len(tracks)}
        for c in pipeline.line_counters:
            row[f"{c.config.name}_in"] = c.in_count
            row[f"{c.config.name}_out"] = c.out_count
        history.append(row)

        if idx % 5 == 0:
            df = pd.DataFrame(history)
            fig = px.line(df, x="frame", y=[c for c in df.columns if c != "frame"],
                          title="Counts over time", height=320)
            chart_slot.plotly_chart(fig, use_container_width=True)

    st.success(f"Finished. Processed {idx} frames.")  # noqa: F821


if __name__ == "__main__":
    main()
