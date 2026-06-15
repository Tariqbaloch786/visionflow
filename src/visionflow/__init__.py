"""VisionFlow: real-time multi-object tracking and traffic analytics."""
from visionflow.analytics import HeatmapAccumulator, LineCounter, SpeedEstimator
from visionflow.config import PipelineConfig
from visionflow.detector import Detection, Detector
from visionflow.pipeline import Pipeline
from visionflow.tracker import IoUTracker, Track

__all__ = [
    "Detection",
    "Detector",
    "HeatmapAccumulator",
    "IoUTracker",
    "LineCounter",
    "Pipeline",
    "PipelineConfig",
    "SpeedEstimator",
    "Track",
]

__version__ = "0.2.0"
