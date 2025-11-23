"""Real-time dashboard for consciousness research.

Provides interactive visualization and monitoring of EEG data
and consciousness analysis results.
"""

from .app import create_dashboard, run_dashboard
from .components import (
    EEGSignalPlot,
    ConsciousnessStateDisplay,
    SignalQualityMeter,
    FeatureVisualization,
    MarkerTimeline,
)
from .live_monitor import (
    LiveMonitor,
    StreamingHandler,
    SignalQualityAlert,
)

__all__ = [
    # App
    "create_dashboard",
    "run_dashboard",
    # Components
    "EEGSignalPlot",
    "ConsciousnessStateDisplay",
    "SignalQualityMeter",
    "FeatureVisualization",
    "MarkerTimeline",
    # Live Monitor
    "LiveMonitor",
    "StreamingHandler",
    "SignalQualityAlert",
]
