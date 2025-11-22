"""Consciousness Research Workbench.

A unified platform for exploring AI consciousness through grounded human experience.
Bridges embodied cognition (EEG data) with computational consciousness models.

Modules:
    foundation: Phase 1 data collection (EEG, GPS, Audio)
    analysis: Signal processing and feature extraction
    consciousness: AI consciousness analysis modules
    integrations: Dashboard, reports, API
    plugins: Extensibility layer
"""

__version__ = "0.2.0"
__project__ = "Consciousness Research Workbench"

# Backward compatibility - import from foundation
from .foundation.config.settings import Config, load_config, get_config
from .foundation.eeg.connection import EEGConnection, test_connection
from .foundation.data.logger import DataLogger, Session
from .foundation.scenarios.runner import ScenarioRunner

__all__ = [
    # Core classes
    "Config",
    "load_config",
    "get_config",
    "EEGConnection",
    "test_connection",
    "DataLogger",
    "Session",
    "ScenarioRunner",
]
