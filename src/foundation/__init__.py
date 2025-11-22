"""Foundation layer - Phase 1 data collection components.

This module contains the core data collection infrastructure:
- EEG device handling (Muse 2/S via BrainFlow)
- Multimodal sensors (GPS, Audio)
- Data storage (HDF5, Parquet)
- Scenario execution framework
- Configuration management
"""

from .config.settings import Config, load_config, get_config
from .eeg.connection import EEGConnection, test_connection
from .data.logger import DataLogger, Session
from .scenarios.runner import ScenarioRunner

__all__ = [
    "Config",
    "load_config",
    "get_config",
    "EEGConnection",
    "test_connection",
    "DataLogger",
    "Session",
    "ScenarioRunner",
]
