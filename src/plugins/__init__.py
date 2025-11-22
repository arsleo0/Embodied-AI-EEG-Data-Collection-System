"""Plugin system for extensibility.

This module provides the infrastructure for creating and managing plugins
that extend the Consciousness Research Workbench functionality.

Plugin Types:
    DataCollector: New data sources (sensors, devices)
    AnalysisModule: Data analysis and processing
    ConsciousnessModel: AI consciousness models
    Visualization: Custom visualizations
    ExternalService: Third-party integrations
"""

from .base import (
    BasePlugin,
    DataCollectorPlugin,
    AnalysisPlugin,
    ConsciousnessPlugin,
    VisualizationPlugin,
    ExternalServicePlugin,
)
from .registry import PluginRegistry, get_registry

__all__ = [
    "BasePlugin",
    "DataCollectorPlugin",
    "AnalysisPlugin",
    "ConsciousnessPlugin",
    "VisualizationPlugin",
    "ExternalServicePlugin",
    "PluginRegistry",
    "get_registry",
]
