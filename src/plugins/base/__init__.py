"""Base plugin classes and interfaces.

All plugins must inherit from one of these base classes to be compatible
with the plugin registry and workbench infrastructure.
"""

from .plugin import (
    BasePlugin,
    DataCollectorPlugin,
    AnalysisPlugin,
    ConsciousnessPlugin,
    VisualizationPlugin,
    ExternalServicePlugin,
)

__all__ = [
    "BasePlugin",
    "DataCollectorPlugin",
    "AnalysisPlugin",
    "ConsciousnessPlugin",
    "VisualizationPlugin",
    "ExternalServicePlugin",
]
