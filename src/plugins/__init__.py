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
    Plugin,
    PluginMetadata,
    DataCollectorPlugin,
    AnalysisPlugin,
    ConsciousnessPlugin,
    VisualizationPlugin,
    ExternalServicePlugin,
)
from .registry import PluginRegistry, get_registry
from .loader import PluginLoader, get_loader, load_plugin
from .hooks import (
    HookType,
    HookManager,
    HookResult,
    get_hook_manager,
    hook,
    invoke_hook,
)
from .marketplace import PluginMarketplace, get_marketplace

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BasePlugin",
    "Plugin",
    "PluginMetadata",
    "DataCollectorPlugin",
    "AnalysisPlugin",
    "ConsciousnessPlugin",
    "VisualizationPlugin",
    "ExternalServicePlugin",
    # Registry
    "PluginRegistry",
    "get_registry",
    # Loader
    "PluginLoader",
    "get_loader",
    "load_plugin",
    # Hooks
    "HookType",
    "HookManager",
    "HookResult",
    "get_hook_manager",
    "hook",
    "invoke_hook",
    # Marketplace
    "PluginMarketplace",
    "get_marketplace",
]
