"""Plugin registry implementation."""

from typing import Any, Type

from ..base.plugin import (
    BasePlugin,
    DataCollectorPlugin,
    AnalysisPlugin,
    ConsciousnessPlugin,
    VisualizationPlugin,
    ExternalServicePlugin,
)


class PluginRegistry:
    """Registry for managing and discovering plugins."""

    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: dict[str, dict[str, Type[BasePlugin]]] = {
            "data_collector": {},
            "analysis": {},
            "consciousness": {},
            "visualization": {},
            "external_service": {},
        }
        self._instances: dict[str, BasePlugin] = {}

    def register(self, plugin_class: Type[BasePlugin]) -> None:
        """Register a plugin class.

        Args:
            plugin_class: Plugin class to register.

        Raises:
            ValueError: If plugin type is unknown.
        """
        # Create temporary instance to get metadata
        temp_instance = plugin_class.__new__(plugin_class)

        # Determine plugin type
        if issubclass(plugin_class, DataCollectorPlugin):
            plugin_type = "data_collector"
        elif issubclass(plugin_class, AnalysisPlugin):
            plugin_type = "analysis"
        elif issubclass(plugin_class, ConsciousnessPlugin):
            plugin_type = "consciousness"
        elif issubclass(plugin_class, VisualizationPlugin):
            plugin_type = "visualization"
        elif issubclass(plugin_class, ExternalServicePlugin):
            plugin_type = "external_service"
        else:
            raise ValueError(f"Unknown plugin type: {plugin_class}")

        # Get name from class
        name = plugin_class.__name__

        self._plugins[plugin_type][name] = plugin_class

    def unregister(self, name: str, plugin_type: str) -> None:
        """Unregister a plugin.

        Args:
            name: Plugin name.
            plugin_type: Plugin type category.
        """
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]

        # Also remove any instance
        instance_key = f"{plugin_type}:{name}"
        if instance_key in self._instances:
            self._instances[instance_key].cleanup()
            del self._instances[instance_key]

    def get_plugin(
        self,
        name: str,
        plugin_type: str,
        config: dict[str, Any] | None = None,
    ) -> BasePlugin:
        """Get or create a plugin instance.

        Args:
            name: Plugin name.
            plugin_type: Plugin type category.
            config: Plugin configuration.

        Returns:
            Plugin instance.

        Raises:
            KeyError: If plugin not found.
        """
        if plugin_type not in self._plugins:
            raise KeyError(f"Unknown plugin type: {plugin_type}")

        if name not in self._plugins[plugin_type]:
            raise KeyError(f"Plugin not found: {name} ({plugin_type})")

        instance_key = f"{plugin_type}:{name}"

        # Return existing instance if available
        if instance_key in self._instances:
            return self._instances[instance_key]

        # Create new instance
        plugin_class = self._plugins[plugin_type][name]
        instance = plugin_class()
        instance.initialize(config or {})

        self._instances[instance_key] = instance
        return instance

    def list_plugins(self, plugin_type: str | None = None) -> dict[str, list[str]]:
        """List registered plugins.

        Args:
            plugin_type: Filter by type. None for all types.

        Returns:
            Dictionary mapping plugin types to lists of names.
        """
        if plugin_type:
            return {plugin_type: list(self._plugins.get(plugin_type, {}).keys())}

        return {
            ptype: list(plugins.keys())
            for ptype, plugins in self._plugins.items()
            if plugins
        }

    def get_plugin_info(self, name: str, plugin_type: str) -> dict[str, Any]:
        """Get plugin metadata.

        Args:
            name: Plugin name.
            plugin_type: Plugin type.

        Returns:
            Plugin metadata dictionary.
        """
        if plugin_type not in self._plugins:
            raise KeyError(f"Unknown plugin type: {plugin_type}")

        if name not in self._plugins[plugin_type]:
            raise KeyError(f"Plugin not found: {name}")

        plugin_class = self._plugins[plugin_type][name]

        return {
            "name": name,
            "type": plugin_type,
            "class": plugin_class.__name__,
            "module": plugin_class.__module__,
        }

    def cleanup_all(self) -> None:
        """Clean up all plugin instances."""
        for instance in self._instances.values():
            try:
                instance.cleanup()
            except Exception:
                pass
        self._instances.clear()


# Global registry instance
_global_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """Get global plugin registry.

    Returns:
        Global PluginRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def register_plugin(plugin_class: Type[BasePlugin]) -> Type[BasePlugin]:
    """Decorator to register a plugin class.

    Example:
        @register_plugin
        class MyAnalysisPlugin(AnalysisPlugin):
            ...

    Args:
        plugin_class: Plugin class to register.

    Returns:
        The same plugin class (for decorator use).
    """
    get_registry().register(plugin_class)
    return plugin_class
