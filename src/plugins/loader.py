"""Dynamic plugin loader for the Consciousness Research Workbench.

Provides automatic plugin discovery, loading, dependency management,
and hot-reload support for development.
"""

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging
import json

from .base import Plugin, PluginMetadata


logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a loaded plugin.

    Attributes:
        metadata: Plugin metadata.
        module: Python module object.
        instance: Plugin instance.
        path: Path to plugin file.
        dependencies: List of dependencies.
        is_loaded: Whether plugin is currently loaded.
    """

    metadata: PluginMetadata
    module: Any
    instance: Plugin | None
    path: Path
    dependencies: list[str] = field(default_factory=list)
    is_loaded: bool = False


class PluginLoader:
    """Dynamic plugin loader with hot-reload support.

    Discovers, loads, and manages plugins from specified directories.

    Example:
        >>> loader = PluginLoader()
        >>> loader.add_plugin_directory("./my_plugins")
        >>> loader.discover_plugins()
        >>> loader.load_plugin("MyPlugin")
    """

    def __init__(self, auto_discover: bool = True):
        """Initialize plugin loader.

        Args:
            auto_discover: Whether to auto-discover plugins on init.
        """
        self.plugin_dirs: list[Path] = []
        self.plugins: dict[str, PluginInfo] = {}
        self.load_order: list[str] = []
        self._watchers: dict[Path, Any] = {}

        # Add default plugin directories
        default_dirs = [
            Path(__file__).parent / "examples",
            Path.cwd() / "plugins",
        ]

        for dir_path in default_dirs:
            if dir_path.exists():
                self.add_plugin_directory(dir_path)

        if auto_discover:
            self.discover_plugins()

    def add_plugin_directory(self, path: str | Path) -> None:
        """Add a directory to search for plugins.

        Args:
            path: Directory path.
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            if path not in self.plugin_dirs:
                self.plugin_dirs.append(path)
                logger.info(f"Added plugin directory: {path}")
        else:
            logger.warning(f"Plugin directory not found: {path}")

    def discover_plugins(self) -> list[str]:
        """Discover all plugins in registered directories.

        Returns:
            List of discovered plugin names.
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            for file_path in plugin_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue

                try:
                    plugin_name = self._load_plugin_info(file_path)
                    if plugin_name:
                        discovered.append(plugin_name)
                except Exception as e:
                    logger.error(f"Error discovering plugin {file_path}: {e}")

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def _load_plugin_info(self, file_path: Path) -> str | None:
        """Load plugin information without instantiating.

        Args:
            file_path: Path to plugin file.

        Returns:
            Plugin name if successful, None otherwise.
        """
        # Load module spec
        module_name = f"plugin_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            return None

        # Load module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find plugin class
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
                and hasattr(attr, 'metadata')
            ):
                plugin_class = attr
                break

        if plugin_class is None:
            return None

        # Create temporary instance to get metadata
        try:
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            dependencies = getattr(temp_instance, 'dependencies', [])
        except Exception as e:
            logger.error(f"Error getting plugin metadata: {e}")
            return None

        # Store plugin info
        self.plugins[metadata.name] = PluginInfo(
            metadata=metadata,
            module=module,
            instance=None,
            path=file_path,
            dependencies=dependencies,
            is_loaded=False,
        )

        return metadata.name

    def load_plugin(
        self,
        name: str,
        config: dict[str, Any] | None = None,
    ) -> Plugin | None:
        """Load and initialize a plugin.

        Args:
            name: Plugin name.
            config: Plugin configuration.

        Returns:
            Plugin instance if successful.
        """
        if name not in self.plugins:
            logger.error(f"Plugin not found: {name}")
            return None

        plugin_info = self.plugins[name]

        # Check dependencies
        for dep in plugin_info.dependencies:
            if dep not in self.plugins:
                logger.error(f"Missing dependency: {dep}")
                return None
            if not self.plugins[dep].is_loaded:
                self.load_plugin(dep)

        # Find and instantiate plugin class
        plugin_class = None
        for attr_name in dir(plugin_info.module):
            attr = getattr(plugin_info.module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
            ):
                plugin_class = attr
                break

        if plugin_class is None:
            logger.error(f"No plugin class found in {name}")
            return None

        try:
            instance = plugin_class()
            instance.initialize(config or {})
            plugin_info.instance = instance
            plugin_info.is_loaded = True

            if name not in self.load_order:
                self.load_order.append(name)

            logger.info(f"Loaded plugin: {name} v{plugin_info.metadata.version}")
            return instance

        except Exception as e:
            logger.error(f"Error loading plugin {name}: {e}")
            return None

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if successful.
        """
        if name not in self.plugins:
            return False

        plugin_info = self.plugins[name]

        if plugin_info.instance:
            try:
                plugin_info.instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {name}: {e}")

        plugin_info.instance = None
        plugin_info.is_loaded = False

        if name in self.load_order:
            self.load_order.remove(name)

        logger.info(f"Unloaded plugin: {name}")
        return True

    def reload_plugin(self, name: str) -> Plugin | None:
        """Hot-reload a plugin.

        Args:
            name: Plugin name.

        Returns:
            New plugin instance.
        """
        if name not in self.plugins:
            return None

        plugin_info = self.plugins[name]
        config = {}

        # Get current config if loaded
        if plugin_info.instance:
            config = getattr(plugin_info.instance, '_config', {})
            self.unload_plugin(name)

        # Reload module
        file_path = plugin_info.path
        module_name = f"plugin_{file_path.stem}"

        # Remove from sys.modules to force reload
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Rediscover and reload
        self._load_plugin_info(file_path)
        return self.load_plugin(name, config)

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a loaded plugin instance.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance if loaded.
        """
        if name in self.plugins and self.plugins[name].is_loaded:
            return self.plugins[name].instance
        return None

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all discovered plugins.

        Returns:
            List of plugin information dictionaries.
        """
        return [
            {
                "name": info.metadata.name,
                "version": info.metadata.version,
                "author": info.metadata.author,
                "description": info.metadata.description,
                "type": info.metadata.plugin_type,
                "is_loaded": info.is_loaded,
                "dependencies": info.dependencies,
            }
            for info in self.plugins.values()
        ]

    def check_compatibility(
        self,
        plugin_name: str,
        workbench_version: str = "1.0.0",
    ) -> bool:
        """Check if a plugin is compatible with the workbench version.

        Args:
            plugin_name: Plugin name.
            workbench_version: Workbench version.

        Returns:
            True if compatible.
        """
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]
        min_version = getattr(
            plugin_info.metadata, 'min_workbench_version', '0.0.0'
        )
        max_version = getattr(
            plugin_info.metadata, 'max_workbench_version', '99.99.99'
        )

        return self._version_in_range(
            workbench_version, min_version, max_version
        )

    def _version_in_range(
        self,
        version: str,
        min_version: str,
        max_version: str,
    ) -> bool:
        """Check if version is in range.

        Args:
            version: Version to check.
            min_version: Minimum version.
            max_version: Maximum version.

        Returns:
            True if in range.
        """
        def parse_version(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split('.'))

        v = parse_version(version)
        min_v = parse_version(min_version)
        max_v = parse_version(max_version)

        return min_v <= v <= max_v

    def load_all(self, config: dict[str, dict] | None = None) -> int:
        """Load all discovered plugins.

        Args:
            config: Plugin configurations keyed by name.

        Returns:
            Number of successfully loaded plugins.
        """
        config = config or {}
        loaded = 0

        for name in self.plugins:
            plugin_config = config.get(name, {})
            if self.load_plugin(name, plugin_config):
                loaded += 1

        return loaded

    def unload_all(self) -> None:
        """Unload all plugins."""
        for name in list(reversed(self.load_order)):
            self.unload_plugin(name)


# Global loader instance
_global_loader: PluginLoader | None = None


def get_loader() -> PluginLoader:
    """Get the global plugin loader.

    Returns:
        Global PluginLoader instance.
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = PluginLoader()
    return _global_loader


def load_plugin(name: str, config: dict | None = None) -> Plugin | None:
    """Load a plugin using the global loader.

    Args:
        name: Plugin name.
        config: Plugin configuration.

    Returns:
        Plugin instance.
    """
    return get_loader().load_plugin(name, config)
