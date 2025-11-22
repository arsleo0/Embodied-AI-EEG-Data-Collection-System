"""Configuration loader and settings management."""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager for the EEG collection system."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to config.yaml. If None, searches default locations.
        """
        self._config_path = self._find_config(config_path)
        self._data = self._load_config()

    def _find_config(self, config_path: str | Path | None) -> Path:
        """Find configuration file in default locations.

        Args:
            config_path: Explicit path or None for auto-discovery.

        Returns:
            Path to configuration file.

        Raises:
            FileNotFoundError: If no config file found.
        """
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Search default locations
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd().parent / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Config file not found in: {[str(p) for p in search_paths]}"
        )

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary.
        """
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Dot-notation key (e.g., "eeg.device_type").
            default: Default value if key not found.

        Returns:
            Configuration value or default.

        Example:
            >>> config.get("eeg.brainflow.board_id")
            22
        """
        keys = key.split(".")
        value = self._data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot-notation key.

        Args:
            key: Dot-notation key.
            value: Value to set.
        """
        keys = key.split(".")
        data = self._data

        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        data[keys[-1]] = value

    @property
    def eeg(self) -> dict[str, Any]:
        """Get EEG configuration section."""
        return self._data.get("eeg", {})

    @property
    def storage(self) -> dict[str, Any]:
        """Get storage configuration section."""
        return self._data.get("storage", {})

    @property
    def sensors(self) -> dict[str, Any]:
        """Get sensors configuration section."""
        return self._data.get("sensors", {})

    @property
    def scenarios(self) -> dict[str, Any]:
        """Get scenarios configuration section."""
        return self._data.get("scenarios", {})

    @property
    def logging(self) -> dict[str, Any]:
        """Get logging configuration section."""
        return self._data.get("logging", {})

    def get_board_id(self) -> int:
        """Get BrainFlow board ID for current device.

        Returns:
            BrainFlow board ID.
        """
        device_type = self.get("eeg.device_type", "synthetic")

        # Map device types to BrainFlow board IDs
        board_ids = {
            "muse_2": 22,
            "muse_s": 21,
            "muse_2_bled": 38,  # Muse 2 with BLED dongle
            "muse_s_bled": 39,  # Muse S with BLED dongle
            "openbci_cyton": 0,
            "openbci_ganglion": 1,
            "synthetic": -1,  # Synthetic board for testing
        }

        return board_ids.get(device_type, -1)

    def get_storage_path(self, category: str = "raw") -> Path:
        """Get storage path for data category.

        Args:
            category: Storage category (raw, processed, exports).

        Returns:
            Path to storage directory.
        """
        base_path = Path(self.get("storage.base_path", "data"))
        category_path = self.get(f"storage.{category}.path", f"data/{category}")

        path = Path(category_path)
        path.mkdir(parents=True, exist_ok=True)

        return path

    def __repr__(self) -> str:
        return f"Config(path={self._config_path})"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from file.

    Args:
        config_path: Path to config file or None for auto-discovery.

    Returns:
        Config instance.
    """
    return Config(config_path)


# Global config instance (lazy loaded)
_global_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global Config instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config
