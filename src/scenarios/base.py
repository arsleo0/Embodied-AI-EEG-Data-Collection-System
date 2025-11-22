"""Base scenario class and configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MarkerConfig:
    """Configuration for a scenario marker."""

    name: str
    key: str  # Keyboard shortcut
    description: str = ""


@dataclass
class ScenarioConfig:
    """Configuration for a scenario."""

    name: str
    description: str = ""
    duration: int | None = None  # None for manual stop
    markers: list[MarkerConfig] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        """Load scenario configuration from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            ScenarioConfig instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Parse markers
        markers = []
        for marker_data in data.get("markers", []):
            markers.append(MarkerConfig(
                name=marker_data["name"],
                key=marker_data["key"],
                description=marker_data.get("description", ""),
            ))

        return cls(
            name=data.get("name", Path(path).stem),
            description=data.get("description", ""),
            duration=data.get("duration"),
            markers=markers,
            instructions=data.get("instructions", []),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "duration": self.duration,
            "markers": [
                {"name": m.name, "key": m.key, "description": m.description}
                for m in self.markers
            ],
            "instructions": self.instructions,
            "tags": self.tags,
        }


class BaseScenario:
    """Base class for scenarios."""

    def __init__(self, config: ScenarioConfig):
        """Initialize scenario.

        Args:
            config: Scenario configuration.
        """
        self.config = config
        self._is_running = False

    @property
    def name(self) -> str:
        """Get scenario name."""
        return self.config.name

    @property
    def is_running(self) -> bool:
        """Check if scenario is running."""
        return self._is_running

    def on_start(self) -> None:
        """Called when scenario starts. Override in subclass."""
        pass

    def on_stop(self) -> None:
        """Called when scenario stops. Override in subclass."""
        pass

    def on_marker(self, marker_name: str) -> None:
        """Called when a marker is triggered. Override in subclass.

        Args:
            marker_name: Name of triggered marker.
        """
        pass

    def get_marker_for_key(self, key: str) -> MarkerConfig | None:
        """Get marker configuration for keyboard shortcut.

        Args:
            key: Keyboard key.

        Returns:
            MarkerConfig or None if no marker for key.
        """
        for marker in self.config.markers:
            if marker.key == key:
                return marker
        return None

    def get_instructions(self) -> list[str]:
        """Get scenario instructions.

        Returns:
            List of instruction strings.
        """
        return self.config.instructions

    def start(self) -> None:
        """Start the scenario."""
        self._is_running = True
        self.on_start()

    def stop(self) -> None:
        """Stop the scenario."""
        self._is_running = False
        self.on_stop()
