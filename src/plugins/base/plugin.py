"""Base plugin abstract classes.

These define the interfaces that all plugins must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    dependencies: list[str] | None = None
    config_schema: dict[str, Any] | None = None


class BasePlugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata.

        Returns:
            PluginMetadata instance.
        """
        pass

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin configuration dictionary.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate plugin configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if valid.
        """
        return True


class DataCollectorPlugin(BasePlugin):
    """Base class for data collection plugins.

    Use this for adding new sensor types or data sources.

    Example:
        class EyeTrackerPlugin(DataCollectorPlugin):
            def start_stream(self):
                # Connect to eye tracker
                pass

            def get_data(self) -> pd.DataFrame:
                # Return gaze data
                pass
    """

    @abstractmethod
    def start_stream(self) -> None:
        """Start data streaming from the source."""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop data streaming."""
        pass

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Get collected data.

        Returns:
            DataFrame with collected data and timestamps.
        """
        pass

    @abstractmethod
    def get_sampling_rate(self) -> float:
        """Get data sampling rate.

        Returns:
            Sampling rate in Hz.
        """
        pass

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return False


class AnalysisPlugin(BasePlugin):
    """Base class for analysis plugins.

    Use this for signal processing, feature extraction, or classification.

    Example:
        class WaveletAnalysisPlugin(AnalysisPlugin):
            def process(self, data):
                # Apply wavelet transform
                return wavelet_coefficients

            def visualize(self, results):
                # Create scalogram
                return figure
    """

    @abstractmethod
    def process(self, data: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """Process input data.

        Args:
            data: Input data to process.

        Returns:
            Processing results dictionary.
        """
        pass

    def visualize(self, results: dict[str, Any]) -> Any:
        """Visualize processing results.

        Args:
            results: Results from process().

        Returns:
            Visualization object (Figure, HTML, etc.).
        """
        return None

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features.

        Returns:
            List of feature names.
        """
        return []


class ConsciousnessPlugin(BasePlugin):
    """Base class for consciousness analysis plugins.

    Use this for AI introspection, latent space mapping, or awareness detection.

    Example:
        class AttentionEntropyPlugin(ConsciousnessPlugin):
            def get_internal_state(self):
                # Get attention matrices
                return attention_state

            def introspect(self, query):
                # Analyze attention patterns
                return report
    """

    @abstractmethod
    def get_internal_state(self) -> dict[str, np.ndarray]:
        """Get internal state representation.

        Returns:
            Dictionary of state vectors/matrices.
        """
        pass

    @abstractmethod
    def introspect(self, query: str) -> dict[str, Any]:
        """Perform introspective analysis.

        Args:
            query: Analysis query or focus area.

        Returns:
            Introspection report.
        """
        pass

    def compare_states(
        self,
        state1: dict[str, np.ndarray],
        state2: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compare two internal states.

        Args:
            state1: First state.
            state2: Second state.

        Returns:
            Comparison metrics.
        """
        return {}


class VisualizationPlugin(BasePlugin):
    """Base class for visualization plugins.

    Use this for custom visualizations, dashboards, or interactive displays.

    Example:
        class LatentSpace3DPlugin(VisualizationPlugin):
            def render(self, data):
                # Create 3D scatter plot
                return plotly_figure
    """

    @abstractmethod
    def render(
        self,
        data: pd.DataFrame | np.ndarray | dict[str, Any],
        **kwargs,
    ) -> Any:
        """Render visualization.

        Args:
            data: Data to visualize.
            **kwargs: Additional rendering options.

        Returns:
            Visualization object (Figure, HTML component, etc.).
        """
        pass

    def get_supported_formats(self) -> list[str]:
        """Get supported output formats.

        Returns:
            List of format names (e.g., ['html', 'png', 'svg']).
        """
        return ["html"]

    def export(self, visualization: Any, format: str, path: str) -> None:
        """Export visualization to file.

        Args:
            visualization: Visualization to export.
            format: Output format.
            path: Output file path.
        """
        pass


class ExternalServicePlugin(BasePlugin):
    """Base class for external service integrations.

    Use this for cloud services, APIs, or third-party tools.

    Example:
        class MLflowPlugin(ExternalServicePlugin):
            def push_data(self, experiment):
                # Log to MLflow
                pass

            def pull_insights(self):
                # Get model comparisons
                return insights
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to external service.

        Returns:
            True if connected successfully.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from service."""
        pass

    @abstractmethod
    def push_data(self, data: dict[str, Any]) -> bool:
        """Push data to external service.

        Args:
            data: Data to push.

        Returns:
            True if successful.
        """
        pass

    @abstractmethod
    def pull_insights(self) -> dict[str, Any]:
        """Pull insights from external service.

        Returns:
            Insights dictionary.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected to service."""
        return False
