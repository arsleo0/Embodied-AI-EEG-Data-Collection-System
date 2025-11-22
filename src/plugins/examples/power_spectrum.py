"""Example analysis plugin for power spectrum density calculation."""

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..base.plugin import AnalysisPlugin, PluginMetadata


class PowerSpectrumPlugin(AnalysisPlugin):
    """Example plugin for computing power spectral density.

    This demonstrates how to create an analysis plugin that:
    - Processes EEG data
    - Extracts frequency-domain features
    - Provides visualization

    Usage:
        from src.plugins import get_registry

        registry = get_registry()
        registry.register(PowerSpectrumPlugin)

        plugin = registry.get_plugin("PowerSpectrumPlugin", "analysis")
        results = plugin.process(eeg_data)
    """

    def __init__(self):
        """Initialize the power spectrum plugin."""
        self._config: dict[str, Any] = {}
        self._fs = 256  # Default sampling rate

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="PowerSpectrumPlugin",
            version="1.0.0",
            author="Consciousness Research Workbench",
            description="Computes power spectral density for EEG signals",
            plugin_type="analysis",
            dependencies=["scipy"],
            config_schema={
                "sampling_rate": {"type": "int", "default": 256},
                "nperseg": {"type": "int", "default": 256},
                "bands": {
                    "type": "dict",
                    "default": {
                        "delta": (0.5, 4),
                        "theta": (4, 8),
                        "alpha": (8, 13),
                        "beta": (13, 30),
                        "gamma": (30, 50),
                    },
                },
            },
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for PowerSpectrumPlugin")

        self._config = config
        self._fs = config.get("sampling_rate", 256)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._config = {}

    def process(self, data: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """Compute power spectral density.

        Args:
            data: EEG data (channels x samples) or DataFrame.

        Returns:
            Dictionary with:
                - frequencies: Frequency array
                - psd: Power spectral density (channels x frequencies)
                - band_powers: Power in each frequency band
        """
        # Convert DataFrame to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values.T  # Transpose to (channels, samples)

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Compute PSD for each channel
        nperseg = self._config.get("nperseg", min(256, data.shape[1]))
        frequencies, psd = signal.welch(data, fs=self._fs, nperseg=nperseg)

        # Compute band powers
        bands = self._config.get("bands", {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50),
        })

        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(frequencies >= low, frequencies <= high)
            band_powers[band_name] = np.mean(psd[:, idx], axis=1)

        return {
            "frequencies": frequencies,
            "psd": psd,
            "band_powers": band_powers,
            "sampling_rate": self._fs,
            "num_channels": data.shape[0],
        }

    def visualize(self, results: dict[str, Any]) -> Any:
        """Create power spectrum visualization.

        Args:
            results: Results from process().

        Returns:
            Matplotlib figure or Plotly figure.
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Plot PSD
            frequencies = results["frequencies"]
            psd = results["psd"]

            for i in range(psd.shape[0]):
                axes[0].semilogy(frequencies, psd[i], label=f"Channel {i+1}")

            axes[0].set_xlabel("Frequency (Hz)")
            axes[0].set_ylabel("Power Spectral Density")
            axes[0].set_title("Power Spectrum")
            axes[0].legend()
            axes[0].grid(True)

            # Plot band powers
            band_powers = results["band_powers"]
            bands = list(band_powers.keys())
            x = np.arange(len(bands))
            width = 0.8 / psd.shape[0]

            for i in range(psd.shape[0]):
                powers = [band_powers[band][i] for band in bands]
                axes[1].bar(x + i * width, powers, width, label=f"Channel {i+1}")

            axes[1].set_xlabel("Frequency Band")
            axes[1].set_ylabel("Mean Power")
            axes[1].set_title("Band Powers")
            axes[1].set_xticks(x + width * (psd.shape[0] - 1) / 2)
            axes[1].set_xticklabels(bands)
            axes[1].legend()

            plt.tight_layout()
            return fig

        except ImportError:
            return None

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features."""
        bands = self._config.get("bands", {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50),
        })
        return list(bands.keys())
