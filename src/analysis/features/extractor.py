"""Main feature extractor combining all feature types.

Provides a unified interface for extracting all EEG features.
"""

from typing import Any

import numpy as np
import pandas as pd

from .spectral import SpectralFeatures
from .entropy import EntropyFeatures
from .statistical import StatisticalFeatures
from .connectivity import ConnectivityFeatures


class FeatureExtractor:
    """Comprehensive feature extractor for EEG data.

    Combines spectral, entropy, statistical, and connectivity
    features into a single extraction pipeline.

    Example:
        >>> extractor = FeatureExtractor(fs=256, channel_names=["TP9", "AF7", "AF8", "TP10"])
        >>> features = extractor.extract(eeg_data)
        >>> df = extractor.to_dataframe(features)
    """

    def __init__(
        self,
        fs: float,
        channel_names: list[str] | None = None,
        include_spectral: bool = True,
        include_entropy: bool = True,
        include_statistical: bool = True,
        include_connectivity: bool = True,
    ):
        """Initialize feature extractor.

        Args:
            fs: Sampling frequency in Hz.
            channel_names: Channel names for labeling.
            include_spectral: Include spectral features.
            include_entropy: Include entropy features.
            include_statistical: Include statistical features.
            include_connectivity: Include connectivity features.
        """
        self.fs = fs
        self.channel_names = channel_names

        # Feature type flags
        self._include_spectral = include_spectral
        self._include_entropy = include_entropy
        self._include_statistical = include_statistical
        self._include_connectivity = include_connectivity

        # Initialize extractors
        self._spectral = SpectralFeatures(fs) if include_spectral else None
        self._entropy = EntropyFeatures(fs) if include_entropy else None
        self._statistical = StatisticalFeatures() if include_statistical else None
        self._connectivity = ConnectivityFeatures(fs, channel_names) if include_connectivity else None

    def extract(
        self,
        data: np.ndarray,
        flatten: bool = False,
    ) -> dict[str, Any]:
        """Extract all features from EEG data.

        Args:
            data: EEG data (channels, samples).
            flatten: If True, flatten multichannel features.

        Returns:
            Dictionary of all features.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        features = {}

        # Spectral features
        if self._spectral:
            spectral = self._spectral.extract(data)
            for name, value in spectral.items():
                features[f"spectral_{name}"] = value

        # Entropy features
        if self._entropy:
            entropy = self._entropy.extract(data)
            for name, value in entropy.items():
                features[f"entropy_{name}"] = value

        # Statistical features
        if self._statistical:
            statistical = self._statistical.extract(data)
            for name, value in statistical.items():
                features[f"stat_{name}"] = value

        # Connectivity features
        if self._connectivity and data.shape[0] > 1:
            connectivity = self._connectivity.extract(data)
            for name, value in connectivity.items():
                # Skip matrices if flattening
                if flatten and isinstance(value, np.ndarray) and value.ndim > 1:
                    continue
                features[f"conn_{name}"] = value

        # Flatten if requested
        if flatten:
            features = self._flatten_features(features)

        return features

    def extract_epoch(
        self,
        data: np.ndarray,
        epoch_length: float,
        overlap: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Extract features from epochs (segments) of data.

        Args:
            data: EEG data (channels, samples).
            epoch_length: Length of each epoch in seconds.
            overlap: Overlap ratio between epochs (0-1).

        Returns:
            List of feature dictionaries, one per epoch.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        epoch_samples = int(epoch_length * self.fs)
        step_samples = int(epoch_samples * (1 - overlap))

        n_samples = data.shape[1]
        epochs_features = []

        for start in range(0, n_samples - epoch_samples + 1, step_samples):
            end = start + epoch_samples
            epoch_data = data[:, start:end]

            features = self.extract(epoch_data, flatten=True)
            features["epoch_start"] = start / self.fs
            features["epoch_end"] = end / self.fs

            epochs_features.append(features)

        return epochs_features

    def to_dataframe(
        self,
        features: dict[str, Any] | list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert features to pandas DataFrame.

        Args:
            features: Feature dictionary or list of dictionaries.

        Returns:
            DataFrame with features.
        """
        if isinstance(features, dict):
            features = [features]

        # Flatten any remaining arrays
        flat_features = []
        for feat_dict in features:
            flat = {}
            for name, value in feat_dict.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        flat[name] = float(value)
                    elif value.ndim == 1:
                        for i, v in enumerate(value):
                            ch_name = self.channel_names[i] if self.channel_names else f"ch{i}"
                            flat[f"{name}_{ch_name}"] = v
                    # Skip 2D arrays (matrices)
                else:
                    flat[name] = value
            flat_features.append(flat)

        return pd.DataFrame(flat_features)

    def _flatten_features(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Flatten multichannel features.

        Args:
            features: Feature dictionary.

        Returns:
            Flattened feature dictionary.
        """
        flat = {}

        for name, value in features.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    flat[name] = float(value)
                elif value.ndim == 1:
                    for i, v in enumerate(value):
                        ch_name = self.channel_names[i] if self.channel_names else f"ch{i}"
                        flat[f"{name}_{ch_name}"] = v
                # Skip 2D arrays
            else:
                flat[name] = value

        return flat

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names.

        Returns:
            List of feature names.
        """
        names = []

        if self._spectral:
            for name in self._spectral.get_feature_names():
                names.append(f"spectral_{name}")

        if self._entropy:
            for name in self._entropy.get_feature_names():
                names.append(f"entropy_{name}")

        if self._statistical:
            for name in self._statistical.get_feature_names():
                names.append(f"stat_{name}")

        if self._connectivity:
            for name in self._connectivity.get_feature_names():
                names.append(f"conn_{name}")

        return names

    def __repr__(self) -> str:
        modules = []
        if self._include_spectral:
            modules.append("spectral")
        if self._include_entropy:
            modules.append("entropy")
        if self._include_statistical:
            modules.append("statistical")
        if self._include_connectivity:
            modules.append("connectivity")

        return f"FeatureExtractor(fs={self.fs}, modules={modules})"
