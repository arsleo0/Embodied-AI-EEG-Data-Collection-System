"""Statistical and time-domain features for EEG signals.

Provides statistical features and Hjorth parameters.
"""

from typing import Any

import numpy as np
from scipy import stats


def compute_statistical_features(
    data: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute basic statistical features.

    Args:
        data: Signal array (samples,) or (channels, samples).

    Returns:
        Dictionary of statistical features.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    features = {}

    # Basic statistics
    features["mean"] = np.mean(data, axis=1)
    features["std"] = np.std(data, axis=1)
    features["var"] = np.var(data, axis=1)
    features["min"] = np.min(data, axis=1)
    features["max"] = np.max(data, axis=1)
    features["ptp"] = np.ptp(data, axis=1)  # Peak-to-peak

    # Higher-order statistics
    features["skewness"] = stats.skew(data, axis=1)
    features["kurtosis"] = stats.kurtosis(data, axis=1)

    # Percentiles
    features["median"] = np.median(data, axis=1)
    features["q25"] = np.percentile(data, 25, axis=1)
    features["q75"] = np.percentile(data, 75, axis=1)
    features["iqr"] = features["q75"] - features["q25"]

    # Root mean square
    features["rms"] = np.sqrt(np.mean(data ** 2, axis=1))

    # Zero crossings
    features["zero_crossings"] = np.array([
        np.sum(np.diff(np.sign(ch)) != 0) for ch in data
    ])

    # Mean absolute value
    features["mav"] = np.mean(np.abs(data), axis=1)

    # Squeeze if single channel
    if data.shape[0] == 1:
        features = {k: v.squeeze() for k, v in features.items()}

    return features


def compute_hjorth_parameters(
    data: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Hjorth parameters.

    Hjorth parameters describe signal characteristics:
    - Activity: Signal power (variance)
    - Mobility: Mean frequency
    - Complexity: Change in frequency

    Args:
        data: Signal array (samples,) or (channels, samples).

    Returns:
        Dictionary with activity, mobility, complexity.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # First derivative
    d1 = np.diff(data, axis=1)

    # Second derivative
    d2 = np.diff(d1, axis=1)

    # Activity: variance of signal
    activity = np.var(data, axis=1)

    # Variance of derivatives
    var_d1 = np.var(d1, axis=1)
    var_d2 = np.var(d2, axis=1)

    # Mobility: sqrt(var(d1) / var(signal))
    mobility = np.sqrt(var_d1 / np.maximum(activity, 1e-10))

    # Complexity: mobility(d1) / mobility(signal)
    mobility_d1 = np.sqrt(var_d2 / np.maximum(var_d1, 1e-10))
    complexity = mobility_d1 / np.maximum(mobility, 1e-10)

    result = {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }

    # Squeeze if single channel
    if data.shape[0] == 1:
        result = {k: v.squeeze() for k, v in result.items()}

    return result


def compute_line_length(data: np.ndarray) -> np.ndarray:
    """Compute line length feature.

    Line length measures signal complexity as the sum of
    absolute differences between consecutive samples.

    Args:
        data: Signal array (samples,) or (channels, samples).

    Returns:
        Line length value(s).
    """
    if data.ndim == 1:
        return np.sum(np.abs(np.diff(data)))
    else:
        return np.sum(np.abs(np.diff(data, axis=1)), axis=1)


def compute_nonlinear_energy(data: np.ndarray) -> np.ndarray:
    """Compute nonlinear energy operator (Teager-Kaiser).

    Args:
        data: Signal array (samples,) or (channels, samples).

    Returns:
        Mean nonlinear energy value(s).
    """
    if data.ndim == 1:
        return np.mean(data[1:-1] ** 2 - data[:-2] * data[2:])
    else:
        return np.mean(
            data[:, 1:-1] ** 2 - data[:, :-2] * data[:, 2:],
            axis=1
        )


class StatisticalFeatures:
    """Statistical feature extractor for EEG signals.

    Computes time-domain and statistical features.

    Example:
        >>> sf = StatisticalFeatures()
        >>> features = sf.extract(eeg_data)
        >>> print(features["hjorth_mobility"])
    """

    def __init__(self):
        """Initialize statistical feature extractor."""
        pass

    def extract(
        self,
        data: np.ndarray,
        include_all: bool = True,
    ) -> dict[str, Any]:
        """Extract all statistical features.

        Args:
            data: EEG data (channels, samples).
            include_all: Include extended features.

        Returns:
            Dictionary of features.
        """
        features = {}

        # Basic statistics
        stats_features = compute_statistical_features(data)
        features.update(stats_features)

        # Hjorth parameters
        hjorth = compute_hjorth_parameters(data)
        features.update(hjorth)

        if include_all:
            # Line length
            features["line_length"] = compute_line_length(data)

            # Nonlinear energy
            features["nonlinear_energy"] = compute_nonlinear_energy(data)

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names.
        """
        return [
            "mean", "std", "var", "min", "max", "ptp",
            "skewness", "kurtosis",
            "median", "q25", "q75", "iqr",
            "rms", "zero_crossings", "mav",
            "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
            "line_length", "nonlinear_energy",
        ]
