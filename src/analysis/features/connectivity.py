"""Connectivity features for multichannel EEG.

Computes coherence, correlation, and other connectivity metrics
between EEG channels.
"""

from typing import Any

import numpy as np
from scipy import signal


def compute_coherence(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute coherence between all channel pairs.

    Coherence measures the linear relationship between two signals
    at each frequency.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency in Hz.
        nperseg: Segment length for coherence calculation.
        freq_bands: Frequency bands to average coherence over.

    Returns:
        Dictionary with coherence matrices for each frequency band.
    """
    if data.ndim == 1:
        return {}

    n_channels = data.shape[0]

    if nperseg is None:
        nperseg = min(256, data.shape[1])

    if freq_bands is None:
        freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50),
        }

    # Initialize coherence matrices
    coherence_matrices = {
        band: np.zeros((n_channels, n_channels))
        for band in freq_bands
    }

    # Compute coherence for each pair
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh = signal.coherence(
                data[i], data[j],
                fs=fs, nperseg=nperseg
            )

            # Average coherence in each band
            for band, (low, high) in freq_bands.items():
                idx = (freqs >= low) & (freqs <= high)
                if np.any(idx):
                    band_coh = np.mean(coh[idx])
                    coherence_matrices[band][i, j] = band_coh
                    coherence_matrices[band][j, i] = band_coh

        # Diagonal is 1
        for band in freq_bands:
            coherence_matrices[band][i, i] = 1.0

    return coherence_matrices


def compute_correlation(
    data: np.ndarray,
    method: str = "pearson",
) -> np.ndarray:
    """Compute correlation matrix between channels.

    Args:
        data: EEG data (channels, samples).
        method: Correlation method ("pearson" or "spearman").

    Returns:
        Correlation matrix (channels, channels).
    """
    if data.ndim == 1:
        return np.array([[1.0]])

    if method == "pearson":
        return np.corrcoef(data)
    elif method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(data, axis=1)
        return corr
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def compute_phase_locking_value(
    data: np.ndarray,
    fs: float,
    freq_band: tuple[float, float] = (8, 13),
) -> np.ndarray:
    """Compute phase locking value (PLV) between channels.

    PLV measures phase synchronization between signals.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency in Hz.
        freq_band: Frequency band for phase extraction.

    Returns:
        PLV matrix (channels, channels).
    """
    if data.ndim == 1:
        return np.array([[1.0]])

    n_channels = data.shape[0]

    # Bandpass filter to frequency band
    from .spectral import FREQUENCY_BANDS
    low, high = freq_band

    # Filter design
    nyq = fs / 2
    b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")

    # Filter and extract phase
    phases = []
    for ch in data:
        filtered = signal.filtfilt(b, a, ch)
        analytic = signal.hilbert(filtered)
        phase = np.angle(analytic)
        phases.append(phase)

    phases = np.array(phases)

    # Compute PLV matrix
    plv_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

        plv_matrix[i, i] = 1.0

    return plv_matrix


class ConnectivityFeatures:
    """Connectivity feature extractor for multichannel EEG.

    Computes coherence, correlation, and other connectivity metrics.

    Example:
        >>> cf = ConnectivityFeatures(fs=256)
        >>> features = cf.extract(eeg_data)
        >>> print(features["alpha_coherence_mean"])
    """

    def __init__(
        self,
        fs: float,
        channel_names: list[str] | None = None,
    ):
        """Initialize connectivity feature extractor.

        Args:
            fs: Sampling frequency in Hz.
            channel_names: Channel names for labeling.
        """
        self.fs = fs
        self.channel_names = channel_names

    def extract(
        self,
        data: np.ndarray,
        include_plv: bool = False,
    ) -> dict[str, Any]:
        """Extract all connectivity features.

        Args:
            data: EEG data (channels, samples).
            include_plv: Include phase locking value (slower).

        Returns:
            Dictionary of features.
        """
        if data.ndim == 1:
            return {}

        features = {}

        # Coherence matrices
        coh_matrices = compute_coherence(data, self.fs)

        for band, matrix in coh_matrices.items():
            # Mean coherence (excluding diagonal)
            n = matrix.shape[0]
            upper_tri = matrix[np.triu_indices(n, k=1)]
            features[f"{band}_coherence_mean"] = np.mean(upper_tri)
            features[f"{band}_coherence_std"] = np.std(upper_tri)
            features[f"{band}_coherence_matrix"] = matrix

        # Correlation
        corr_matrix = compute_correlation(data)
        n = corr_matrix.shape[0]
        upper_tri = corr_matrix[np.triu_indices(n, k=1)]
        features["correlation_mean"] = np.mean(upper_tri)
        features["correlation_std"] = np.std(upper_tri)
        features["correlation_matrix"] = corr_matrix

        # Phase locking value (optional, slower)
        if include_plv:
            for band_name, freq_band in [("alpha", (8, 13)), ("theta", (4, 8))]:
                plv = compute_phase_locking_value(data, self.fs, freq_band)
                upper_tri = plv[np.triu_indices(n, k=1)]
                features[f"{band_name}_plv_mean"] = np.mean(upper_tri)
                features[f"{band_name}_plv_matrix"] = plv

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names.
        """
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        names = []

        for band in bands:
            names.extend([
                f"{band}_coherence_mean",
                f"{band}_coherence_std",
            ])

        names.extend([
            "correlation_mean",
            "correlation_std",
        ])

        return names
