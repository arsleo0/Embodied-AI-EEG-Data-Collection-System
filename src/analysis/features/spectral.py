"""Spectral feature extraction for EEG signals.

Provides power spectral density computation and band power analysis.
"""

from typing import Any

import numpy as np
from scipy import signal


# Standard EEG frequency bands
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def compute_psd(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    method: str = "welch",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density.

    Args:
        data: Signal array (samples,) or (channels, samples).
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment for Welch method.
        noverlap: Number of overlapping samples.
        method: PSD method ("welch" or "periodogram").

    Returns:
        Tuple of (frequencies, psd).
        For multichannel: psd shape is (channels, frequencies).
    """
    if nperseg is None:
        nperseg = min(256, data.shape[-1])

    if noverlap is None:
        noverlap = nperseg // 2

    if data.ndim == 1:
        if method == "welch":
            freqs, psd = signal.welch(
                data, fs=fs, nperseg=nperseg, noverlap=noverlap
            )
        else:
            freqs, psd = signal.periodogram(data, fs=fs)
        return freqs, psd

    else:
        # Multichannel
        psds = []
        for ch in data:
            if method == "welch":
                freqs, psd = signal.welch(
                    ch, fs=fs, nperseg=nperseg, noverlap=noverlap
                )
            else:
                freqs, psd = signal.periodogram(ch, fs=fs)
            psds.append(psd)

        return freqs, np.array(psds)


def compute_band_power(
    data: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    relative: bool = False,
) -> dict[str, np.ndarray]:
    """Compute power in frequency bands.

    Args:
        data: Signal array (samples,) or (channels, samples).
        fs: Sampling frequency in Hz.
        bands: Dictionary of band names to (low, high) frequencies.
            Uses standard EEG bands if None.
        relative: If True, return relative (normalized) power.

    Returns:
        Dictionary mapping band names to power values.
        For multichannel: values are arrays of shape (channels,).
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    # Compute PSD
    freqs, psd = compute_psd(data, fs)

    # Handle 1D case
    if psd.ndim == 1:
        psd = psd.reshape(1, -1)

    band_powers = {}
    total_power = np.trapz(psd, freqs, axis=-1)

    for band_name, (low, high) in bands.items():
        # Find frequency indices
        idx = (freqs >= low) & (freqs <= high)

        if np.any(idx):
            # Integrate power in band
            power = np.trapz(psd[:, idx], freqs[idx], axis=-1)

            if relative:
                power = power / np.maximum(total_power, 1e-10)

            band_powers[band_name] = power.squeeze()
        else:
            band_powers[band_name] = np.zeros(psd.shape[0]).squeeze()

    return band_powers


def compute_relative_band_power(
    data: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute relative band power (normalized by total power).

    Args:
        data: Signal array.
        fs: Sampling frequency.
        bands: Frequency bands.

    Returns:
        Dictionary of relative band powers.
    """
    return compute_band_power(data, fs, bands, relative=True)


def compute_band_ratios(
    data: np.ndarray,
    fs: float,
    ratios: list[tuple[str, str]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute ratios between frequency bands.

    Common ratios include:
    - theta/beta (attention)
    - alpha/theta (relaxation)
    - beta/alpha (alertness)

    Args:
        data: Signal array.
        fs: Sampling frequency.
        ratios: List of (numerator, denominator) band name tuples.

    Returns:
        Dictionary of ratio values.
    """
    if ratios is None:
        ratios = [
            ("theta", "beta"),
            ("alpha", "theta"),
            ("beta", "alpha"),
            ("theta", "alpha"),
        ]

    band_powers = compute_band_power(data, fs)

    ratio_values = {}
    for num, den in ratios:
        ratio_name = f"{num}/{den}"
        num_power = band_powers.get(num, 0)
        den_power = band_powers.get(den, 1)

        # Avoid division by zero
        ratio_values[ratio_name] = num_power / np.maximum(den_power, 1e-10)

    return ratio_values


class SpectralFeatures:
    """Spectral feature extractor for EEG signals.

    Computes PSD, band powers, and spectral metrics.

    Example:
        >>> sf = SpectralFeatures(fs=256)
        >>> features = sf.extract(eeg_data)
        >>> print(features["alpha"])  # Alpha band power
    """

    def __init__(
        self,
        fs: float,
        bands: dict[str, tuple[float, float]] | None = None,
        nperseg: int | None = None,
    ):
        """Initialize spectral feature extractor.

        Args:
            fs: Sampling frequency in Hz.
            bands: Frequency bands.
            nperseg: Segment length for PSD.
        """
        self.fs = fs
        self.bands = bands or FREQUENCY_BANDS
        self.nperseg = nperseg

    def extract(
        self,
        data: np.ndarray,
        include_ratios: bool = True,
    ) -> dict[str, Any]:
        """Extract all spectral features.

        Args:
            data: EEG data (channels, samples).
            include_ratios: Include band ratios.

        Returns:
            Dictionary of features.
        """
        features = {}

        # Absolute band powers
        abs_powers = compute_band_power(data, self.fs, self.bands)
        for band, power in abs_powers.items():
            features[f"{band}_abs"] = power

        # Relative band powers
        rel_powers = compute_relative_band_power(data, self.fs, self.bands)
        for band, power in rel_powers.items():
            features[f"{band}_rel"] = power

        # Band ratios
        if include_ratios:
            ratios = compute_band_ratios(data, self.fs)
            for ratio_name, value in ratios.items():
                features[f"ratio_{ratio_name.replace('/', '_')}"] = value

        # Spectral edge frequency (95% power)
        features["spectral_edge_95"] = self._compute_spectral_edge(data, 0.95)

        # Peak frequency
        features["peak_frequency"] = self._compute_peak_frequency(data)

        return features

    def _compute_spectral_edge(
        self,
        data: np.ndarray,
        percentile: float = 0.95,
    ) -> np.ndarray:
        """Compute spectral edge frequency.

        Args:
            data: Signal data.
            percentile: Power percentile (e.g., 0.95 for 95%).

        Returns:
            Spectral edge frequency for each channel.
        """
        freqs, psd = compute_psd(data, self.fs)

        if psd.ndim == 1:
            psd = psd.reshape(1, -1)

        edges = []
        for ch_psd in psd:
            cumsum = np.cumsum(ch_psd)
            total = cumsum[-1]
            edge_idx = np.searchsorted(cumsum, percentile * total)
            edge_idx = min(edge_idx, len(freqs) - 1)
            edges.append(freqs[edge_idx])

        return np.array(edges).squeeze()

    def _compute_peak_frequency(self, data: np.ndarray) -> np.ndarray:
        """Compute peak (dominant) frequency.

        Args:
            data: Signal data.

        Returns:
            Peak frequency for each channel.
        """
        freqs, psd = compute_psd(data, self.fs)

        if psd.ndim == 1:
            return freqs[np.argmax(psd)]

        peaks = []
        for ch_psd in psd:
            peaks.append(freqs[np.argmax(ch_psd)])

        return np.array(peaks).squeeze()

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names.
        """
        names = []

        for band in self.bands:
            names.extend([f"{band}_abs", f"{band}_rel"])

        names.extend([
            "ratio_theta_beta",
            "ratio_alpha_theta",
            "ratio_beta_alpha",
            "ratio_theta_alpha",
            "spectral_edge_95",
            "peak_frequency",
        ])

        return names
