"""Attention pattern analysis for consciousness states.

Analyzes attention patterns from EEG data using Global
Workspace Theory inspired metrics and entropy measures.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy import signal
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class GlobalWorkspaceMetrics:
    """Metrics inspired by Global Workspace Theory.

    Global Workspace Theory posits that consciousness arises
    from widespread information broadcasting across brain regions.

    Attributes:
        integration: Information integration across channels.
        differentiation: Diversity of neural patterns.
        broadcast_strength: Strength of global broadcasting.
        workspace_stability: Stability of global workspace.
        ignition_events: Detected ignition events.
    """

    integration: float
    differentiation: float
    broadcast_strength: float
    workspace_stability: float
    ignition_events: int


class AttentionAnalyzer:
    """Analyze attention patterns from EEG data.

    Computes attention-related metrics including entropy,
    distribution, and Global Workspace Theory measures.

    Example:
        >>> analyzer = AttentionAnalyzer(fs=256)
        >>> metrics = analyzer.compute_global_workspace(eeg_data)
        >>> entropy = analyzer.compute_attention_entropy(eeg_data)
    """

    def __init__(
        self,
        fs: float = 256.0,
        attention_bands: dict[str, tuple[float, float]] | None = None,
    ):
        """Initialize attention analyzer.

        Args:
            fs: Sampling frequency in Hz.
            attention_bands: Frequency bands for attention analysis.
        """
        self.fs = fs

        # Default attention-related frequency bands
        self.attention_bands = attention_bands or {
            "theta": (4, 8),      # Working memory, attention
            "alpha": (8, 13),     # Inhibition, attention gating
            "beta": (13, 30),     # Active attention, focus
            "gamma": (30, 50),    # Binding, conscious awareness
        }

    def compute_global_workspace(
        self,
        data: np.ndarray,
        window_size: float = 1.0,
    ) -> GlobalWorkspaceMetrics:
        """Compute Global Workspace Theory metrics.

        Args:
            data: EEG data (channels, samples).
            window_size: Analysis window in seconds.

        Returns:
            GlobalWorkspaceMetrics object.
        """
        n_channels, n_samples = data.shape

        # Integration: Mutual information between channels
        integration = self._compute_integration(data)

        # Differentiation: Complexity/diversity of patterns
        differentiation = self._compute_differentiation(data)

        # Broadcast strength: Cross-channel correlation in gamma
        broadcast_strength = self._compute_broadcast_strength(data)

        # Workspace stability: Temporal consistency
        workspace_stability = self._compute_workspace_stability(data, window_size)

        # Ignition events: Sudden widespread activation
        ignition_events = self._detect_ignition_events(data)

        return GlobalWorkspaceMetrics(
            integration=integration,
            differentiation=differentiation,
            broadcast_strength=broadcast_strength,
            workspace_stability=workspace_stability,
            ignition_events=ignition_events,
        )

    def _compute_integration(self, data: np.ndarray) -> float:
        """Compute information integration across channels.

        Uses correlation-based approximation of mutual information.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Integration score (0-1).
        """
        # Correlation matrix
        corr_matrix = np.corrcoef(data)

        # Remove diagonal
        np.fill_diagonal(corr_matrix, 0)

        # Integration = mean absolute correlation
        integration = np.mean(np.abs(corr_matrix))

        return float(integration)

    def _compute_differentiation(self, data: np.ndarray) -> float:
        """Compute pattern differentiation/complexity.

        Uses singular value decomposition to measure the
        diversity of neural patterns.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Differentiation score.
        """
        # Normalize data
        data_norm = (data - np.mean(data, axis=1, keepdims=True))
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1
        data_norm = data_norm / std

        # SVD
        try:
            _, s, _ = np.linalg.svd(data_norm, full_matrices=False)

            # Normalize singular values
            s_norm = s / np.sum(s)

            # Spectral entropy as differentiation measure
            differentiation = -np.sum(s_norm * np.log(s_norm + 1e-10))
        except np.linalg.LinAlgError:
            differentiation = 0.0

        return float(differentiation)

    def _compute_broadcast_strength(self, data: np.ndarray) -> float:
        """Compute global broadcasting strength.

        Measures synchronization in gamma band (30-50 Hz)
        across all channels.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Broadcast strength (0-1).
        """
        if not SCIPY_AVAILABLE:
            return 0.0

        # Filter to gamma band
        low, high = self.attention_bands["gamma"]
        nyq = self.fs / 2
        b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")

        gamma_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            gamma_data[i] = signal.filtfilt(b, a, data[i])

        # Compute phase locking value (PLV) approximation
        # Using Hilbert transform
        analytic = signal.hilbert(gamma_data, axis=1)
        phases = np.angle(analytic)

        # Mean phase coherence across all channel pairs
        n_channels = data.shape[0]
        plv_sum = 0
        n_pairs = 0

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_sum += plv
                n_pairs += 1

        broadcast_strength = plv_sum / n_pairs if n_pairs > 0 else 0

        return float(broadcast_strength)

    def _compute_workspace_stability(
        self,
        data: np.ndarray,
        window_size: float,
    ) -> float:
        """Compute temporal stability of global workspace.

        Measures how consistent the attention pattern is
        over time.

        Args:
            data: EEG data (channels, samples).
            window_size: Window size in seconds.

        Returns:
            Stability score (0-1).
        """
        window_samples = int(window_size * self.fs)
        n_windows = data.shape[1] // window_samples

        if n_windows < 2:
            return 1.0

        # Compute feature vector for each window
        window_features = []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window_data = data[:, start:end]

            # Feature: power in each band for each channel
            features = []
            for band_name, (low, high) in self.attention_bands.items():
                band_power = self._compute_band_power(window_data, low, high)
                features.extend(band_power)

            window_features.append(features)

        window_features = np.array(window_features)

        # Stability: correlation between consecutive windows
        correlations = []
        for i in range(len(window_features) - 1):
            corr = np.corrcoef(window_features[i], window_features[i + 1])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        stability = np.mean(correlations) if correlations else 0

        return float(np.clip(stability, 0, 1))

    def _detect_ignition_events(
        self,
        data: np.ndarray,
        threshold: float = 2.0,
    ) -> int:
        """Detect global ignition events.

        Ignition events are sudden, widespread activations
        across multiple channels.

        Args:
            data: EEG data (channels, samples).
            threshold: Z-score threshold for detection.

        Returns:
            Number of detected ignition events.
        """
        # Compute global amplitude (mean across channels)
        global_amp = np.mean(np.abs(data), axis=0)

        # Z-score
        z_scores = (global_amp - np.mean(global_amp)) / (np.std(global_amp) + 1e-10)

        # Find peaks above threshold
        above_threshold = z_scores > threshold

        # Count transitions from below to above threshold
        events = np.sum(np.diff(above_threshold.astype(int)) > 0)

        return int(events)

    def _compute_band_power(
        self,
        data: np.ndarray,
        low_freq: float,
        high_freq: float,
    ) -> np.ndarray:
        """Compute power in frequency band for each channel.

        Args:
            data: EEG data (channels, samples).
            low_freq: Low frequency cutoff.
            high_freq: High frequency cutoff.

        Returns:
            Band power for each channel.
        """
        if not SCIPY_AVAILABLE:
            return np.zeros(data.shape[0])

        powers = []
        for i in range(data.shape[0]):
            freqs, psd = signal.welch(data[i], self.fs, nperseg=min(256, len(data[i])))

            # Find indices for frequency band
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)

            # Compute band power
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            powers.append(band_power)

        return np.array(powers)

    def compute_attention_entropy(
        self,
        data: np.ndarray,
        n_bins: int = 20,
    ) -> dict[str, float]:
        """Compute attention entropy measures.

        Args:
            data: EEG data (channels, samples).
            n_bins: Number of bins for histogram.

        Returns:
            Dictionary of entropy measures.
        """
        results = {}

        # Spatial entropy: distribution across channels
        channel_power = np.sum(data ** 2, axis=1)
        channel_dist = channel_power / (np.sum(channel_power) + 1e-10)
        results["spatial_entropy"] = float(entropy(channel_dist + 1e-10))

        # Temporal entropy: complexity over time
        results["temporal_entropy"] = float(np.mean([
            self._sample_entropy(data[i]) for i in range(data.shape[0])
        ]))

        # Spectral entropy per band
        for band_name, (low, high) in self.attention_bands.items():
            band_power = self._compute_band_power(data, low, high)
            band_dist = band_power / (np.sum(band_power) + 1e-10)
            results[f"{band_name}_entropy"] = float(entropy(band_dist + 1e-10))

        return results

    def _sample_entropy(
        self,
        x: np.ndarray,
        m: int = 2,
        r: float | None = None,
    ) -> float:
        """Compute sample entropy.

        Args:
            x: Time series.
            m: Embedding dimension.
            r: Tolerance (default: 0.2 * std).

        Returns:
            Sample entropy value.
        """
        n = len(x)
        if n < m + 1:
            return 0.0

        if r is None:
            r = 0.2 * np.std(x)

        # Create template vectors
        def _count_matches(templates: np.ndarray, r: float) -> int:
            count = 0
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count

        # Templates of length m
        templates_m = np.array([x[i:i + m] for i in range(n - m)])
        count_m = _count_matches(templates_m, r)

        # Templates of length m+1
        templates_m1 = np.array([x[i:i + m + 1] for i in range(n - m - 1)])
        count_m1 = _count_matches(templates_m1, r)

        if count_m == 0 or count_m1 == 0:
            return 0.0

        return -np.log(count_m1 / count_m)

    def compute_attention_distribution(
        self,
        data: np.ndarray,
    ) -> dict[str, Any]:
        """Compute attention distribution across bands and channels.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Attention distribution metrics.
        """
        n_channels = data.shape[0]

        # Band powers for each channel
        band_powers = {}
        for band_name, (low, high) in self.attention_bands.items():
            band_powers[band_name] = self._compute_band_power(data, low, high)

        # Total power per band
        total_band_power = {
            band: float(np.sum(powers))
            for band, powers in band_powers.items()
        }

        # Normalize to get distribution
        total = sum(total_band_power.values())
        band_distribution = {
            band: power / (total + 1e-10)
            for band, power in total_band_power.items()
        }

        # Channel distribution (total power per channel)
        channel_power = np.sum(
            [powers for powers in band_powers.values()],
            axis=0
        )
        channel_distribution = channel_power / (np.sum(channel_power) + 1e-10)

        # Attention focus: how concentrated is attention?
        # High focus = low entropy
        focus_score = 1 - entropy(list(band_distribution.values())) / np.log(len(band_distribution))

        return {
            "band_powers": {
                band: powers.tolist()
                for band, powers in band_powers.items()
            },
            "band_distribution": band_distribution,
            "channel_distribution": channel_distribution.tolist(),
            "focus_score": float(focus_score),
            "dominant_band": max(band_distribution, key=band_distribution.get),
        }


def compute_attention_entropy(
    data: np.ndarray,
    fs: float = 256.0,
) -> dict[str, float]:
    """Convenience function for computing attention entropy.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency.

    Returns:
        Entropy measures.
    """
    analyzer = AttentionAnalyzer(fs)
    return analyzer.compute_attention_entropy(data)


def compute_attention_distribution(
    data: np.ndarray,
    fs: float = 256.0,
) -> dict[str, Any]:
    """Convenience function for computing attention distribution.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency.

    Returns:
        Distribution metrics.
    """
    analyzer = AttentionAnalyzer(fs)
    return analyzer.compute_attention_distribution(data)
