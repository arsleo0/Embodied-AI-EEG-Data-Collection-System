"""Temporal binding window analysis for consciousness.

Analyzes temporal integration windows, context retention,
and memory consolidation patterns in consciousness states.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy import signal
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TemporalBindingMetrics:
    """Metrics for temporal binding analysis.

    Attributes:
        integration_window: Estimated integration window in seconds.
        context_retention: Context retention score (0-1).
        binding_strength: Cross-channel binding strength.
        memory_consolidation: Memory consolidation pattern score.
        temporal_coherence: Temporal coherence across time.
    """

    integration_window: float
    context_retention: float
    binding_strength: float
    memory_consolidation: float
    temporal_coherence: float


class TemporalBindingTracker:
    """Track temporal binding windows in consciousness.

    Analyzes how neural signals integrate information
    across time, providing insights into the specious
    present and working memory span.

    Example:
        >>> tracker = TemporalBindingTracker(fs=256)
        >>> metrics = tracker.analyze(eeg_data)
        >>> print(f"Integration window: {metrics.integration_window:.3f}s")
    """

    def __init__(
        self,
        fs: float = 256.0,
        max_window: float = 3.0,
        min_window: float = 0.1,
    ):
        """Initialize temporal binding tracker.

        Args:
            fs: Sampling frequency in Hz.
            max_window: Maximum integration window to test (seconds).
            min_window: Minimum integration window to test (seconds).
        """
        self.fs = fs
        self.max_window = max_window
        self.min_window = min_window

    def analyze(self, data: np.ndarray) -> TemporalBindingMetrics:
        """Analyze temporal binding in EEG data.

        Args:
            data: EEG data (channels, samples).

        Returns:
            TemporalBindingMetrics object.
        """
        # Estimate integration window
        integration_window = self._estimate_integration_window(data)

        # Compute context retention
        context_retention = self._compute_context_retention(data)

        # Compute binding strength across channels
        binding_strength = self._compute_binding_strength(data)

        # Analyze memory consolidation patterns
        memory_consolidation = self._analyze_memory_consolidation(data)

        # Compute temporal coherence
        temporal_coherence = self._compute_temporal_coherence(data)

        return TemporalBindingMetrics(
            integration_window=integration_window,
            context_retention=context_retention,
            binding_strength=binding_strength,
            memory_consolidation=memory_consolidation,
            temporal_coherence=temporal_coherence,
        )

    def _estimate_integration_window(self, data: np.ndarray) -> float:
        """Estimate temporal integration window.

        Uses autocorrelation decay to estimate how long
        information persists in the neural signal.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Integration window in seconds.
        """
        n_channels = data.shape[0]
        windows = []

        for ch in range(n_channels):
            # Compute autocorrelation
            signal_ch = data[ch] - np.mean(data[ch])
            autocorr = np.correlate(signal_ch, signal_ch, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            # Find decay time (when autocorr drops to 1/e)
            threshold = 1 / np.e
            decay_idx = np.where(autocorr < threshold)[0]

            if len(decay_idx) > 0:
                window = decay_idx[0] / self.fs
            else:
                window = len(autocorr) / self.fs

            windows.append(np.clip(window, self.min_window, self.max_window))

        return float(np.mean(windows))

    def _compute_context_retention(self, data: np.ndarray) -> float:
        """Compute context retention score.

        Measures how well context is maintained over time
        by analyzing correlation between past and present.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Context retention score (0-1).
        """
        n_samples = data.shape[1]
        window_samples = int(0.5 * self.fs)  # 500ms windows

        if n_samples < 3 * window_samples:
            return 0.5

        correlations = []

        # Compare windows across time
        for i in range(0, n_samples - 2 * window_samples, window_samples):
            past = data[:, i:i + window_samples].flatten()
            present = data[:, i + window_samples:i + 2 * window_samples].flatten()

            if SCIPY_AVAILABLE:
                corr, _ = pearsonr(past, present)
            else:
                corr = np.corrcoef(past, present)[0, 1]

            if not np.isnan(corr):
                correlations.append(abs(corr))

        if not correlations:
            return 0.5

        return float(np.mean(correlations))

    def _compute_binding_strength(self, data: np.ndarray) -> float:
        """Compute cross-channel binding strength.

        Measures how strongly different channels are
        bound together temporally.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Binding strength (0-1).
        """
        n_channels = data.shape[0]

        if n_channels < 2:
            return 1.0

        # Compute cross-correlations between all channel pairs
        binding_scores = []

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Cross-correlation
                xcorr = np.correlate(
                    data[i] - np.mean(data[i]),
                    data[j] - np.mean(data[j]),
                    mode='full'
                )

                # Normalize
                norm = np.sqrt(np.sum(data[i]**2) * np.sum(data[j]**2))
                if norm > 0:
                    xcorr = xcorr / norm

                # Maximum correlation as binding strength
                binding_scores.append(np.max(np.abs(xcorr)))

        return float(np.mean(binding_scores))

    def _analyze_memory_consolidation(self, data: np.ndarray) -> float:
        """Analyze memory consolidation patterns.

        Looks for patterns indicative of memory consolidation
        such as replay-like activity.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Memory consolidation score (0-1).
        """
        if not SCIPY_AVAILABLE:
            return 0.5

        # Analyze theta-gamma coupling (related to memory)
        n_samples = data.shape[1]

        if n_samples < 256:
            return 0.5

        # Filter for theta (4-8 Hz) and gamma (30-50 Hz)
        nyq = self.fs / 2

        try:
            b_theta, a_theta = signal.butter(4, [4/nyq, 8/nyq], btype='band')
            b_gamma, a_gamma = signal.butter(4, [30/nyq, min(50/nyq, 0.95)], btype='band')
        except ValueError:
            return 0.5

        consolidation_scores = []

        for ch in range(data.shape[0]):
            # Filter signals
            theta = signal.filtfilt(b_theta, a_theta, data[ch])
            gamma = signal.filtfilt(b_gamma, a_gamma, data[ch])

            # Compute theta phase
            theta_analytic = signal.hilbert(theta)
            theta_phase = np.angle(theta_analytic)

            # Compute gamma amplitude
            gamma_amp = np.abs(signal.hilbert(gamma))

            # Phase-amplitude coupling (simplified)
            # Higher coupling suggests memory consolidation
            coupling = np.abs(np.mean(gamma_amp * np.exp(1j * theta_phase)))
            consolidation_scores.append(coupling)

        # Normalize
        score = np.mean(consolidation_scores)
        score = score / (np.std(data) + 1e-10)  # Normalize by signal variance

        return float(np.clip(score, 0, 1))

    def _compute_temporal_coherence(self, data: np.ndarray) -> float:
        """Compute temporal coherence across time.

        Measures consistency of neural patterns over time.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Temporal coherence score (0-1).
        """
        n_samples = data.shape[1]
        window_samples = int(0.2 * self.fs)  # 200ms windows

        if n_samples < 2 * window_samples:
            return 0.5

        # Compute variance of window patterns
        patterns = []

        for i in range(0, n_samples - window_samples, window_samples // 2):
            window = data[:, i:i + window_samples]
            # Feature: flattened and normalized
            pattern = window.flatten()
            pattern = pattern / (np.linalg.norm(pattern) + 1e-10)
            patterns.append(pattern)

        if len(patterns) < 2:
            return 0.5

        patterns = np.array(patterns)

        # Coherence = mean pairwise similarity
        similarities = []
        for i in range(len(patterns) - 1):
            sim = np.dot(patterns[i], patterns[i + 1])
            similarities.append(sim)

        return float(np.clip(np.mean(similarities), 0, 1))

    def sliding_window_analysis(
        self,
        data: np.ndarray,
        window_size: float = 2.0,
        step_size: float = 0.5,
    ) -> list[TemporalBindingMetrics]:
        """Perform sliding window analysis.

        Args:
            data: EEG data (channels, samples).
            window_size: Window size in seconds.
            step_size: Step size in seconds.

        Returns:
            List of metrics for each window.
        """
        window_samples = int(window_size * self.fs)
        step_samples = int(step_size * self.fs)
        n_samples = data.shape[1]

        results = []

        for start in range(0, n_samples - window_samples, step_samples):
            window = data[:, start:start + window_samples]
            metrics = self.analyze(window)
            results.append(metrics)

        return results


def estimate_specious_present(
    data: np.ndarray,
    fs: float = 256.0,
) -> float:
    """Estimate the duration of the specious present.

    The specious present is the perceived duration of "now"
    in consciousness, typically 2-3 seconds.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency.

    Returns:
        Estimated specious present duration in seconds.
    """
    tracker = TemporalBindingTracker(fs=fs, max_window=5.0)
    metrics = tracker.analyze(data)

    # Specious present is related to integration window
    # but typically longer (encompasses past + present)
    specious_present = metrics.integration_window * 2

    # Typical range: 1-5 seconds
    return float(np.clip(specious_present, 1.0, 5.0))


def compute_working_memory_span(
    data: np.ndarray,
    fs: float = 256.0,
) -> dict[str, float]:
    """Estimate working memory span from EEG.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency.

    Returns:
        Working memory metrics.
    """
    tracker = TemporalBindingTracker(fs=fs)
    metrics = tracker.analyze(data)

    # WM span related to context retention and integration
    span_seconds = metrics.integration_window * metrics.context_retention * 10

    return {
        "span_seconds": float(span_seconds),
        "span_items": float(np.clip(span_seconds * 2, 1, 9)),  # Approximate items
        "retention_quality": float(metrics.context_retention),
    }
