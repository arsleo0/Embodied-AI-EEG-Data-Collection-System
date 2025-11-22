"""Introspection and meta-cognitive state detection.

Detects neural signatures of self-awareness, introspection,
and meta-cognitive monitoring from EEG data.
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
class MetaCognitiveState:
    """Meta-cognitive state indicators.

    Attributes:
        self_awareness: Level of self-awareness.
        introspection: Introspection intensity.
        monitoring: Self-monitoring activity.
        evaluation: Self-evaluation indicators.
        confidence: Confidence in state detection.
    """

    self_awareness: float
    introspection: float
    monitoring: float
    evaluation: float
    confidence: float

    @property
    def metacognitive_index(self) -> float:
        """Combined metacognitive index."""
        return (
            0.3 * self.self_awareness +
            0.3 * self.introspection +
            0.2 * self.monitoring +
            0.2 * self.evaluation
        )


class IntrospectionDetector:
    """Detect introspection and meta-cognitive states from EEG.

    Uses neural markers associated with self-referential
    processing, default mode network activity, and
    metacognitive monitoring.

    Example:
        >>> detector = IntrospectionDetector(fs=256)
        >>> state = detector.detect(eeg_data)
        >>> print(f"Metacognitive index: {state.metacognitive_index:.2f}")
    """

    def __init__(
        self,
        fs: float = 256.0,
        channel_names: list[str] | None = None,
    ):
        """Initialize introspection detector.

        Args:
            fs: Sampling frequency in Hz.
            channel_names: EEG channel names.
        """
        self.fs = fs
        self.channel_names = channel_names or ["TP9", "AF7", "AF8", "TP10"]

        # Frequency bands relevant to metacognition
        self.bands = {
            "theta": (4, 8),     # Working memory, self-referential
            "alpha": (8, 13),    # Inhibition, internal attention
            "low_beta": (13, 20), # Active monitoring
            "high_beta": (20, 30), # Alertness, evaluation
        }

        # Frontal channels for meta-cognition (AF7, AF8 in Muse)
        self._frontal_indices = self._get_frontal_indices()

    def _get_frontal_indices(self) -> list[int]:
        """Get indices of frontal channels."""
        frontal_prefixes = ["AF", "Fp", "F"]
        indices = []

        for i, name in enumerate(self.channel_names):
            if any(name.startswith(prefix) for prefix in frontal_prefixes):
                indices.append(i)

        # Default to first two if no frontal channels found
        return indices if indices else [0, 1]

    def detect(
        self,
        data: np.ndarray,
        baseline: np.ndarray | None = None,
    ) -> MetaCognitiveState:
        """Detect meta-cognitive state.

        Args:
            data: EEG data (channels, samples).
            baseline: Optional baseline for comparison.

        Returns:
            MetaCognitiveState object.
        """
        # Compute band powers
        band_powers = self._compute_band_powers(data)

        # Self-awareness: Frontal alpha/theta ratio
        self_awareness = self._compute_self_awareness(band_powers)

        # Introspection: Frontal midline theta
        introspection = self._compute_introspection(band_powers)

        # Monitoring: Low beta activity
        monitoring = self._compute_monitoring(band_powers)

        # Evaluation: High beta / alpha ratio
        evaluation = self._compute_evaluation(band_powers)

        # Confidence based on signal quality
        confidence = self._compute_confidence(data)

        # Normalize if baseline provided
        if baseline is not None:
            baseline_state = self.detect(baseline)
            self_awareness = self_awareness - baseline_state.self_awareness
            introspection = introspection - baseline_state.introspection
            monitoring = monitoring - baseline_state.monitoring
            evaluation = evaluation - baseline_state.evaluation

        return MetaCognitiveState(
            self_awareness=float(np.clip(self_awareness, 0, 1)),
            introspection=float(np.clip(introspection, 0, 1)),
            monitoring=float(np.clip(monitoring, 0, 1)),
            evaluation=float(np.clip(evaluation, 0, 1)),
            confidence=float(confidence),
        )

    def _compute_band_powers(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Compute power in each frequency band.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Band powers per channel.
        """
        band_powers = {}

        for band_name, (low, high) in self.bands.items():
            powers = []
            for i in range(data.shape[0]):
                if SCIPY_AVAILABLE:
                    freqs, psd = signal.welch(
                        data[i], self.fs,
                        nperseg=min(256, len(data[i]))
                    )
                    idx = np.logical_and(freqs >= low, freqs <= high)
                    power = np.trapz(psd[idx], freqs[idx])
                else:
                    # Simple approximation without scipy
                    power = np.var(data[i])
                powers.append(power)

            band_powers[band_name] = np.array(powers)

        return band_powers

    def _compute_self_awareness(
        self,
        band_powers: dict[str, np.ndarray],
    ) -> float:
        """Compute self-awareness indicator.

        Self-referential processing associated with
        alpha suppression and increased theta in frontal regions.

        Args:
            band_powers: Band powers per channel.

        Returns:
            Self-awareness score (0-1).
        """
        # Frontal alpha/theta ratio (lower = more self-awareness)
        frontal_theta = np.mean(band_powers["theta"][self._frontal_indices])
        frontal_alpha = np.mean(band_powers["alpha"][self._frontal_indices])

        # Normalize ratio
        ratio = frontal_theta / (frontal_alpha + 1e-10)

        # Map to 0-1 (typical ratio range: 0.5 - 2.0)
        score = np.clip((ratio - 0.5) / 1.5, 0, 1)

        return score

    def _compute_introspection(
        self,
        band_powers: dict[str, np.ndarray],
    ) -> float:
        """Compute introspection indicator.

        Frontal midline theta associated with internal attention
        and introspective processing.

        Args:
            band_powers: Band powers per channel.

        Returns:
            Introspection score (0-1).
        """
        # Frontal theta power
        frontal_theta = np.mean(band_powers["theta"][self._frontal_indices])

        # Compare to posterior
        posterior_indices = [i for i in range(len(self.channel_names))
                           if i not in self._frontal_indices]
        if posterior_indices:
            posterior_theta = np.mean(band_powers["theta"][posterior_indices])
        else:
            posterior_theta = frontal_theta

        # Ratio (higher = more introspection)
        ratio = frontal_theta / (posterior_theta + 1e-10)

        # Map to 0-1
        score = np.clip((ratio - 0.8) / 0.4, 0, 1)

        return score

    def _compute_monitoring(
        self,
        band_powers: dict[str, np.ndarray],
    ) -> float:
        """Compute self-monitoring indicator.

        Low beta activity associated with active monitoring
        and error detection.

        Args:
            band_powers: Band powers per channel.

        Returns:
            Monitoring score (0-1).
        """
        # Frontal low beta
        frontal_low_beta = np.mean(band_powers["low_beta"][self._frontal_indices])

        # Relative to total power
        total_power = sum(
            np.mean(powers[self._frontal_indices])
            for powers in band_powers.values()
        )

        relative_power = frontal_low_beta / (total_power + 1e-10)

        # Map to 0-1 (typical range: 0.1 - 0.3)
        score = np.clip((relative_power - 0.1) / 0.2, 0, 1)

        return score

    def _compute_evaluation(
        self,
        band_powers: dict[str, np.ndarray],
    ) -> float:
        """Compute self-evaluation indicator.

        High beta / alpha ratio associated with evaluative
        and critical thinking processes.

        Args:
            band_powers: Band powers per channel.

        Returns:
            Evaluation score (0-1).
        """
        # High beta to alpha ratio
        frontal_high_beta = np.mean(band_powers["high_beta"][self._frontal_indices])
        frontal_alpha = np.mean(band_powers["alpha"][self._frontal_indices])

        ratio = frontal_high_beta / (frontal_alpha + 1e-10)

        # Map to 0-1
        score = np.clip(ratio / 0.5, 0, 1)

        return score

    def _compute_confidence(self, data: np.ndarray) -> float:
        """Compute confidence in detection.

        Based on signal quality metrics.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Confidence score (0-1).
        """
        # Check for artifacts
        max_amplitude = np.max(np.abs(data))
        amplitude_ok = max_amplitude < 100  # μV

        # Check for sufficient variation
        std = np.mean(np.std(data, axis=1))
        variation_ok = std > 1 and std < 50

        # Base confidence
        confidence = 0.5

        if amplitude_ok:
            confidence += 0.25
        if variation_ok:
            confidence += 0.25

        return confidence

    def track_over_time(
        self,
        data: np.ndarray,
        window_size: float = 2.0,
        overlap: float = 0.5,
    ) -> list[MetaCognitiveState]:
        """Track meta-cognitive state over time.

        Args:
            data: EEG data (channels, samples).
            window_size: Window size in seconds.
            overlap: Overlap fraction.

        Returns:
            List of states for each window.
        """
        window_samples = int(window_size * self.fs)
        step_samples = int(window_samples * (1 - overlap))

        states = []
        start = 0

        while start + window_samples <= data.shape[1]:
            window = data[:, start:start + window_samples]
            state = self.detect(window)
            states.append(state)
            start += step_samples

        return states


def detect_self_awareness(
    data: np.ndarray,
    fs: float = 256.0,
    channel_names: list[str] | None = None,
) -> float:
    """Convenience function for self-awareness detection.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency.
        channel_names: Channel names.

    Returns:
        Self-awareness score (0-1).
    """
    detector = IntrospectionDetector(fs, channel_names)
    state = detector.detect(data)
    return state.self_awareness


def compute_metacognitive_index(
    data: np.ndarray,
    fs: float = 256.0,
    channel_names: list[str] | None = None,
) -> float:
    """Compute overall metacognitive index.

    Args:
        data: EEG data (channels, samples).
        fs: Sampling frequency.
        channel_names: Channel names.

    Returns:
        Metacognitive index (0-1).
    """
    detector = IntrospectionDetector(fs, channel_names)
    state = detector.detect(data)
    return state.metacognitive_index
