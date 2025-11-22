"""Artifact detection and removal for EEG signals.

Detects and removes common EEG artifacts:
- Eye blinks (frontal channels)
- Muscle artifacts (high-frequency)
- Movement artifacts (large amplitude)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal, stats


@dataclass
class ArtifactSegment:
    """Represents a detected artifact segment."""

    start_idx: int
    end_idx: int
    artifact_type: str
    channel: int | None = None
    amplitude: float = 0.0


def detect_eye_blinks(
    data: np.ndarray,
    fs: float,
    threshold: float = 100.0,
    min_duration: float = 0.1,
    max_duration: float = 0.4,
) -> list[ArtifactSegment]:
    """Detect eye blink artifacts in EEG data.

    Eye blinks typically appear as large amplitude deflections
    lasting 100-400ms, primarily in frontal channels.

    Args:
        data: EEG data (samples,) or (channels, samples).
        fs: Sampling frequency in Hz.
        threshold: Amplitude threshold in µV.
        min_duration: Minimum blink duration in seconds.
        max_duration: Maximum blink duration in seconds.

    Returns:
        List of detected artifact segments.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    artifacts = []
    min_samples = int(min_duration * fs)
    max_samples = int(max_duration * fs)

    for ch_idx in range(data.shape[0]):
        channel = data[ch_idx]

        # Find samples exceeding threshold
        above_threshold = np.abs(channel) > threshold

        # Find contiguous segments
        segments = _find_contiguous_segments(above_threshold)

        for start, end in segments:
            duration = end - start

            # Check if duration is consistent with eye blink
            if min_samples <= duration <= max_samples:
                artifacts.append(ArtifactSegment(
                    start_idx=start,
                    end_idx=end,
                    artifact_type="eye_blink",
                    channel=ch_idx,
                    amplitude=np.max(np.abs(channel[start:end])),
                ))

    return artifacts


def detect_muscle_artifacts(
    data: np.ndarray,
    fs: float,
    high_freq_threshold: float = 2.0,
    freq_range: tuple[float, float] = (20, 50),
) -> list[ArtifactSegment]:
    """Detect muscle artifacts in EEG data.

    Muscle artifacts appear as high-frequency activity (>20 Hz).
    Uses ratio of high-frequency to low-frequency power.

    Args:
        data: EEG data (samples,) or (channels, samples).
        fs: Sampling frequency in Hz.
        high_freq_threshold: Ratio threshold for detection.
        freq_range: Frequency range for muscle activity.

    Returns:
        List of detected artifact segments.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    artifacts = []
    window_size = int(fs)  # 1-second windows
    hop_size = int(fs / 4)  # 250ms hop

    for ch_idx in range(data.shape[0]):
        channel = data[ch_idx]

        for start in range(0, len(channel) - window_size, hop_size):
            end = start + window_size
            segment = channel[start:end]

            # Compute power in high and low frequency bands
            freqs, psd = signal.welch(segment, fs=fs, nperseg=min(256, len(segment)))

            low_idx = (freqs >= 1) & (freqs < 20)
            high_idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

            low_power = np.mean(psd[low_idx]) if np.any(low_idx) else 1
            high_power = np.mean(psd[high_idx]) if np.any(high_idx) else 0

            ratio = high_power / max(low_power, 1e-10)

            if ratio > high_freq_threshold:
                artifacts.append(ArtifactSegment(
                    start_idx=start,
                    end_idx=end,
                    artifact_type="muscle",
                    channel=ch_idx,
                    amplitude=ratio,
                ))

    return artifacts


def detect_movement_artifacts(
    data: np.ndarray,
    fs: float,
    std_threshold: float = 4.0,
    window_size: float = 0.5,
) -> list[ArtifactSegment]:
    """Detect movement artifacts using amplitude variability.

    Movement artifacts cause sudden large amplitude changes.

    Args:
        data: EEG data (samples,) or (channels, samples).
        fs: Sampling frequency in Hz.
        std_threshold: Standard deviation multiplier for threshold.
        window_size: Window size in seconds.

    Returns:
        List of detected artifact segments.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    artifacts = []
    window_samples = int(window_size * fs)

    for ch_idx in range(data.shape[0]):
        channel = data[ch_idx]

        # Compute rolling standard deviation
        global_std = np.std(channel)
        threshold = std_threshold * global_std

        for start in range(0, len(channel) - window_samples, window_samples // 2):
            end = start + window_samples
            segment = channel[start:end]

            # Check for large amplitude deviations
            if np.max(np.abs(segment)) > threshold:
                artifacts.append(ArtifactSegment(
                    start_idx=start,
                    end_idx=end,
                    artifact_type="movement",
                    channel=ch_idx,
                    amplitude=np.max(np.abs(segment)),
                ))

    return artifacts


def remove_artifacts(
    data: np.ndarray,
    artifacts: list[ArtifactSegment],
    method: str = "interpolate",
) -> np.ndarray:
    """Remove detected artifacts from EEG data.

    Args:
        data: EEG data (channels, samples).
        artifacts: List of artifact segments to remove.
        method: Removal method:
            - "interpolate": Linear interpolation
            - "zero": Set to zero
            - "nan": Set to NaN

    Returns:
        Cleaned data.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    cleaned = data.copy()

    for artifact in artifacts:
        start = artifact.start_idx
        end = artifact.end_idx

        if artifact.channel is not None:
            channels = [artifact.channel]
        else:
            channels = range(data.shape[0])

        for ch in channels:
            if method == "interpolate":
                # Linear interpolation
                if start > 0 and end < data.shape[1]:
                    x = np.array([start - 1, end])
                    y = np.array([cleaned[ch, start - 1], cleaned[ch, end]])
                    xp = np.arange(start, end)
                    cleaned[ch, start:end] = np.interp(xp, x, y)
                else:
                    cleaned[ch, start:end] = 0
            elif method == "zero":
                cleaned[ch, start:end] = 0
            elif method == "nan":
                cleaned[ch, start:end] = np.nan

    return cleaned


class ArtifactDetector:
    """Comprehensive artifact detection for EEG data.

    Combines multiple detection methods and provides
    a unified interface for artifact management.

    Example:
        >>> detector = ArtifactDetector(fs=256)
        >>> detector.detect_all(eeg_data)
        >>> cleaned = detector.remove_all(eeg_data)
    """

    def __init__(
        self,
        fs: float,
        blink_threshold: float = 100.0,
        muscle_threshold: float = 2.0,
        movement_threshold: float = 4.0,
    ):
        """Initialize artifact detector.

        Args:
            fs: Sampling frequency in Hz.
            blink_threshold: Threshold for eye blink detection.
            muscle_threshold: Threshold for muscle artifact detection.
            movement_threshold: Threshold for movement detection.
        """
        self.fs = fs
        self.blink_threshold = blink_threshold
        self.muscle_threshold = muscle_threshold
        self.movement_threshold = movement_threshold
        self._artifacts: list[ArtifactSegment] = []

    def detect_all(self, data: np.ndarray) -> list[ArtifactSegment]:
        """Detect all artifact types.

        Args:
            data: EEG data (channels, samples).

        Returns:
            List of all detected artifacts.
        """
        self._artifacts = []

        # Detect eye blinks
        blinks = detect_eye_blinks(
            data, self.fs, threshold=self.blink_threshold
        )
        self._artifacts.extend(blinks)

        # Detect muscle artifacts
        muscle = detect_muscle_artifacts(
            data, self.fs, high_freq_threshold=self.muscle_threshold
        )
        self._artifacts.extend(muscle)

        # Detect movement artifacts
        movement = detect_movement_artifacts(
            data, self.fs, std_threshold=self.movement_threshold
        )
        self._artifacts.extend(movement)

        return self._artifacts

    def remove_all(
        self,
        data: np.ndarray,
        method: str = "interpolate",
    ) -> np.ndarray:
        """Remove all detected artifacts.

        Args:
            data: EEG data.
            method: Removal method.

        Returns:
            Cleaned data.
        """
        if not self._artifacts:
            self.detect_all(data)

        return remove_artifacts(data, self._artifacts, method)

    def get_artifact_mask(self, n_samples: int) -> np.ndarray:
        """Get boolean mask of artifact locations.

        Args:
            n_samples: Total number of samples.

        Returns:
            Boolean array (True = artifact).
        """
        mask = np.zeros(n_samples, dtype=bool)

        for artifact in self._artifacts:
            mask[artifact.start_idx:artifact.end_idx] = True

        return mask

    def get_clean_ratio(self, n_samples: int) -> float:
        """Get ratio of clean (non-artifact) data.

        Args:
            n_samples: Total number of samples.

        Returns:
            Ratio between 0 and 1.
        """
        mask = self.get_artifact_mask(n_samples)
        return 1 - (np.sum(mask) / n_samples)

    @property
    def artifacts(self) -> list[ArtifactSegment]:
        """Get detected artifacts."""
        return self._artifacts

    def summary(self) -> dict[str, Any]:
        """Get summary of detected artifacts.

        Returns:
            Dictionary with artifact counts by type.
        """
        summary = {
            "total": len(self._artifacts),
            "eye_blink": 0,
            "muscle": 0,
            "movement": 0,
        }

        for artifact in self._artifacts:
            if artifact.artifact_type in summary:
                summary[artifact.artifact_type] += 1

        return summary


def _find_contiguous_segments(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Find contiguous True segments in boolean mask.

    Args:
        mask: Boolean array.

    Returns:
        List of (start, end) tuples.
    """
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(mask)))

    return segments
