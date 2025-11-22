"""EEG preprocessing pipeline.

Provides a complete preprocessing pipeline combining:
- Filtering
- Artifact removal
- Baseline correction
- Normalization
"""

from typing import Any

import numpy as np
from scipy import signal

from .filters import FilterBank, bandpass_filter, notch_filter
from .artifacts import ArtifactDetector
from .quality import SignalQualityAssessor


def baseline_correct(
    data: np.ndarray,
    baseline_samples: int | None = None,
) -> np.ndarray:
    """Apply baseline correction to signal.

    Subtracts the mean of the baseline period from the entire signal.

    Args:
        data: Input signal (samples,) or (channels, samples).
        baseline_samples: Number of samples for baseline.
            If None, uses entire signal mean.

    Returns:
        Baseline-corrected signal.
    """
    if data.ndim == 1:
        if baseline_samples:
            baseline = np.mean(data[:baseline_samples])
        else:
            baseline = np.mean(data)
        return data - baseline
    else:
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            if baseline_samples:
                baseline = np.mean(data[i, :baseline_samples])
            else:
                baseline = np.mean(data[i])
            result[i] = data[i] - baseline
        return result


def normalize_signal(
    data: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Normalize signal amplitude.

    Args:
        data: Input signal (samples,) or (channels, samples).
        method: Normalization method:
            - "zscore": Z-score normalization (mean=0, std=1)
            - "minmax": Min-max scaling to [0, 1]
            - "robust": Robust scaling using median and IQR

    Returns:
        Normalized signal.
    """
    if data.ndim == 1:
        return _normalize_1d(data, method)
    else:
        return np.array([_normalize_1d(ch, method) for ch in data])


def _normalize_1d(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize 1D signal."""
    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std

    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    elif method == "robust":
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return data - median
        return (data - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_signal(
    data: np.ndarray,
    original_fs: float,
    target_fs: float,
) -> np.ndarray:
    """Resample signal to a different sampling rate.

    Args:
        data: Input signal (samples,) or (channels, samples).
        original_fs: Original sampling frequency.
        target_fs: Target sampling frequency.

    Returns:
        Resampled signal.
    """
    if original_fs == target_fs:
        return data

    # Compute resampling ratio
    ratio = target_fs / original_fs

    if data.ndim == 1:
        new_length = int(len(data) * ratio)
        return signal.resample(data, new_length)
    else:
        new_length = int(data.shape[1] * ratio)
        return np.array([signal.resample(ch, new_length) for ch in data])


class Preprocessor:
    """Complete EEG preprocessing pipeline.

    Combines filtering, artifact removal, and normalization
    into a configurable pipeline.

    Example:
        >>> preprocessor = Preprocessor(fs=256)
        >>> preprocessor.configure(
        ...     notch_freq=60,
        ...     bandpass=(1, 50),
        ...     remove_artifacts=True,
        ...     normalize=True
        ... )
        >>> clean_eeg = preprocessor.process(raw_eeg)
    """

    def __init__(
        self,
        fs: float,
        line_freq: float = 60.0,
    ):
        """Initialize preprocessor.

        Args:
            fs: Sampling frequency in Hz.
            line_freq: Power line frequency (50 or 60 Hz).
        """
        self.fs = fs
        self.line_freq = line_freq

        # Default configuration
        self._config = {
            "notch_freq": line_freq,
            "bandpass": (1, 50),
            "highpass": None,
            "lowpass": None,
            "remove_artifacts": True,
            "artifact_method": "interpolate",
            "baseline_correct": True,
            "normalize": False,
            "normalize_method": "zscore",
        }

        # Components
        self._filter_bank = FilterBank(fs)
        self._artifact_detector = ArtifactDetector(fs)
        self._quality_assessor = SignalQualityAssessor(fs, line_freq)

        # Processing history
        self._history: list[dict[str, Any]] = []

    def configure(self, **kwargs) -> "Preprocessor":
        """Configure preprocessing parameters.

        Args:
            **kwargs: Configuration options:
                - notch_freq: Notch filter frequency (None to disable)
                - bandpass: (low, high) tuple or None
                - highpass: Highpass cutoff or None
                - lowpass: Lowpass cutoff or None
                - remove_artifacts: Enable artifact removal
                - artifact_method: "interpolate", "zero", or "nan"
                - baseline_correct: Enable baseline correction
                - normalize: Enable normalization
                - normalize_method: "zscore", "minmax", or "robust"

        Returns:
            Self for chaining.
        """
        self._config.update(kwargs)
        return self

    def process(
        self,
        data: np.ndarray,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """Process EEG data through the pipeline.

        Args:
            data: Raw EEG data (channels, samples).
            return_info: If True, return processing info dict.

        Returns:
            Processed data, and optionally processing info.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        info = {
            "input_shape": data.shape,
            "steps": [],
        }

        result = data.copy()

        # Step 1: Notch filter
        if self._config["notch_freq"]:
            result = notch_filter(result, self._config["notch_freq"], self.fs)
            info["steps"].append(f"notch_{self._config['notch_freq']}Hz")

        # Step 2: Bandpass filter
        if self._config["bandpass"]:
            low, high = self._config["bandpass"]
            result = bandpass_filter(result, low, high, self.fs)
            info["steps"].append(f"bandpass_{low}-{high}Hz")

        # Step 3: Highpass filter
        if self._config["highpass"]:
            from .filters import highpass_filter
            result = highpass_filter(result, self._config["highpass"], self.fs)
            info["steps"].append(f"highpass_{self._config['highpass']}Hz")

        # Step 4: Lowpass filter
        if self._config["lowpass"]:
            from .filters import lowpass_filter
            result = lowpass_filter(result, self._config["lowpass"], self.fs)
            info["steps"].append(f"lowpass_{self._config['lowpass']}Hz")

        # Step 5: Artifact removal
        if self._config["remove_artifacts"]:
            self._artifact_detector.detect_all(result)
            result = self._artifact_detector.remove_all(
                result, self._config["artifact_method"]
            )
            artifact_summary = self._artifact_detector.summary()
            info["artifacts"] = artifact_summary
            info["steps"].append(f"artifacts_removed_{artifact_summary['total']}")

        # Step 6: Baseline correction
        if self._config["baseline_correct"]:
            result = baseline_correct(result)
            info["steps"].append("baseline_corrected")

        # Step 7: Normalization
        if self._config["normalize"]:
            result = normalize_signal(result, self._config["normalize_method"])
            info["steps"].append(f"normalized_{self._config['normalize_method']}")

        # Assess output quality
        quality_score, quality_label = self._quality_assessor.get_overall_quality(result)
        info["quality_score"] = quality_score
        info["quality_label"] = quality_label
        info["output_shape"] = result.shape

        # Store in history
        self._history.append(info)

        if return_info:
            return result, info
        return result

    def get_quality_report(self, data: np.ndarray) -> dict[str, Any]:
        """Get quality report for data.

        Args:
            data: EEG data to assess.

        Returns:
            Quality report dictionary.
        """
        return self._quality_assessor.generate_report(data)

    def get_artifact_summary(self) -> dict[str, Any]:
        """Get summary of detected artifacts.

        Returns:
            Artifact summary dictionary.
        """
        return self._artifact_detector.summary()

    @property
    def config(self) -> dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get processing history."""
        return self._history.copy()

    def reset_history(self) -> None:
        """Clear processing history."""
        self._history = []

    def __repr__(self) -> str:
        return (
            f"Preprocessor(fs={self.fs}, "
            f"bandpass={self._config['bandpass']}, "
            f"artifacts={self._config['remove_artifacts']})"
        )
