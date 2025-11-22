"""Signal processing module for EEG preprocessing.

This module provides tools for cleaning and preprocessing EEG signals:
- Filtering (bandpass, notch, highpass)
- Artifact detection and removal
- Signal quality assessment
- Baseline correction
"""

from .filters import (
    bandpass_filter,
    notch_filter,
    highpass_filter,
    lowpass_filter,
    FilterBank,
)
from .artifacts import (
    ArtifactDetector,
    detect_eye_blinks,
    detect_muscle_artifacts,
    remove_artifacts,
)
from .quality import (
    SignalQualityAssessor,
    compute_snr,
    compute_flatline_ratio,
)
from .preprocessing import (
    Preprocessor,
    baseline_correct,
    normalize_signal,
    resample_signal,
)

__all__ = [
    # Filters
    "bandpass_filter",
    "notch_filter",
    "highpass_filter",
    "lowpass_filter",
    "FilterBank",
    # Artifacts
    "ArtifactDetector",
    "detect_eye_blinks",
    "detect_muscle_artifacts",
    "remove_artifacts",
    # Quality
    "SignalQualityAssessor",
    "compute_snr",
    "compute_flatline_ratio",
    # Preprocessing
    "Preprocessor",
    "baseline_correct",
    "normalize_signal",
    "resample_signal",
]
