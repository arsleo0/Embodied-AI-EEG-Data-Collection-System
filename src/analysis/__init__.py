"""Analysis layer - Signal processing and feature extraction.

Modules:
    signal: EEG signal processing (filtering, artifact removal)
    features: Feature extraction (PSD, coherence, entropy)
    classification: State classification models
"""

from .signal import (
    Preprocessor,
    FilterBank,
    ArtifactDetector,
    SignalQualityAssessor,
    bandpass_filter,
    notch_filter,
)
from .features import (
    FeatureExtractor,
    SpectralFeatures,
    EntropyFeatures,
    StatisticalFeatures,
    compute_psd,
    compute_band_power,
)
from .classification import (
    StateClassifier,
    ClassificationPipeline,
    evaluate_classifier,
    create_train_test_split,
)

__all__ = [
    # Signal processing
    "Preprocessor",
    "FilterBank",
    "ArtifactDetector",
    "SignalQualityAssessor",
    "bandpass_filter",
    "notch_filter",
    # Feature extraction
    "FeatureExtractor",
    "SpectralFeatures",
    "EntropyFeatures",
    "StatisticalFeatures",
    "compute_psd",
    "compute_band_power",
    # Classification
    "StateClassifier",
    "ClassificationPipeline",
    "evaluate_classifier",
    "create_train_test_split",
]
