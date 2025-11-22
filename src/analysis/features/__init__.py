"""Feature extraction module for EEG analysis.

This module provides comprehensive feature extraction:
- Power spectral density (PSD)
- Band power calculation
- Statistical features
- Entropy measures
- Hjorth parameters
- Coherence between channels
"""

from .spectral import (
    compute_psd,
    compute_band_power,
    compute_relative_band_power,
    compute_band_ratios,
    SpectralFeatures,
)
from .entropy import (
    spectral_entropy,
    sample_entropy,
    approximate_entropy,
    permutation_entropy,
    EntropyFeatures,
)
from .statistical import (
    compute_statistical_features,
    compute_hjorth_parameters,
    StatisticalFeatures,
)
from .connectivity import (
    compute_coherence,
    compute_correlation,
    ConnectivityFeatures,
)
from .extractor import FeatureExtractor

__all__ = [
    # Spectral
    "compute_psd",
    "compute_band_power",
    "compute_relative_band_power",
    "compute_band_ratios",
    "SpectralFeatures",
    # Entropy
    "spectral_entropy",
    "sample_entropy",
    "approximate_entropy",
    "permutation_entropy",
    "EntropyFeatures",
    # Statistical
    "compute_statistical_features",
    "compute_hjorth_parameters",
    "StatisticalFeatures",
    # Connectivity
    "compute_coherence",
    "compute_correlation",
    "ConnectivityFeatures",
    # Main extractor
    "FeatureExtractor",
]
