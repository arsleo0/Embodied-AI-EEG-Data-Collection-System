"""Entropy measures for EEG signals.

Provides various entropy computations:
- Spectral entropy
- Sample entropy
- Approximate entropy
- Permutation entropy
"""

from typing import Any

import numpy as np
from scipy import signal, stats


def spectral_entropy(
    data: np.ndarray,
    fs: float,
    normalize: bool = True,
) -> float | np.ndarray:
    """Compute spectral entropy.

    Spectral entropy measures the "flatness" of the power spectrum.
    Higher values indicate more uniform frequency distribution.

    Args:
        data: Signal array (samples,) or (channels, samples).
        fs: Sampling frequency in Hz.
        normalize: If True, normalize to [0, 1].

    Returns:
        Spectral entropy value(s).
    """
    if data.ndim == 1:
        return _spectral_entropy_1d(data, fs, normalize)
    else:
        return np.array([_spectral_entropy_1d(ch, fs, normalize) for ch in data])


def _spectral_entropy_1d(
    data: np.ndarray,
    fs: float,
    normalize: bool,
) -> float:
    """Compute spectral entropy for 1D signal."""
    # Compute PSD
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))

    # Normalize PSD to probability distribution
    psd_norm = psd / np.sum(psd)

    # Remove zeros for log
    psd_norm = psd_norm[psd_norm > 0]

    # Compute entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm))

    if normalize:
        entropy = entropy / np.log2(len(psd_norm))

    return entropy


def sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r: float | None = None,
) -> float | np.ndarray:
    """Compute sample entropy.

    Sample entropy measures signal complexity/regularity.
    Lower values indicate more regular patterns.

    Args:
        data: Signal array (samples,) or (channels, samples).
        m: Embedding dimension.
        r: Tolerance (default: 0.2 * std).

    Returns:
        Sample entropy value(s).
    """
    if data.ndim == 1:
        return _sample_entropy_1d(data, m, r)
    else:
        return np.array([_sample_entropy_1d(ch, m, r) for ch in data])


def _sample_entropy_1d(
    data: np.ndarray,
    m: int,
    r: float | None,
) -> float:
    """Compute sample entropy for 1D signal."""
    N = len(data)

    if r is None:
        r = 0.2 * np.std(data)

    # Count matching patterns
    def count_matches(template_length):
        count = 0
        templates = np.array([
            data[i:i + template_length]
            for i in range(N - template_length)
        ])

        for i in range(len(templates)):
            # Chebyshev distance
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            # Count matches (excluding self)
            count += np.sum(dists <= r) - 1

        return count

    # Count for m and m+1
    A = count_matches(m + 1)
    B = count_matches(m)

    if B == 0:
        return 0.0

    return -np.log(A / B) if A > 0 else 0.0


def approximate_entropy(
    data: np.ndarray,
    m: int = 2,
    r: float | None = None,
) -> float | np.ndarray:
    """Compute approximate entropy.

    Similar to sample entropy but includes self-matches.

    Args:
        data: Signal array (samples,) or (channels, samples).
        m: Embedding dimension.
        r: Tolerance (default: 0.2 * std).

    Returns:
        Approximate entropy value(s).
    """
    if data.ndim == 1:
        return _approximate_entropy_1d(data, m, r)
    else:
        return np.array([_approximate_entropy_1d(ch, m, r) for ch in data])


def _approximate_entropy_1d(
    data: np.ndarray,
    m: int,
    r: float | None,
) -> float:
    """Compute approximate entropy for 1D signal."""
    N = len(data)

    if r is None:
        r = 0.2 * np.std(data)

    def phi(template_length):
        templates = np.array([
            data[i:i + template_length]
            for i in range(N - template_length + 1)
        ])

        C = np.zeros(len(templates))
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            C[i] = np.sum(dists <= r) / len(templates)

        return np.mean(np.log(C[C > 0]))

    return phi(m) - phi(m + 1)


def permutation_entropy(
    data: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float | np.ndarray:
    """Compute permutation entropy.

    Measures complexity based on ordinal patterns.

    Args:
        data: Signal array (samples,) or (channels, samples).
        order: Embedding dimension (pattern length).
        delay: Time delay between samples.
        normalize: If True, normalize to [0, 1].

    Returns:
        Permutation entropy value(s).
    """
    if data.ndim == 1:
        return _permutation_entropy_1d(data, order, delay, normalize)
    else:
        return np.array([
            _permutation_entropy_1d(ch, order, delay, normalize)
            for ch in data
        ])


def _permutation_entropy_1d(
    data: np.ndarray,
    order: int,
    delay: int,
    normalize: bool,
) -> float:
    """Compute permutation entropy for 1D signal."""
    from itertools import permutations

    N = len(data)
    n_patterns = N - (order - 1) * delay

    if n_patterns <= 0:
        return 0.0

    # Generate all permutations
    perms = list(permutations(range(order)))
    perm_dict = {perm: i for i, perm in enumerate(perms)}

    # Count pattern occurrences
    counts = np.zeros(len(perms))

    for i in range(n_patterns):
        # Extract embedded pattern
        idx = np.arange(order) * delay + i
        pattern = data[idx]

        # Get ordinal pattern (ranks)
        ranks = tuple(np.argsort(np.argsort(pattern)))

        if ranks in perm_dict:
            counts[perm_dict[ranks]] += 1

    # Compute entropy
    probs = counts / n_patterns
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    if normalize:
        max_entropy = np.log2(np.math.factorial(order))
        entropy = entropy / max_entropy

    return entropy


class EntropyFeatures:
    """Entropy feature extractor for EEG signals.

    Computes multiple entropy measures.

    Example:
        >>> ef = EntropyFeatures(fs=256)
        >>> features = ef.extract(eeg_data)
        >>> print(features["sample_entropy"])
    """

    def __init__(
        self,
        fs: float,
        sample_entropy_m: int = 2,
        permutation_order: int = 3,
    ):
        """Initialize entropy feature extractor.

        Args:
            fs: Sampling frequency in Hz.
            sample_entropy_m: Embedding dimension for sample entropy.
            permutation_order: Order for permutation entropy.
        """
        self.fs = fs
        self.sample_entropy_m = sample_entropy_m
        self.permutation_order = permutation_order

    def extract(self, data: np.ndarray) -> dict[str, Any]:
        """Extract all entropy features.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Dictionary of features.
        """
        features = {}

        # Spectral entropy
        features["spectral_entropy"] = spectral_entropy(data, self.fs)

        # Sample entropy
        features["sample_entropy"] = sample_entropy(
            data, m=self.sample_entropy_m
        )

        # Approximate entropy
        features["approximate_entropy"] = approximate_entropy(
            data, m=self.sample_entropy_m
        )

        # Permutation entropy
        features["permutation_entropy"] = permutation_entropy(
            data, order=self.permutation_order
        )

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names.
        """
        return [
            "spectral_entropy",
            "sample_entropy",
            "approximate_entropy",
            "permutation_entropy",
        ]
