"""Complexity metrics for consciousness research.

Implements various complexity measures including Lempel-Ziv,
Kolmogorov approximation, fractal dimension, and multi-scale entropy.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics.

    Attributes:
        lempel_ziv: Lempel-Ziv complexity.
        kolmogorov_estimate: Estimated Kolmogorov complexity.
        fractal_dimension: Fractal dimension.
        multiscale_entropy: Multi-scale entropy profile.
        overall: Overall complexity score.
    """

    lempel_ziv: float
    kolmogorov_estimate: float
    fractal_dimension: float
    multiscale_entropy: list[float]
    overall: float


class ComplexityAnalyzer:
    """Analyze signal complexity for consciousness research.

    Computes various complexity metrics that are relevant
    for understanding consciousness and neural dynamics.

    Example:
        >>> analyzer = ComplexityAnalyzer()
        >>> metrics = analyzer.analyze(eeg_data)
        >>> print(f"LZ complexity: {metrics.lempel_ziv:.3f}")
    """

    def __init__(self, n_scales: int = 10):
        """Initialize complexity analyzer.

        Args:
            n_scales: Number of scales for multi-scale entropy.
        """
        self.n_scales = n_scales

    def analyze(self, data: np.ndarray) -> ComplexityMetrics:
        """Compute all complexity metrics.

        Args:
            data: Signal data (1D or channels x samples).

        Returns:
            ComplexityMetrics object.
        """
        # Flatten if multi-channel
        if data.ndim > 1:
            signal = data.flatten()
        else:
            signal = data

        # Compute individual metrics
        lz = compute_lempel_ziv_complexity(signal)
        kolmogorov = estimate_kolmogorov_complexity(signal)
        fractal = compute_fractal_dimension(signal)
        mse = compute_multiscale_entropy(signal, self.n_scales)

        # Overall complexity (weighted average)
        overall = 0.3 * lz + 0.2 * kolmogorov + 0.2 * fractal + 0.3 * np.mean(mse)

        return ComplexityMetrics(
            lempel_ziv=lz,
            kolmogorov_estimate=kolmogorov,
            fractal_dimension=fractal,
            multiscale_entropy=mse,
            overall=float(overall),
        )

    def analyze_channels(
        self,
        data: np.ndarray,
    ) -> dict[str, ComplexityMetrics]:
        """Analyze complexity per channel.

        Args:
            data: Multi-channel data (channels, samples).

        Returns:
            Dictionary mapping channel index to metrics.
        """
        results = {}

        for i in range(data.shape[0]):
            metrics = self.analyze(data[i])
            results[f"channel_{i}"] = metrics

        return results


def compute_lempel_ziv_complexity(
    signal: np.ndarray,
    threshold: str = "median",
) -> float:
    """Compute Lempel-Ziv complexity.

    LZ complexity measures the number of distinct patterns
    in a binary sequence derived from the signal.

    Args:
        signal: Input signal.
        threshold: Thresholding method ("median", "mean").

    Returns:
        Normalized LZ complexity (0-1).
    """
    # Binarize signal
    if threshold == "median":
        thresh = np.median(signal)
    else:
        thresh = np.mean(signal)

    binary = (signal > thresh).astype(int)

    # Compute LZ complexity
    n = len(binary)
    if n == 0:
        return 0.0

    # Convert to string for pattern matching
    s = ''.join(map(str, binary))

    # LZ76 algorithm
    complexity = 1
    i = 0
    k = 1
    k_max = 1
    substring = set()
    substring.add(s[0])

    while i + k <= n:
        if s[i:i + k] not in substring:
            substring.add(s[i:i + k])
            complexity += 1
            i = i + k_max
            k = 1
            k_max = 1
        else:
            k += 1
            if k > k_max:
                k_max = k

    # Normalize by theoretical maximum
    b = n / np.log2(n + 1e-10) if n > 1 else 1
    normalized = complexity / b

    return float(np.clip(normalized, 0, 1))


def estimate_kolmogorov_complexity(
    signal: np.ndarray,
    precision: int = 8,
) -> float:
    """Estimate Kolmogorov complexity via compression.

    Uses zlib compression ratio as a proxy for
    Kolmogorov complexity.

    Args:
        signal: Input signal.
        precision: Quantization precision (bits).

    Returns:
        Normalized Kolmogorov complexity estimate.
    """
    import zlib

    # Quantize signal
    min_val = np.min(signal)
    max_val = np.max(signal)

    if max_val == min_val:
        return 0.0

    normalized = (signal - min_val) / (max_val - min_val)
    quantized = (normalized * (2**precision - 1)).astype(np.uint8)

    # Compress
    original_size = len(quantized)
    compressed = zlib.compress(quantized.tobytes(), level=9)
    compressed_size = len(compressed)

    # Compression ratio as complexity estimate
    ratio = compressed_size / original_size

    return float(np.clip(ratio, 0, 1))


def compute_fractal_dimension(
    signal: np.ndarray,
    k_max: int = 10,
) -> float:
    """Compute Higuchi fractal dimension.

    Measures the complexity of the signal's shape
    using the Higuchi method.

    Args:
        signal: Input signal.
        k_max: Maximum interval.

    Returns:
        Fractal dimension (typically 1-2).
    """
    n = len(signal)

    if n < k_max * 2:
        k_max = n // 2

    if k_max < 1:
        return 1.0

    # Compute curve lengths for different k
    lengths = []
    ks = []

    for k in range(1, k_max + 1):
        Lk = []

        for m in range(1, k + 1):
            # Compute length for this (k, m) pair
            Lmk = 0
            max_idx = (n - m) // k

            if max_idx < 1:
                continue

            for i in range(1, max_idx + 1):
                Lmk += abs(signal[m + i * k - 1] - signal[m + (i - 1) * k - 1])

            Lmk = (Lmk * (n - 1)) / (k * max_idx * k)
            Lk.append(Lmk)

        if Lk:
            lengths.append(np.mean(Lk))
            ks.append(k)

    if len(ks) < 2:
        return 1.0

    # Linear regression in log-log space
    log_ks = np.log(ks)
    log_lengths = np.log(np.array(lengths) + 1e-10)

    # Slope is the fractal dimension
    slope, _ = np.polyfit(log_ks, log_lengths, 1)

    return float(np.clip(-slope, 1, 2))


def compute_multiscale_entropy(
    signal: np.ndarray,
    n_scales: int = 10,
    m: int = 2,
    r: float | None = None,
) -> list[float]:
    """Compute multi-scale entropy.

    Sample entropy computed at multiple time scales
    to capture complexity at different resolutions.

    Args:
        signal: Input signal.
        n_scales: Number of scales.
        m: Embedding dimension.
        r: Tolerance (default: 0.15 * std).

    Returns:
        List of entropy values at each scale.
    """
    if r is None:
        r = 0.15 * np.std(signal)

    mse = []

    for scale in range(1, n_scales + 1):
        # Coarse-grain the signal
        n_coarse = len(signal) // scale
        if n_coarse < m + 1:
            mse.append(0.0)
            continue

        coarse = np.zeros(n_coarse)
        for i in range(n_coarse):
            coarse[i] = np.mean(signal[i * scale:(i + 1) * scale])

        # Compute sample entropy
        se = _sample_entropy(coarse, m, r)
        mse.append(se)

    return mse


def _sample_entropy(
    signal: np.ndarray,
    m: int,
    r: float,
) -> float:
    """Compute sample entropy.

    Args:
        signal: Input signal.
        m: Embedding dimension.
        r: Tolerance.

    Returns:
        Sample entropy value.
    """
    n = len(signal)

    if n < m + 1:
        return 0.0

    def _count_matches(templates: np.ndarray, r: float) -> int:
        count = 0
        n_templates = len(templates)

        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    # Create templates of length m
    templates_m = np.array([signal[i:i + m] for i in range(n - m)])
    count_m = _count_matches(templates_m, r)

    # Create templates of length m+1
    templates_m1 = np.array([signal[i:i + m + 1] for i in range(n - m - 1)])
    count_m1 = _count_matches(templates_m1, r)

    if count_m == 0 or count_m1 == 0:
        return 0.0

    return -np.log(count_m1 / count_m)


def compute_permutation_entropy(
    signal: np.ndarray,
    m: int = 3,
    delay: int = 1,
) -> float:
    """Compute permutation entropy.

    Measures complexity based on ordinal patterns.

    Args:
        signal: Input signal.
        m: Embedding dimension.
        delay: Time delay.

    Returns:
        Normalized permutation entropy (0-1).
    """
    from math import factorial

    n = len(signal)
    n_patterns = n - (m - 1) * delay

    if n_patterns < 1:
        return 0.0

    # Extract ordinal patterns
    pattern_counts: dict[tuple, int] = {}

    for i in range(n_patterns):
        # Extract embedding vector
        indices = [i + j * delay for j in range(m)]
        values = signal[indices]

        # Get ordinal pattern (rank ordering)
        pattern = tuple(np.argsort(values).tolist())
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Compute entropy
    total = sum(pattern_counts.values())
    entropy = 0.0

    for count in pattern_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    # Normalize by maximum entropy
    max_entropy = np.log2(factorial(m))
    normalized = entropy / max_entropy if max_entropy > 0 else 0

    return float(normalized)


def compute_approximate_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: float | None = None,
) -> float:
    """Compute approximate entropy.

    Similar to sample entropy but includes self-matches.

    Args:
        signal: Input signal.
        m: Embedding dimension.
        r: Tolerance (default: 0.2 * std).

    Returns:
        Approximate entropy value.
    """
    n = len(signal)

    if r is None:
        r = 0.2 * np.std(signal)

    if n < m + 1:
        return 0.0

    def _phi(m: int) -> float:
        templates = np.array([signal[i:i + m] for i in range(n - m + 1)])
        n_templates = len(templates)

        if n_templates == 0:
            return 0.0

        counts = np.zeros(n_templates)

        for i in range(n_templates):
            for j in range(n_templates):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    counts[i] += 1

        # Average log count
        return np.mean(np.log(counts / n_templates + 1e-10))

    return _phi(m) - _phi(m + 1)


def compute_spectral_entropy(
    signal: np.ndarray,
    fs: float = 256.0,
) -> float:
    """Compute spectral entropy.

    Measures the flatness of the power spectrum.

    Args:
        signal: Input signal.
        fs: Sampling frequency.

    Returns:
        Normalized spectral entropy (0-1).
    """
    # Compute power spectrum
    n = len(signal)
    fft = np.fft.rfft(signal)
    psd = np.abs(fft) ** 2

    # Normalize to probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-10)

    # Compute entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Normalize by maximum entropy
    max_entropy = np.log2(len(psd))
    normalized = entropy / max_entropy if max_entropy > 0 else 0

    return float(normalized)
