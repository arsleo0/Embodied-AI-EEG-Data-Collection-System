"""Integrated Information Theory (IIT) metrics.

Implements approximations of Phi and other IIT-inspired
metrics for consciousness research.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IITMetrics:
    """Integrated Information Theory metrics.

    Attributes:
        phi: Integrated information (Phi) approximation.
        integration: Information integration score.
        differentiation: Information differentiation score.
        exclusion: Exclusion measure.
        composition: Compositional structure score.
    """

    phi: float
    integration: float
    differentiation: float
    exclusion: float
    composition: float


class IITAnalyzer:
    """Analyze neural data using IIT-inspired metrics.

    Computes approximations of Integrated Information Theory
    measures that are computationally tractable.

    Example:
        >>> analyzer = IITAnalyzer()
        >>> metrics = analyzer.analyze(eeg_data)
        >>> print(f"Phi: {metrics.phi:.3f}")
    """

    def __init__(self, n_bins: int = 10):
        """Initialize IIT analyzer.

        Args:
            n_bins: Number of bins for discretization.
        """
        self.n_bins = n_bins

    def analyze(self, data: np.ndarray) -> IITMetrics:
        """Compute IIT metrics from neural data.

        Args:
            data: EEG data (channels, samples).

        Returns:
            IITMetrics object.
        """
        # Compute individual metrics
        integration = self._compute_integration(data)
        differentiation = self._compute_differentiation(data)
        exclusion = self._compute_exclusion(data)
        composition = self._compute_composition(data)

        # Approximate Phi as combination of metrics
        # (True Phi is computationally intractable)
        phi = integration * differentiation * (1 - exclusion)

        return IITMetrics(
            phi=float(phi),
            integration=float(integration),
            differentiation=float(differentiation),
            exclusion=float(exclusion),
            composition=float(composition),
        )

    def _compute_integration(self, data: np.ndarray) -> float:
        """Compute information integration.

        Integration measures how much information the whole
        system has beyond its parts.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Integration score (0-1).
        """
        n_channels = data.shape[0]

        if n_channels < 2:
            return 0.0

        # Compute mutual information between all pairs
        total_mi = 0
        n_pairs = 0

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                mi = self._mutual_information(data[i], data[j])
                total_mi += mi
                n_pairs += 1

        if n_pairs == 0:
            return 0.0

        # Normalize
        mean_mi = total_mi / n_pairs
        max_mi = np.log2(self.n_bins)

        return float(np.clip(mean_mi / max_mi, 0, 1))

    def _compute_differentiation(self, data: np.ndarray) -> float:
        """Compute information differentiation.

        Differentiation measures the diversity of states
        the system can be in.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Differentiation score (0-1).
        """
        # Compute entropy of the joint distribution
        # High entropy = high differentiation

        # Discretize each channel
        discretized = np.zeros_like(data, dtype=int)
        for i in range(data.shape[0]):
            min_val, max_val = np.min(data[i]), np.max(data[i])
            if max_val > min_val:
                discretized[i] = np.clip(
                    ((data[i] - min_val) / (max_val - min_val) * (self.n_bins - 1)).astype(int),
                    0, self.n_bins - 1
                )

        # Joint state index
        n_channels = data.shape[0]
        joint_states = np.zeros(data.shape[1], dtype=int)

        for i in range(n_channels):
            joint_states += discretized[i] * (self.n_bins ** i)

        # Count unique states
        unique, counts = np.unique(joint_states, return_counts=True)
        n_states = len(unique)

        # Entropy
        probs = counts / np.sum(counts)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by maximum possible entropy
        max_states = self.n_bins ** min(n_channels, 4)  # Limit to avoid overflow
        max_entropy = np.log2(max_states)

        return float(np.clip(entropy / max_entropy, 0, 1))

    def _compute_exclusion(self, data: np.ndarray) -> float:
        """Compute exclusion measure.

        Exclusion measures redundancy - how much information
        is duplicated across parts.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Exclusion score (0-1, higher = more redundancy).
        """
        n_channels = data.shape[0]

        if n_channels < 2:
            return 0.0

        # Compute redundancy via correlation
        corr_matrix = np.corrcoef(data)
        np.fill_diagonal(corr_matrix, 0)

        # Mean absolute correlation as redundancy
        redundancy = np.mean(np.abs(corr_matrix))

        return float(np.clip(redundancy, 0, 1))

    def _compute_composition(self, data: np.ndarray) -> float:
        """Compute compositional structure.

        Measures hierarchical organization of information.

        Args:
            data: Neural data (channels, samples).

        Returns:
            Composition score (0-1).
        """
        n_channels = data.shape[0]

        if n_channels < 2:
            return 0.5

        # Analyze at different levels of grouping
        # Compare information at different scales

        # Individual channel entropy
        individual_entropy = 0
        for i in range(n_channels):
            h = self._entropy_1d(data[i])
            individual_entropy += h
        individual_entropy /= n_channels

        # Pair-wise entropy
        pair_entropy = 0
        n_pairs = 0
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                h = self._joint_entropy(data[i], data[j])
                pair_entropy += h
                n_pairs += 1

        if n_pairs > 0:
            pair_entropy /= n_pairs

        # Composition = how much pair entropy exceeds individual
        # High composition = complex hierarchical structure
        if individual_entropy > 0:
            composition = min(pair_entropy / (2 * individual_entropy), 1.0)
        else:
            composition = 0.5

        return float(composition)

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two signals.

        Args:
            x: First signal.
            y: Second signal.

        Returns:
            Mutual information in bits.
        """
        # Discretize
        x_disc = self._discretize(x)
        y_disc = self._discretize(y)

        # Joint histogram
        joint_hist = np.zeros((self.n_bins, self.n_bins))
        for i in range(len(x_disc)):
            joint_hist[x_disc[i], y_disc[i]] += 1

        # Marginals
        x_hist = np.sum(joint_hist, axis=1)
        y_hist = np.sum(joint_hist, axis=0)

        # Probabilities
        total = np.sum(joint_hist)
        p_xy = joint_hist / total
        p_x = x_hist / total
        p_y = y_hist / total

        # Mutual information
        mi = 0
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return max(0, mi)

    def _entropy_1d(self, x: np.ndarray) -> float:
        """Compute entropy of single signal.

        Args:
            x: Signal array.

        Returns:
            Entropy in bits.
        """
        x_disc = self._discretize(x)
        hist = np.bincount(x_disc, minlength=self.n_bins)
        probs = hist / np.sum(hist)

        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _joint_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute joint entropy of two signals.

        Args:
            x: First signal.
            y: Second signal.

        Returns:
            Joint entropy in bits.
        """
        x_disc = self._discretize(x)
        y_disc = self._discretize(y)

        # Joint histogram
        joint_hist = np.zeros((self.n_bins, self.n_bins))
        for i in range(len(x_disc)):
            joint_hist[x_disc[i], y_disc[i]] += 1

        probs = joint_hist.flatten() / np.sum(joint_hist)

        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize signal into bins.

        Args:
            x: Continuous signal.

        Returns:
            Discretized signal.
        """
        min_val, max_val = np.min(x), np.max(x)

        if max_val == min_val:
            return np.zeros(len(x), dtype=int)

        normalized = (x - min_val) / (max_val - min_val)
        discretized = np.clip(
            (normalized * (self.n_bins - 1)).astype(int),
            0, self.n_bins - 1
        )

        return discretized


def compute_phi_approximation(
    data: np.ndarray,
    method: str = "geometric_mean",
) -> float:
    """Compute approximation of Phi.

    True Phi (Φ) is computationally intractable for
    large systems, so we use approximations.

    Args:
        data: Neural data (channels, samples).
        method: Approximation method.

    Returns:
        Phi approximation.
    """
    analyzer = IITAnalyzer()
    metrics = analyzer.analyze(data)

    if method == "geometric_mean":
        return metrics.phi
    elif method == "integration_only":
        return metrics.integration
    elif method == "balanced":
        return (metrics.integration + metrics.differentiation) / 2
    else:
        return metrics.phi


def compute_causal_density(
    data: np.ndarray,
    lag: int = 1,
) -> float:
    """Compute causal density (simplified).

    Measures the density of causal connections
    between variables.

    Args:
        data: Neural data (channels, samples).
        lag: Time lag for causality.

    Returns:
        Causal density (0-1).
    """
    n_channels = data.shape[0]
    n_samples = data.shape[1]

    if n_channels < 2 or n_samples <= lag:
        return 0.0

    # Simplified Granger-like causality
    causal_connections = 0
    total_pairs = 0

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                continue

            # Test if channel i predicts channel j
            x = data[i, :-lag]
            y = data[j, lag:]

            # Correlation as simplified causality measure
            corr = np.abs(np.corrcoef(x, y)[0, 1])

            if not np.isnan(corr) and corr > 0.3:  # Threshold
                causal_connections += 1

            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return float(causal_connections / total_pairs)


def compute_integration_differentiation_balance(
    data: np.ndarray,
) -> dict[str, float]:
    """Compute balance between integration and differentiation.

    Consciousness requires both integration (unified)
    and differentiation (diverse).

    Args:
        data: Neural data (channels, samples).

    Returns:
        Balance metrics.
    """
    analyzer = IITAnalyzer()
    metrics = analyzer.analyze(data)

    # Balance = how close to optimal (high both)
    # Use geometric mean as balance indicator
    balance = np.sqrt(metrics.integration * metrics.differentiation)

    return {
        "integration": metrics.integration,
        "differentiation": metrics.differentiation,
        "balance": float(balance),
        "dominance": "integration" if metrics.integration > metrics.differentiation else "differentiation",
    }
