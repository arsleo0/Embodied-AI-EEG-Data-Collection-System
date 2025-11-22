"""Uncertainty quantification for consciousness analysis.

Quantifies different types of uncertainty in consciousness
state predictions and neural signal interpretation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty metrics.

    Attributes:
        epistemic: Uncertainty from model limitations.
        aleatoric: Uncertainty from data variability.
        total: Combined uncertainty.
        confidence_interval: Confidence bounds.
        entropy: Predictive entropy.
    """

    epistemic: float
    aleatoric: float
    total: float
    confidence_interval: tuple[float, float]
    entropy: float

    @property
    def reliability(self) -> float:
        """Reliability score (inverse of total uncertainty)."""
        return max(0, 1 - self.total)


class UncertaintyQuantifier:
    """Quantify uncertainty in consciousness predictions.

    Separates epistemic uncertainty (from limited knowledge)
    from aleatoric uncertainty (from inherent variability).

    Example:
        >>> quantifier = UncertaintyQuantifier()
        >>> metrics = quantifier.quantify(predictions, features)
        >>> print(f"Total uncertainty: {metrics.total:.3f}")
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
    ):
        """Initialize uncertainty quantifier.

        Args:
            n_bootstrap: Number of bootstrap samples.
            confidence_level: Confidence level for intervals.
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    def quantify(
        self,
        predictions: np.ndarray,
        features: np.ndarray | None = None,
        true_labels: np.ndarray | None = None,
    ) -> UncertaintyMetrics:
        """Quantify uncertainty in predictions.

        Args:
            predictions: Model predictions (samples,) or (samples, classes).
            features: Input features for aleatoric estimation.
            true_labels: True labels for calibration.

        Returns:
            UncertaintyMetrics object.
        """
        # Epistemic uncertainty (model uncertainty)
        epistemic = self._compute_epistemic(predictions)

        # Aleatoric uncertainty (data uncertainty)
        aleatoric = self._compute_aleatoric(predictions, features)

        # Total uncertainty
        total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

        # Confidence interval
        ci = self._compute_confidence_interval(predictions)

        # Predictive entropy
        entropy = self._compute_entropy(predictions)

        return UncertaintyMetrics(
            epistemic=float(epistemic),
            aleatoric=float(aleatoric),
            total=float(total),
            confidence_interval=ci,
            entropy=float(entropy),
        )

    def _compute_epistemic(self, predictions: np.ndarray) -> float:
        """Compute epistemic uncertainty.

        Uses bootstrap variance as estimate of model uncertainty.

        Args:
            predictions: Model predictions.

        Returns:
            Epistemic uncertainty.
        """
        if predictions.ndim == 1:
            # Regression or single-class probability
            bootstrap_means = []

            for _ in range(self.n_bootstrap):
                indices = np.random.choice(
                    len(predictions),
                    size=len(predictions),
                    replace=True
                )
                bootstrap_means.append(np.mean(predictions[indices]))

            return np.std(bootstrap_means)

        else:
            # Classification with probabilities
            # Variance of predicted probabilities
            return np.mean(np.std(predictions, axis=0))

    def _compute_aleatoric(
        self,
        predictions: np.ndarray,
        features: np.ndarray | None,
    ) -> float:
        """Compute aleatoric uncertainty.

        Estimates inherent data variability from prediction
        distribution or feature variance.

        Args:
            predictions: Model predictions.
            features: Input features.

        Returns:
            Aleatoric uncertainty.
        """
        if predictions.ndim == 1:
            # Use prediction variance as aleatoric estimate
            aleatoric = np.std(predictions)
        else:
            # For classification: average entropy of individual predictions
            # High entropy = high aleatoric uncertainty
            eps = 1e-10
            entropies = -np.sum(
                predictions * np.log(predictions + eps),
                axis=1
            )
            aleatoric = np.mean(entropies)

        # Incorporate feature variability if available
        if features is not None:
            feature_var = np.mean(np.std(features, axis=0))
            aleatoric = 0.5 * aleatoric + 0.5 * feature_var

        return aleatoric

    def _compute_confidence_interval(
        self,
        predictions: np.ndarray,
    ) -> tuple[float, float]:
        """Compute confidence interval.

        Args:
            predictions: Model predictions.

        Returns:
            (lower, upper) confidence bounds.
        """
        if predictions.ndim == 1:
            values = predictions
        else:
            # Use predicted class probabilities
            values = np.max(predictions, axis=1)

        alpha = 1 - self.confidence_level
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def _compute_entropy(self, predictions: np.ndarray) -> float:
        """Compute predictive entropy.

        Args:
            predictions: Model predictions.

        Returns:
            Entropy value.
        """
        if predictions.ndim == 1:
            # Normalize to probability distribution
            probs = predictions / (np.sum(predictions) + 1e-10)
        else:
            # Mean prediction
            probs = np.mean(predictions, axis=0)

        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps))

        return entropy

    def quantify_temporal(
        self,
        predictions_over_time: list[np.ndarray],
        window_size: int = 10,
    ) -> list[UncertaintyMetrics]:
        """Quantify uncertainty over time.

        Args:
            predictions_over_time: List of predictions.
            window_size: Window for temporal analysis.

        Returns:
            List of uncertainty metrics.
        """
        metrics = []

        for i in range(len(predictions_over_time)):
            start = max(0, i - window_size + 1)
            window = predictions_over_time[start:i + 1]

            if len(window) > 0:
                stacked = np.vstack(window)
                metric = self.quantify(stacked)
                metrics.append(metric)

        return metrics

    def calibrate(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, Any]:
        """Calibrate uncertainty estimates against true labels.

        Args:
            predictions: Model predictions.
            true_labels: Ground truth labels.

        Returns:
            Calibration metrics.
        """
        if predictions.ndim == 1:
            # Regression calibration
            errors = np.abs(predictions - true_labels)
            uncertainties = np.std(predictions) * np.ones_like(predictions)

            # Correlation between error and uncertainty
            correlation = np.corrcoef(errors, uncertainties)[0, 1]

            return {
                "correlation": float(correlation) if not np.isnan(correlation) else 0,
                "mean_error": float(np.mean(errors)),
                "calibration_score": float(1 - np.mean(np.abs(errors - uncertainties))),
            }
        else:
            # Classification calibration
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            # Accuracy per confidence bin
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_accuracies = []
            bin_confidences = []

            for i in range(n_bins):
                mask = (confidences >= bin_boundaries[i]) & (
                    confidences < bin_boundaries[i + 1]
                )
                if np.any(mask):
                    bin_acc = np.mean(predicted_classes[mask] == true_labels[mask])
                    bin_conf = np.mean(confidences[mask])
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)

            # Expected calibration error
            ece = np.mean(np.abs(
                np.array(bin_accuracies) - np.array(bin_confidences)
            )) if bin_accuracies else 0

            return {
                "expected_calibration_error": float(ece),
                "bin_accuracies": bin_accuracies,
                "bin_confidences": bin_confidences,
                "overall_accuracy": float(np.mean(predicted_classes == true_labels)),
            }


def compute_epistemic_uncertainty(
    predictions: np.ndarray,
    n_bootstrap: int = 100,
) -> float:
    """Compute epistemic (model) uncertainty.

    Args:
        predictions: Model predictions.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Epistemic uncertainty.
    """
    quantifier = UncertaintyQuantifier(n_bootstrap=n_bootstrap)
    return quantifier._compute_epistemic(predictions)


def compute_aleatoric_uncertainty(
    predictions: np.ndarray,
    features: np.ndarray | None = None,
) -> float:
    """Compute aleatoric (data) uncertainty.

    Args:
        predictions: Model predictions.
        features: Input features.

    Returns:
        Aleatoric uncertainty.
    """
    quantifier = UncertaintyQuantifier()
    return quantifier._compute_aleatoric(predictions, features)


def compute_prediction_entropy(predictions: np.ndarray) -> float:
    """Compute entropy of predictions.

    Args:
        predictions: Model predictions.

    Returns:
        Predictive entropy.
    """
    quantifier = UncertaintyQuantifier()
    return quantifier._compute_entropy(predictions)


def compute_uncertainty_decomposition(
    predictions: np.ndarray,
    features: np.ndarray | None = None,
) -> dict[str, float]:
    """Decompose total uncertainty into components.

    Args:
        predictions: Model predictions.
        features: Input features.

    Returns:
        Dictionary with uncertainty components.
    """
    quantifier = UncertaintyQuantifier()
    metrics = quantifier.quantify(predictions, features)

    return {
        "epistemic": metrics.epistemic,
        "aleatoric": metrics.aleatoric,
        "total": metrics.total,
        "entropy": metrics.entropy,
        "reliability": metrics.reliability,
    }
