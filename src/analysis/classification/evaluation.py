"""Evaluation metrics for state classification.

Provides confusion matrix, classification report, and feature importance.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ClassificationReport:
    """Classification evaluation report."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    class_report: str
    classes: list[str]


def evaluate_classifier(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    labels: list[str] | None = None,
) -> ClassificationReport:
    """Evaluate classifier performance.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels.

    Returns:
        Classification report.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for evaluation")

    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique labels if not provided
    if labels is None:
        labels = list(np.unique(np.concatenate([y_true, y_pred])))

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Handle multiclass
    average = "weighted" if len(labels) > 2 else "binary"

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
        class_report=report,
        classes=labels,
    )


def compute_confusion_matrix(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    labels: list[str] | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels.
        normalize: "true", "pred", "all", or None.

    Returns:
        Confusion matrix.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required")

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def compute_feature_importance(
    classifier,
    feature_names: list[str] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Compute and sort feature importance.

    Args:
        classifier: Fitted classifier with feature_importances_.
        feature_names: Feature names.
        top_k: Return only top K features.

    Returns:
        List of (feature_name, importance) tuples, sorted by importance.
    """
    if hasattr(classifier, "get_feature_importance"):
        importance_dict = classifier.get_feature_importance()
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        sorted_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
    else:
        raise ValueError("Classifier does not have feature importances")

    if top_k:
        sorted_features = sorted_features[:top_k]

    return sorted_features


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix.
        labels: Class labels.
        title: Plot title.
        normalize: Normalize values.

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return None

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_importance: list[tuple[str, float]],
    top_k: int = 20,
    title: str = "Feature Importance",
):
    """Plot feature importance.

    Args:
        feature_importance: List of (name, importance) tuples.
        top_k: Number of features to show.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Get top K features
    top_features = feature_importance[:top_k]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def cross_validate_report(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict[str, Any]:
    """Perform cross-validation with detailed report.

    Args:
        classifier: Classifier instance.
        X: Feature matrix.
        y: Labels.
        cv: Number of folds.

    Returns:
        Cross-validation report.
    """
    from sklearn.model_selection import cross_validate

    # Encode labels if needed
    if hasattr(classifier, "_label_encoder"):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y

    # Cross-validate
    cv_results = cross_validate(
        classifier._model if hasattr(classifier, "_model") else classifier,
        X, y_encoded,
        cv=cv,
        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        return_train_score=True,
    )

    return {
        "accuracy": {
            "train": float(np.mean(cv_results["train_accuracy"])),
            "test": float(np.mean(cv_results["test_accuracy"])),
            "std": float(np.std(cv_results["test_accuracy"])),
        },
        "precision": {
            "train": float(np.mean(cv_results["train_precision_weighted"])),
            "test": float(np.mean(cv_results["test_precision_weighted"])),
        },
        "recall": {
            "train": float(np.mean(cv_results["train_recall_weighted"])),
            "test": float(np.mean(cv_results["test_recall_weighted"])),
        },
        "f1": {
            "train": float(np.mean(cv_results["train_f1_weighted"])),
            "test": float(np.mean(cv_results["test_f1_weighted"])),
        },
        "cv_folds": cv,
    }
