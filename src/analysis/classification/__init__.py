"""State classification module for EEG analysis.

This module provides tools for classifying consciousness states:
- Random Forest classifier
- Training pipeline
- Model persistence
- Evaluation metrics
"""

from .classifier import (
    StateClassifier,
    STATE_LABELS,
)
from .pipeline import (
    ClassificationPipeline,
    create_train_test_split,
)
from .evaluation import (
    evaluate_classifier,
    compute_confusion_matrix,
    compute_feature_importance,
    ClassificationReport,
)

__all__ = [
    # Classifier
    "StateClassifier",
    "STATE_LABELS",
    # Pipeline
    "ClassificationPipeline",
    "create_train_test_split",
    # Evaluation
    "evaluate_classifier",
    "compute_confusion_matrix",
    "compute_feature_importance",
    "ClassificationReport",
]
