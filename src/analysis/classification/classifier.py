"""State classifier for EEG-based consciousness state classification.

Provides Random Forest classifier with training and prediction.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Standard consciousness state labels
STATE_LABELS = [
    "neutral",
    "flow",
    "meditation",
    "focus",
    "creativity",
    "anxiety",
]


class StateClassifier:
    """Consciousness state classifier using Random Forest.

    Classifies EEG features into consciousness states.

    Example:
        >>> classifier = StateClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
        >>> classifier.save("model.pkl")
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
        normalize: bool = True,
    ):
        """Initialize state classifier.

        Args:
            model_type: "random_forest" or "gradient_boosting".
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            random_state: Random seed.
            normalize: Whether to normalize features.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )

        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.normalize = normalize

        # Initialize model
        if model_type == "random_forest":
            self._model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            self._model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 3,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Preprocessors
        self._scaler = StandardScaler() if normalize else None
        self._label_encoder = LabelEncoder()

        # State
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._classes: list[str] = []

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | list[str],
    ) -> "StateClassifier":
        """Fit classifier to training data.

        Args:
            X: Feature matrix (samples, features).
            y: Target labels (samples,).

        Returns:
            Self for chaining.
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values

        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)
        self._classes = list(self._label_encoder.classes_)

        # Normalize features
        if self._scaler:
            X = self._scaler.fit_transform(X)

        # Fit model
        self._model.fit(X, y_encoded)
        self._is_fitted = True

        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Predict state labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self._scaler:
            X = self._scaler.transform(X)

        y_pred = self._model.predict(X)

        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix (samples, classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self._scaler:
            X = self._scaler.transform(X)

        return self._model.predict_proba(X)

    def score(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | list[str],
    ) -> float:
        """Compute accuracy score.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Accuracy score.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted")

        if isinstance(X, pd.DataFrame):
            X = X.values

        y_encoded = self._label_encoder.transform(y)

        if self._scaler:
            X = self._scaler.transform(X)

        return self._model.score(X, y_encoded)

    def cross_validate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | list[str],
        cv: int = 5,
    ) -> dict[str, float]:
        """Perform cross-validation.

        Args:
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds.

        Returns:
            Dictionary with mean and std scores.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        y_encoded = self._label_encoder.fit_transform(y)
        self._classes = list(self._label_encoder.classes_)

        if self._scaler:
            X = self._scaler.fit_transform(X)

        scores = cross_val_score(self._model, X, y_encoded, cv=cv)

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted")

        importances = self._model.feature_importances_

        if self._feature_names:
            return dict(zip(self._feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def save(self, path: str | Path) -> None:
        """Save classifier to file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self._model,
            "scaler": self._scaler,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "classes": self._classes,
            "config": {
                "model_type": self.model_type,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "normalize": self.normalize,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "StateClassifier":
        """Load classifier from file.

        Args:
            path: Input file path.

        Returns:
            Loaded classifier.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        classifier = cls(
            model_type=config["model_type"],
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            normalize=config["normalize"],
        )

        classifier._model = state["model"]
        classifier._scaler = state["scaler"]
        classifier._label_encoder = state["label_encoder"]
        classifier._feature_names = state["feature_names"]
        classifier._classes = state["classes"]
        classifier._is_fitted = True

        return classifier

    @property
    def classes(self) -> list[str]:
        """Get class labels."""
        return self._classes

    @property
    def is_fitted(self) -> bool:
        """Check if classifier is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"StateClassifier(type={self.model_type}, {status})"
