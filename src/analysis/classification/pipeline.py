"""Classification pipeline for EEG analysis.

Provides end-to-end pipeline from raw EEG to state predictions.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..signal import Preprocessor
from ..features import FeatureExtractor
from .classifier import StateClassifier


def create_train_test_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | list,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple:
    """Create train/test split of data.

    Args:
        X: Feature matrix.
        y: Target labels.
        test_size: Fraction for test set.
        random_state: Random seed.
        stratify: Stratify by labels.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    stratify_arr = y if stratify else None

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arr,
    )


class ClassificationPipeline:
    """End-to-end classification pipeline.

    Combines preprocessing, feature extraction, and classification.

    Example:
        >>> pipeline = ClassificationPipeline(fs=256)
        >>> pipeline.fit(eeg_sessions, labels)
        >>> predictions = pipeline.predict(new_eeg)
        >>> pipeline.save("pipeline.pkl")
    """

    def __init__(
        self,
        fs: float,
        channel_names: list[str] | None = None,
        preprocess: bool = True,
        classifier_type: str = "random_forest",
        n_estimators: int = 100,
    ):
        """Initialize classification pipeline.

        Args:
            fs: Sampling frequency in Hz.
            channel_names: Channel names.
            preprocess: Enable preprocessing.
            classifier_type: Classifier type.
            n_estimators: Number of trees.
        """
        self.fs = fs
        self.channel_names = channel_names

        # Components
        self._preprocessor = Preprocessor(fs) if preprocess else None
        self._extractor = FeatureExtractor(fs, channel_names)
        self._classifier = StateClassifier(
            model_type=classifier_type,
            n_estimators=n_estimators,
        )

        # Configure preprocessor
        if self._preprocessor:
            self._preprocessor.configure(
                notch_freq=60,
                bandpass=(1, 50),
                remove_artifacts=True,
                baseline_correct=True,
            )

        self._is_fitted = False

    def fit(
        self,
        eeg_data: list[np.ndarray] | np.ndarray,
        labels: list[str] | np.ndarray,
        epoch_length: float | None = None,
    ) -> "ClassificationPipeline":
        """Fit pipeline to training data.

        Args:
            eeg_data: List of EEG sessions or single session.
                Each session is (channels, samples).
            labels: Labels for each session (or epoch if epoch_length set).
            epoch_length: If set, extract epochs from each session.

        Returns:
            Self for chaining.
        """
        # Extract features from all sessions
        all_features = []

        if isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 2:
            eeg_data = [eeg_data]

        for i, session in enumerate(eeg_data):
            # Preprocess
            if self._preprocessor:
                session = self._preprocessor.process(session)

            # Extract features
            if epoch_length:
                # Extract from epochs
                epoch_features = self._extractor.extract_epoch(
                    session, epoch_length
                )
                all_features.extend(epoch_features)
            else:
                # Extract from whole session
                features = self._extractor.extract(session, flatten=True)
                all_features.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Remove non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        # Handle labels for epochs
        if epoch_length and len(labels) == len(eeg_data):
            # Expand labels to match epochs
            expanded_labels = []
            for i, session in enumerate(eeg_data):
                n_epochs = len([f for f in all_features if f.get("session_idx", i) == i])
                expanded_labels.extend([labels[i]] * n_epochs)
            labels = expanded_labels

        # Fit classifier
        self._classifier.fit(X, labels)
        self._is_fitted = True

        return self

    def predict(
        self,
        eeg_data: np.ndarray,
        epoch_length: float | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Predict state labels.

        Args:
            eeg_data: EEG data (channels, samples).
            epoch_length: If set, predict for epochs.

        Returns:
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")

        # Preprocess
        if self._preprocessor:
            eeg_data = self._preprocessor.process(eeg_data)

        # Extract features
        if epoch_length:
            features_list = self._extractor.extract_epoch(eeg_data, epoch_length)
            df = pd.DataFrame(features_list)
        else:
            features = self._extractor.extract(eeg_data, flatten=True)
            df = pd.DataFrame([features])

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        return self._classifier.predict(X)

    def predict_proba(
        self,
        eeg_data: np.ndarray,
        epoch_length: float | None = None,
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            eeg_data: EEG data.
            epoch_length: Epoch length.

        Returns:
            Probability matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")

        # Preprocess
        if self._preprocessor:
            eeg_data = self._preprocessor.process(eeg_data)

        # Extract features
        if epoch_length:
            features_list = self._extractor.extract_epoch(eeg_data, epoch_length)
            df = pd.DataFrame(features_list)
        else:
            features = self._extractor.extract(eeg_data, flatten=True)
            df = pd.DataFrame([features])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        return self._classifier.predict_proba(X)

    def score(
        self,
        eeg_data: list[np.ndarray],
        labels: list[str],
        epoch_length: float | None = None,
    ) -> float:
        """Compute accuracy score.

        Args:
            eeg_data: List of EEG sessions.
            labels: True labels.
            epoch_length: Epoch length.

        Returns:
            Accuracy score.
        """
        # Extract features
        all_features = []

        for session in eeg_data:
            if self._preprocessor:
                session = self._preprocessor.process(session)

            if epoch_length:
                features_list = self._extractor.extract_epoch(session, epoch_length)
                all_features.extend(features_list)
            else:
                features = self._extractor.extract(session, flatten=True)
                all_features.append(features)

        df = pd.DataFrame(all_features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        # Expand labels if needed
        if epoch_length and len(labels) == len(eeg_data):
            expanded_labels = []
            for i, session in enumerate(eeg_data):
                n_epochs = len(X) // len(eeg_data)  # Approximate
                expanded_labels.extend([labels[i]] * n_epochs)
            labels = expanded_labels[:len(X)]

        return self._classifier.score(X, labels)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance.

        Returns:
            Feature importance dictionary.
        """
        return self._classifier.get_feature_importance()

    def save(self, path: str | Path) -> None:
        """Save pipeline to file.

        Args:
            path: Output file path.
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "fs": self.fs,
            "channel_names": self.channel_names,
            "classifier": self._classifier,
            "preprocessor_config": self._preprocessor.config if self._preprocessor else None,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "ClassificationPipeline":
        """Load pipeline from file.

        Args:
            path: Input file path.

        Returns:
            Loaded pipeline.
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        pipeline = cls(
            fs=state["fs"],
            channel_names=state["channel_names"],
        )

        pipeline._classifier = state["classifier"]
        pipeline._is_fitted = True

        if state["preprocessor_config"]:
            pipeline._preprocessor.configure(**state["preprocessor_config"])

        return pipeline

    @property
    def is_fitted(self) -> bool:
        """Check if pipeline is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"ClassificationPipeline(fs={self.fs}, {status})"
