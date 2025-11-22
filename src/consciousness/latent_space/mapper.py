"""Consciousness mapper for EEG to embedding space projection.

Maps EEG feature vectors to semantic embedding spaces for
consciousness state analysis and comparison.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class EEGEncoder:
    """Encode EEG features into dense embeddings.

    Uses dimensionality reduction to create compact
    representations of EEG feature vectors.

    Example:
        >>> encoder = EEGEncoder(embedding_dim=128)
        >>> encoder.fit(training_features)
        >>> embeddings = encoder.encode(test_features)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        method: str = "pca",
        normalize: bool = True,
    ):
        """Initialize EEG encoder.

        Args:
            embedding_dim: Output embedding dimension.
            method: Encoding method ("pca" or "autoencoder").
            normalize: Whether to normalize inputs.
        """
        self.embedding_dim = embedding_dim
        self.method = method
        self.normalize = normalize

        self._scaler = StandardScaler() if normalize else None
        self._encoder = None
        self._is_fitted = False

    def fit(self, features: np.ndarray | pd.DataFrame) -> "EEGEncoder":
        """Fit encoder to training features.

        Args:
            features: Feature matrix (samples, features).

        Returns:
            Self for chaining.
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        # Normalize
        if self._scaler:
            features = self._scaler.fit_transform(features)

        # Fit encoder
        if self.method == "pca":
            n_components = min(self.embedding_dim, features.shape[1])
            self._encoder = PCA(n_components=n_components)
            self._encoder.fit(features)

        self._is_fitted = True
        return self

    def encode(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Encode features to embeddings.

        Args:
            features: Feature matrix.

        Returns:
            Embedding matrix (samples, embedding_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted")

        if isinstance(features, pd.DataFrame):
            features = features.values

        # Normalize
        if self._scaler:
            features = self._scaler.transform(features)

        # Encode
        embeddings = self._encoder.transform(features)

        return embeddings

    def fit_encode(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fit and encode in one step.

        Args:
            features: Feature matrix.

        Returns:
            Embedding matrix.
        """
        self.fit(features)
        return self.encode(features)

    @property
    def is_fitted(self) -> bool:
        """Check if encoder is fitted."""
        return self._is_fitted


class SemanticLabeler:
    """Label consciousness states with semantic descriptions.

    Uses sentence transformers to create semantic embeddings
    of state labels for comparison with EEG embeddings.

    Example:
        >>> labeler = SemanticLabeler()
        >>> embeddings = labeler.embed_states(["meditation", "focus", "flow"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic labeler.

        Args:
            model_name: Sentence transformer model name.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

        # Predefined state descriptions
        self._state_descriptions = {
            "meditation": "Calm, relaxed meditation with focused attention on breath",
            "focus": "Deep concentration and focused attention on a task",
            "flow": "Creative flow state with effortless engagement",
            "neutral": "Neutral baseline resting state",
            "anxiety": "Anxious, worried mental state with racing thoughts",
            "creativity": "Creative thinking and imagination",
            "relaxation": "Relaxed, calm mental state",
        }

    def embed_states(
        self,
        states: list[str],
        use_descriptions: bool = True,
    ) -> np.ndarray:
        """Embed state labels into semantic space.

        Args:
            states: List of state names.
            use_descriptions: Use detailed descriptions.

        Returns:
            Embedding matrix (n_states, embedding_dim).
        """
        if use_descriptions:
            texts = [
                self._state_descriptions.get(s, s)
                for s in states
            ]
        else:
            texts = states

        embeddings = self._model.encode(texts)
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return self._model.encode([text])[0]

    def add_state_description(self, state: str, description: str) -> None:
        """Add custom state description.

        Args:
            state: State name.
            description: Semantic description.
        """
        self._state_descriptions[state] = description

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._model.get_sentence_embedding_dimension()


class ConsciousnessMapper:
    """Map EEG features to consciousness embedding space.

    Combines EEG encoding with semantic state labeling
    to create a unified consciousness representation space.

    Example:
        >>> mapper = ConsciousnessMapper(embedding_dim=64)
        >>> mapper.fit(eeg_features, state_labels)
        >>> embeddings = mapper.transform(new_features)
        >>> nearest_state = mapper.find_nearest_state(embeddings[0])
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        use_semantic: bool = True,
        semantic_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize consciousness mapper.

        Args:
            embedding_dim: EEG embedding dimension.
            use_semantic: Use semantic state embeddings.
            semantic_model: Sentence transformer model.
        """
        self.embedding_dim = embedding_dim
        self.use_semantic = use_semantic

        # Components
        self._eeg_encoder = EEGEncoder(embedding_dim)
        self._semantic_labeler = None
        if use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._semantic_labeler = SemanticLabeler(semantic_model)

        # State information
        self._state_embeddings: dict[str, np.ndarray] = {}
        self._state_centroids: dict[str, np.ndarray] = {}
        self._is_fitted = False

    def fit(
        self,
        features: np.ndarray | pd.DataFrame,
        labels: list[str] | np.ndarray,
    ) -> "ConsciousnessMapper":
        """Fit mapper to training data.

        Args:
            features: EEG feature matrix.
            labels: State labels.

        Returns:
            Self for chaining.
        """
        if isinstance(features, pd.DataFrame):
            features = features.values

        labels = np.array(labels)

        # Fit EEG encoder
        self._eeg_encoder.fit(features)

        # Encode all features
        embeddings = self._eeg_encoder.encode(features)

        # Compute state centroids
        unique_states = np.unique(labels)
        for state in unique_states:
            state_mask = labels == state
            state_embeddings = embeddings[state_mask]
            self._state_centroids[state] = np.mean(state_embeddings, axis=0)

        # Get semantic embeddings if available
        if self._semantic_labeler:
            semantic_emb = self._semantic_labeler.embed_states(list(unique_states))
            for i, state in enumerate(unique_states):
                self._state_embeddings[state] = semantic_emb[i]

        self._is_fitted = True
        return self

    def transform(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Transform EEG features to embeddings.

        Args:
            features: EEG feature matrix.

        Returns:
            Embedding matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("Mapper not fitted")

        return self._eeg_encoder.encode(features)

    def fit_transform(
        self,
        features: np.ndarray | pd.DataFrame,
        labels: list[str] | np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            features: EEG features.
            labels: State labels.

        Returns:
            Embeddings.
        """
        self.fit(features, labels)
        return self.transform(features)

    def find_nearest_state(
        self,
        embedding: np.ndarray,
        top_k: int = 1,
    ) -> list[tuple[str, float]]:
        """Find nearest state(s) to an embedding.

        Args:
            embedding: Query embedding.
            top_k: Number of nearest states.

        Returns:
            List of (state, distance) tuples.
        """
        if not self._state_centroids:
            raise RuntimeError("No state centroids available")

        distances = []
        for state, centroid in self._state_centroids.items():
            dist = np.linalg.norm(embedding - centroid)
            distances.append((state, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def compute_state_similarity(
        self,
        embedding: np.ndarray,
    ) -> dict[str, float]:
        """Compute similarity to all states.

        Args:
            embedding: Query embedding.

        Returns:
            Dictionary of state similarities.
        """
        if not self._state_centroids:
            raise RuntimeError("No state centroids available")

        similarities = {}
        for state, centroid in self._state_centroids.items():
            # Cosine similarity
            sim = np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-10
            )
            similarities[state] = float(sim)

        return similarities

    def get_state_centroid(self, state: str) -> np.ndarray | None:
        """Get centroid embedding for a state.

        Args:
            state: State name.

        Returns:
            Centroid embedding or None.
        """
        return self._state_centroids.get(state)

    def save(self, path: str | Path) -> None:
        """Save mapper to file.

        Args:
            path: Output file path.
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "eeg_encoder": self._eeg_encoder,
            "state_centroids": self._state_centroids,
            "state_embeddings": self._state_embeddings,
            "config": {
                "embedding_dim": self.embedding_dim,
                "use_semantic": self.use_semantic,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "ConsciousnessMapper":
        """Load mapper from file.

        Args:
            path: Input file path.

        Returns:
            Loaded mapper.
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        mapper = cls(
            embedding_dim=state["config"]["embedding_dim"],
            use_semantic=state["config"]["use_semantic"],
        )

        mapper._eeg_encoder = state["eeg_encoder"]
        mapper._state_centroids = state["state_centroids"]
        mapper._state_embeddings = state["state_embeddings"]
        mapper._is_fitted = True

        return mapper

    @property
    def is_fitted(self) -> bool:
        """Check if mapper is fitted."""
        return self._is_fitted

    @property
    def states(self) -> list[str]:
        """Get list of known states."""
        return list(self._state_centroids.keys())
