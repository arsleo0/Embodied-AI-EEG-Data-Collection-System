"""Cross-modal mapping between EEG and sensory modalities.

Maps EEG features to audio, visual, and text representations
for cross-modal consciousness research.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModalityMapping:
    """Result of modality mapping.

    Attributes:
        source_modality: Source modality name.
        target_modality: Target modality name.
        features: Mapped features.
        confidence: Mapping confidence.
        metadata: Additional metadata.
    """

    source_modality: str
    target_modality: str
    features: np.ndarray
    confidence: float
    metadata: dict[str, Any]


class ModalityMapper(ABC):
    """Abstract base class for modality mappers."""

    @abstractmethod
    def fit(self, source: np.ndarray, target: np.ndarray) -> "ModalityMapper":
        """Fit mapper on paired data.

        Args:
            source: Source modality data.
            target: Target modality data.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source to target modality.

        Args:
            source: Source modality data.

        Returns:
            Predicted target modality features.
        """
        pass

    def fit_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            source: Source modality data.
            target: Target modality data.

        Returns:
            Transformed data.
        """
        self.fit(source, target)
        return self.transform(source)


class EEGToAudioMapper(ModalityMapper):
    """Map EEG features to audio features.

    Maps neural signals to audio representations including
    spectral features, rhythm, and timbre characteristics.

    Example:
        >>> mapper = EEGToAudioMapper()
        >>> mapper.fit(eeg_features, audio_features)
        >>> audio = mapper.transform(new_eeg)
    """

    def __init__(
        self,
        n_audio_features: int = 64,
        regularization: float = 1.0,
    ):
        """Initialize EEG to audio mapper.

        Args:
            n_audio_features: Number of output audio features.
            regularization: Ridge regression regularization.
        """
        self.n_audio_features = n_audio_features
        self.regularization = regularization

        self._model = None
        self._scaler_source = None
        self._scaler_target = None
        self._is_fitted = False

        if SKLEARN_AVAILABLE:
            self._scaler_source = StandardScaler()
            self._scaler_target = StandardScaler()
            self._model = Ridge(alpha=regularization)

    def fit(self, source: np.ndarray, target: np.ndarray) -> "EEGToAudioMapper":
        """Fit mapper on paired EEG-audio data.

        Args:
            source: EEG features (samples, eeg_features).
            target: Audio features (samples, audio_features).

        Returns:
            Self for chaining.
        """
        if not SKLEARN_AVAILABLE:
            # Simple linear mapping fallback
            self._weight_matrix = np.random.randn(
                source.shape[1], target.shape[1]
            ) * 0.01
            self._is_fitted = True
            return self

        # Scale features
        source_scaled = self._scaler_source.fit_transform(source)
        target_scaled = self._scaler_target.fit_transform(target)

        # Fit regression model
        self._model.fit(source_scaled, target_scaled)

        self._is_fitted = True
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform EEG to audio features.

        Args:
            source: EEG features.

        Returns:
            Predicted audio features.
        """
        if not self._is_fitted:
            raise RuntimeError("Mapper not fitted")

        if not SKLEARN_AVAILABLE:
            return source @ self._weight_matrix

        # Scale and transform
        if source.ndim == 1:
            source = source.reshape(1, -1)

        source_scaled = self._scaler_source.transform(source)
        target_scaled = self._model.predict(source_scaled)
        target = self._scaler_target.inverse_transform(target_scaled)

        return target.squeeze()

    def generate_audio_params(
        self,
        eeg_features: np.ndarray,
    ) -> dict[str, Any]:
        """Generate audio synthesis parameters from EEG.

        Args:
            eeg_features: EEG feature vector.

        Returns:
            Audio synthesis parameters.
        """
        # Map to audio feature space
        if self._is_fitted:
            audio_features = self.transform(eeg_features)
        else:
            audio_features = eeg_features[:self.n_audio_features] if len(eeg_features) > self.n_audio_features else eeg_features

        # Extract interpretable parameters
        if len(audio_features) >= 8:
            params = {
                "fundamental_freq": 100 + 400 * sigmoid(audio_features[0]),
                "spectral_centroid": 500 + 2000 * sigmoid(audio_features[1]),
                "spectral_bandwidth": 100 + 900 * sigmoid(audio_features[2]),
                "spectral_rolloff": 1000 + 4000 * sigmoid(audio_features[3]),
                "zero_crossing_rate": 0.01 + 0.19 * sigmoid(audio_features[4]),
                "rms_energy": 0.1 + 0.9 * sigmoid(audio_features[5]),
                "tempo": 40 + 160 * sigmoid(audio_features[6]),
                "harmony": sigmoid(audio_features[7]),
            }
        else:
            # Default parameters
            params = {
                "fundamental_freq": 220,
                "spectral_centroid": 1000,
                "spectral_bandwidth": 500,
                "spectral_rolloff": 2000,
                "zero_crossing_rate": 0.05,
                "rms_energy": 0.5,
                "tempo": 100,
                "harmony": 0.5,
            }

        return params


class EEGToVisualMapper(ModalityMapper):
    """Map EEG features to visual features.

    Maps neural signals to visual representations including
    color, texture, shape, and motion characteristics.

    Example:
        >>> mapper = EEGToVisualMapper()
        >>> mapper.fit(eeg_features, visual_features)
        >>> visual = mapper.transform(new_eeg)
    """

    def __init__(
        self,
        n_visual_features: int = 64,
        regularization: float = 1.0,
    ):
        """Initialize EEG to visual mapper.

        Args:
            n_visual_features: Number of output visual features.
            regularization: Ridge regression regularization.
        """
        self.n_visual_features = n_visual_features
        self.regularization = regularization

        self._model = None
        self._scaler_source = None
        self._scaler_target = None
        self._is_fitted = False

        if SKLEARN_AVAILABLE:
            self._scaler_source = StandardScaler()
            self._scaler_target = StandardScaler()
            self._model = Ridge(alpha=regularization)

    def fit(self, source: np.ndarray, target: np.ndarray) -> "EEGToVisualMapper":
        """Fit mapper on paired EEG-visual data.

        Args:
            source: EEG features (samples, eeg_features).
            target: Visual features (samples, visual_features).

        Returns:
            Self for chaining.
        """
        if not SKLEARN_AVAILABLE:
            self._weight_matrix = np.random.randn(
                source.shape[1], target.shape[1]
            ) * 0.01
            self._is_fitted = True
            return self

        source_scaled = self._scaler_source.fit_transform(source)
        target_scaled = self._scaler_target.fit_transform(target)
        self._model.fit(source_scaled, target_scaled)

        self._is_fitted = True
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform EEG to visual features.

        Args:
            source: EEG features.

        Returns:
            Predicted visual features.
        """
        if not self._is_fitted:
            raise RuntimeError("Mapper not fitted")

        if not SKLEARN_AVAILABLE:
            return source @ self._weight_matrix

        if source.ndim == 1:
            source = source.reshape(1, -1)

        source_scaled = self._scaler_source.transform(source)
        target_scaled = self._model.predict(source_scaled)
        target = self._scaler_target.inverse_transform(target_scaled)

        return target.squeeze()

    def generate_visual_params(
        self,
        eeg_features: np.ndarray,
    ) -> dict[str, Any]:
        """Generate visual synthesis parameters from EEG.

        Args:
            eeg_features: EEG feature vector.

        Returns:
            Visual synthesis parameters.
        """
        if self._is_fitted:
            visual_features = self.transform(eeg_features)
        else:
            visual_features = eeg_features[:self.n_visual_features] if len(eeg_features) > self.n_visual_features else eeg_features

        if len(visual_features) >= 10:
            params = {
                "hue": 360 * sigmoid(visual_features[0]),
                "saturation": sigmoid(visual_features[1]),
                "brightness": 0.2 + 0.8 * sigmoid(visual_features[2]),
                "contrast": 0.5 + 0.5 * sigmoid(visual_features[3]),
                "edge_density": sigmoid(visual_features[4]),
                "texture_roughness": sigmoid(visual_features[5]),
                "motion_speed": sigmoid(visual_features[6]),
                "motion_direction": 360 * sigmoid(visual_features[7]),
                "complexity": sigmoid(visual_features[8]),
                "symmetry": sigmoid(visual_features[9]),
            }
        else:
            params = {
                "hue": 180,
                "saturation": 0.5,
                "brightness": 0.7,
                "contrast": 0.5,
                "edge_density": 0.5,
                "texture_roughness": 0.3,
                "motion_speed": 0.5,
                "motion_direction": 0,
                "complexity": 0.5,
                "symmetry": 0.5,
            }

        return params


class EEGToTextMapper(ModalityMapper):
    """Map EEG features to text embeddings/descriptions.

    Maps neural signals to semantic text representations
    for natural language description of consciousness states.

    Example:
        >>> mapper = EEGToTextMapper()
        >>> mapper.fit(eeg_features, text_embeddings)
        >>> text_emb = mapper.transform(new_eeg)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        regularization: float = 1.0,
    ):
        """Initialize EEG to text mapper.

        Args:
            embedding_dim: Text embedding dimension.
            regularization: Ridge regression regularization.
        """
        self.embedding_dim = embedding_dim
        self.regularization = regularization

        self._model = None
        self._scaler_source = None
        self._is_fitted = False

        if SKLEARN_AVAILABLE:
            self._scaler_source = StandardScaler()
            self._model = Ridge(alpha=regularization)

        # State templates for text generation
        self._state_templates = {
            "high_alpha": [
                "The mind settles into a calm, relaxed awareness.",
                "A state of peaceful contemplation emerges.",
                "Relaxed alertness with gentle mental clarity.",
            ],
            "high_beta": [
                "Active mental processing with focused attention.",
                "Engaged cognitive activity and concentration.",
                "Alert and mentally active state.",
            ],
            "high_theta": [
                "Deep introspective awareness and memory processing.",
                "Dreamlike state with creative associations.",
                "Meditative depth with intuitive insights.",
            ],
            "high_gamma": [
                "Heightened perception and information binding.",
                "Peak cognitive integration and awareness.",
                "Unified conscious experience with clarity.",
            ],
            "balanced": [
                "Balanced mental state with steady awareness.",
                "Equilibrium between relaxation and alertness.",
                "Neutral, grounded consciousness.",
            ],
        }

    def fit(self, source: np.ndarray, target: np.ndarray) -> "EEGToTextMapper":
        """Fit mapper on paired EEG-text embedding data.

        Args:
            source: EEG features (samples, eeg_features).
            target: Text embeddings (samples, embedding_dim).

        Returns:
            Self for chaining.
        """
        if not SKLEARN_AVAILABLE:
            self._weight_matrix = np.random.randn(
                source.shape[1], target.shape[1]
            ) * 0.01
            self._is_fitted = True
            return self

        source_scaled = self._scaler_source.fit_transform(source)
        self._model.fit(source_scaled, target)

        self._is_fitted = True
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform EEG to text embedding.

        Args:
            source: EEG features.

        Returns:
            Predicted text embedding.
        """
        if not self._is_fitted:
            raise RuntimeError("Mapper not fitted")

        if not SKLEARN_AVAILABLE:
            return source @ self._weight_matrix

        if source.ndim == 1:
            source = source.reshape(1, -1)

        source_scaled = self._scaler_source.transform(source)
        embedding = self._model.predict(source_scaled)

        return embedding.squeeze()

    def generate_description(
        self,
        eeg_features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate text description from EEG features.

        Args:
            eeg_features: EEG feature vector.
            feature_names: Optional feature names for interpretation.

        Returns:
            Text description and metadata.
        """
        # Analyze dominant patterns
        dominant_state = self._identify_dominant_state(eeg_features, feature_names)

        # Get template
        templates = self._state_templates.get(dominant_state, self._state_templates["balanced"])

        # Select based on feature variance
        idx = int(np.var(eeg_features) * 10) % len(templates)
        description = templates[idx]

        return {
            "description": description,
            "dominant_state": dominant_state,
            "confidence": float(np.max(np.abs(eeg_features))),
        }

    def _identify_dominant_state(
        self,
        features: np.ndarray,
        names: list[str] | None,
    ) -> str:
        """Identify dominant brain state from features.

        Args:
            features: Feature vector.
            names: Feature names.

        Returns:
            Dominant state identifier.
        """
        if names is None:
            # Use simple heuristics
            if len(features) >= 4:
                # Assume features are band powers
                max_idx = np.argmax(features[:4])
                states = ["high_theta", "high_alpha", "high_beta", "high_gamma"]
                return states[max_idx] if max_idx < len(states) else "balanced"
            return "balanced"

        # Look for band power features
        band_powers = {}
        for i, name in enumerate(names):
            name_lower = name.lower()
            if "alpha" in name_lower:
                band_powers["alpha"] = features[i]
            elif "beta" in name_lower:
                band_powers["beta"] = features[i]
            elif "theta" in name_lower:
                band_powers["theta"] = features[i]
            elif "gamma" in name_lower:
                band_powers["gamma"] = features[i]

        if not band_powers:
            return "balanced"

        # Find dominant band
        dominant = max(band_powers, key=band_powers.get)
        return f"high_{dominant}"


class CrossModalTranslator:
    """Translate between any two modalities.

    Generic translator supporting EEG, audio, visual, and text.

    Example:
        >>> translator = CrossModalTranslator()
        >>> translator.fit("eeg", eeg_data, "audio", audio_data)
        >>> audio = translator.translate("eeg", new_eeg, "audio")
    """

    def __init__(self, regularization: float = 1.0):
        """Initialize cross-modal translator.

        Args:
            regularization: Regularization for regression models.
        """
        self.regularization = regularization
        self._mappers: dict[tuple[str, str], ModalityMapper] = {}

    def fit(
        self,
        source_modality: str,
        source_data: np.ndarray,
        target_modality: str,
        target_data: np.ndarray,
    ) -> "CrossModalTranslator":
        """Fit translator for a modality pair.

        Args:
            source_modality: Source modality name.
            source_data: Source data.
            target_modality: Target modality name.
            target_data: Target data.

        Returns:
            Self for chaining.
        """
        # Create appropriate mapper
        key = (source_modality, target_modality)

        if source_modality == "eeg" and target_modality == "audio":
            mapper = EEGToAudioMapper(
                n_audio_features=target_data.shape[1],
                regularization=self.regularization,
            )
        elif source_modality == "eeg" and target_modality == "visual":
            mapper = EEGToVisualMapper(
                n_visual_features=target_data.shape[1],
                regularization=self.regularization,
            )
        elif source_modality == "eeg" and target_modality == "text":
            mapper = EEGToTextMapper(
                embedding_dim=target_data.shape[1],
                regularization=self.regularization,
            )
        else:
            # Generic mapper
            mapper = GenericMapper(
                output_dim=target_data.shape[1],
                regularization=self.regularization,
            )

        mapper.fit(source_data, target_data)
        self._mappers[key] = mapper

        return self

    def translate(
        self,
        source_modality: str,
        source_data: np.ndarray,
        target_modality: str,
    ) -> ModalityMapping:
        """Translate from source to target modality.

        Args:
            source_modality: Source modality name.
            source_data: Source data.
            target_modality: Target modality name.

        Returns:
            ModalityMapping with translated features.
        """
        key = (source_modality, target_modality)

        if key not in self._mappers:
            raise ValueError(
                f"No mapper fitted for {source_modality} -> {target_modality}"
            )

        mapper = self._mappers[key]
        features = mapper.transform(source_data)

        # Estimate confidence from mapping quality
        if hasattr(mapper, '_model') and mapper._model is not None:
            try:
                r2 = mapper._model.score(
                    mapper._scaler_source.transform(source_data.reshape(1, -1)),
                    features.reshape(1, -1)
                )
                confidence = max(0, r2)
            except Exception:
                confidence = 0.5
        else:
            confidence = 0.5

        return ModalityMapping(
            source_modality=source_modality,
            target_modality=target_modality,
            features=features,
            confidence=confidence,
            metadata={
                "mapper_type": type(mapper).__name__,
            },
        )

    def get_supported_translations(self) -> list[tuple[str, str]]:
        """Get list of supported translations.

        Returns:
            List of (source, target) modality pairs.
        """
        return list(self._mappers.keys())


class GenericMapper(ModalityMapper):
    """Generic linear mapper between modalities."""

    def __init__(self, output_dim: int, regularization: float = 1.0):
        """Initialize generic mapper.

        Args:
            output_dim: Output dimension.
            regularization: Regularization parameter.
        """
        self.output_dim = output_dim
        self.regularization = regularization

        self._model = None
        self._scaler_source = None
        self._is_fitted = False

        if SKLEARN_AVAILABLE:
            self._scaler_source = StandardScaler()
            self._model = Ridge(alpha=regularization)

    def fit(self, source: np.ndarray, target: np.ndarray) -> "GenericMapper":
        """Fit generic mapper.

        Args:
            source: Source data.
            target: Target data.

        Returns:
            Self for chaining.
        """
        if not SKLEARN_AVAILABLE:
            self._weight_matrix = np.random.randn(
                source.shape[1], target.shape[1]
            ) * 0.01
            self._is_fitted = True
            return self

        source_scaled = self._scaler_source.fit_transform(source)
        self._model.fit(source_scaled, target)
        self._is_fitted = True
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source to target.

        Args:
            source: Source data.

        Returns:
            Transformed data.
        """
        if not self._is_fitted:
            raise RuntimeError("Mapper not fitted")

        if not SKLEARN_AVAILABLE:
            return source @ self._weight_matrix

        if source.ndim == 1:
            source = source.reshape(1, -1)

        source_scaled = self._scaler_source.transform(source)
        return self._model.predict(source_scaled).squeeze()


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid activation function.

    Args:
        x: Input value(s).

    Returns:
        Sigmoid output(s).
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
