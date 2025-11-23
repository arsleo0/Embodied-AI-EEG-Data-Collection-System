"""Qualia synthesis and cross-modal fusion.

Synthesizes cross-modal representations of consciousness states,
enabling synesthesia-like translations between modalities.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class QualiaRepresentation:
    """Multi-modal qualia representation.

    Attributes:
        embedding: Unified cross-modal embedding.
        modality_weights: Contribution of each modality.
        emotional_valence: Emotional valence (-1 to 1).
        arousal: Arousal level (0 to 1).
        dominance: Dominance level (0 to 1).
        sensory_mapping: Mapped sensory features.
    """

    embedding: np.ndarray
    modality_weights: dict[str, float]
    emotional_valence: float
    arousal: float
    dominance: float
    sensory_mapping: dict[str, Any]


class CrossModalFusion:
    """Fuse embeddings from multiple modalities.

    Combines EEG, audio, visual, and text embeddings into
    a unified consciousness representation.

    Example:
        >>> fusion = CrossModalFusion(output_dim=128)
        >>> unified = fusion.fuse({
        ...     "eeg": eeg_embedding,
        ...     "audio": audio_embedding,
        ... })
    """

    def __init__(
        self,
        output_dim: int = 128,
        fusion_method: str = "weighted_concat",
    ):
        """Initialize cross-modal fusion.

        Args:
            output_dim: Output embedding dimension.
            fusion_method: Fusion method ("weighted_concat", "attention", "gated").
        """
        self.output_dim = output_dim
        self.fusion_method = fusion_method

        self._projectors: dict[str, Any] = {}
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._pca = None
        self._is_fitted = False

        # Default modality weights
        self.modality_weights = {
            "eeg": 0.4,
            "audio": 0.2,
            "visual": 0.2,
            "text": 0.2,
        }

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set modality weights.

        Args:
            weights: Dictionary of modality weights.
        """
        total = sum(weights.values())
        self.modality_weights = {k: v / total for k, v in weights.items()}

    def fit(
        self,
        modality_embeddings: dict[str, np.ndarray],
    ) -> "CrossModalFusion":
        """Fit fusion model on multi-modal data.

        Args:
            modality_embeddings: Dictionary mapping modality to embeddings.

        Returns:
            Self for chaining.
        """
        # Concatenate all modalities
        all_embeddings = []
        for modality, embeddings in modality_embeddings.items():
            if embeddings is not None and len(embeddings) > 0:
                all_embeddings.append(embeddings)

        if not all_embeddings:
            raise ValueError("No valid embeddings provided")

        # Pad to same length if needed
        max_samples = max(e.shape[0] for e in all_embeddings)
        padded = []
        for emb in all_embeddings:
            if emb.shape[0] < max_samples:
                pad_size = max_samples - emb.shape[0]
                emb = np.vstack([emb, np.zeros((pad_size, emb.shape[1]))])
            padded.append(emb)

        concatenated = np.hstack(padded)

        # Fit scaler and PCA
        if self._scaler is not None:
            concatenated = self._scaler.fit_transform(concatenated)

        if SKLEARN_AVAILABLE:
            self._pca = PCA(n_components=min(self.output_dim, concatenated.shape[1]))
            self._pca.fit(concatenated)

        self._is_fitted = True
        return self

    def fuse(
        self,
        modality_embeddings: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Fuse multi-modal embeddings.

        Args:
            modality_embeddings: Dictionary mapping modality to embedding.
            weights: Optional custom weights.

        Returns:
            Fused embedding.
        """
        weights = weights or self.modality_weights

        if self.fusion_method == "weighted_concat":
            return self._weighted_concat(modality_embeddings, weights)
        elif self.fusion_method == "attention":
            return self._attention_fusion(modality_embeddings, weights)
        elif self.fusion_method == "gated":
            return self._gated_fusion(modality_embeddings, weights)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _weighted_concat(
        self,
        modality_embeddings: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Weighted concatenation fusion.

        Args:
            modality_embeddings: Modality embeddings.
            weights: Modality weights.

        Returns:
            Fused embedding.
        """
        weighted_parts = []

        for modality, embedding in modality_embeddings.items():
            if embedding is not None:
                weight = weights.get(modality, 0.25)
                weighted_parts.append(embedding * weight)

        if not weighted_parts:
            return np.zeros(self.output_dim)

        # Concatenate and reduce dimension
        concatenated = np.concatenate(weighted_parts)

        if self._pca is not None and self._is_fitted:
            # Pad or truncate to match PCA input
            expected_dim = self._pca.n_features_in_
            if len(concatenated) < expected_dim:
                concatenated = np.pad(
                    concatenated,
                    (0, expected_dim - len(concatenated))
                )
            elif len(concatenated) > expected_dim:
                concatenated = concatenated[:expected_dim]

            if self._scaler is not None:
                concatenated = self._scaler.transform([concatenated])[0]
            return self._pca.transform([concatenated])[0]

        # Simple truncation/padding
        if len(concatenated) > self.output_dim:
            return concatenated[:self.output_dim]
        else:
            return np.pad(concatenated, (0, self.output_dim - len(concatenated)))

    def _attention_fusion(
        self,
        modality_embeddings: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Attention-based fusion.

        Args:
            modality_embeddings: Modality embeddings.
            weights: Initial modality weights.

        Returns:
            Fused embedding.
        """
        # Stack embeddings
        embeddings = []
        modalities = []

        for modality, embedding in modality_embeddings.items():
            if embedding is not None:
                embeddings.append(embedding)
                modalities.append(modality)

        if not embeddings:
            return np.zeros(self.output_dim)

        # Compute attention scores based on embedding norms
        norms = [np.linalg.norm(e) for e in embeddings]
        attention = np.array(norms) / (sum(norms) + 1e-10)

        # Weight by both attention and preset weights
        final_weights = []
        for i, modality in enumerate(modalities):
            w = attention[i] * weights.get(modality, 0.25)
            final_weights.append(w)

        # Normalize
        final_weights = np.array(final_weights) / (sum(final_weights) + 1e-10)

        # Weighted sum (pad to same dimension first)
        max_dim = max(len(e) for e in embeddings)
        padded = [np.pad(e, (0, max_dim - len(e))) for e in embeddings]
        fused = sum(w * e for w, e in zip(final_weights, padded))

        # Reduce to output dimension
        if len(fused) > self.output_dim:
            return fused[:self.output_dim]
        return np.pad(fused, (0, self.output_dim - len(fused)))

    def _gated_fusion(
        self,
        modality_embeddings: dict[str, np.ndarray],
        weights: dict[str, float],
    ) -> np.ndarray:
        """Gated fusion with learned gates.

        Args:
            modality_embeddings: Modality embeddings.
            weights: Modality weights as gate values.

        Returns:
            Fused embedding.
        """
        # Similar to attention but with sigmoid gates
        embeddings = []
        gates = []

        for modality, embedding in modality_embeddings.items():
            if embedding is not None:
                embeddings.append(embedding)
                # Sigmoid-like gate based on weight
                gate = 1 / (1 + np.exp(-5 * (weights.get(modality, 0.25) - 0.25)))
                gates.append(gate)

        if not embeddings:
            return np.zeros(self.output_dim)

        # Normalize gates
        gates = np.array(gates) / (sum(gates) + 1e-10)

        # Gated sum
        max_dim = max(len(e) for e in embeddings)
        padded = [np.pad(e, (0, max_dim - len(e))) for e in embeddings]
        fused = sum(g * e for g, e in zip(gates, padded))

        if len(fused) > self.output_dim:
            return fused[:self.output_dim]
        return np.pad(fused, (0, self.output_dim - len(fused)))


class SynesthesiaSimulator:
    """Simulate synesthetic experiences from EEG.

    Maps consciousness states to cross-modal sensory experiences,
    e.g., "What does anxiety look like?" or "What does flow sound like?"

    Example:
        >>> simulator = SynesthesiaSimulator()
        >>> visual = simulator.state_to_visual("anxiety", eeg_features)
        >>> audio = simulator.state_to_audio("flow", eeg_features)
    """

    def __init__(self):
        """Initialize synesthesia simulator."""
        # State → Color mappings (HSV-inspired)
        self.state_colors = {
            "anxiety": {"hue": 0, "saturation": 0.9, "brightness": 0.8},      # Red
            "calm": {"hue": 200, "saturation": 0.5, "brightness": 0.7},       # Blue
            "focus": {"hue": 120, "saturation": 0.7, "brightness": 0.8},      # Green
            "flow": {"hue": 280, "saturation": 0.6, "brightness": 0.9},       # Purple
            "meditation": {"hue": 180, "saturation": 0.4, "brightness": 0.6}, # Cyan
            "excitement": {"hue": 45, "saturation": 0.9, "brightness": 0.95}, # Yellow/Orange
        }

        # State → Sound mappings (frequency, timbre characteristics)
        self.state_sounds = {
            "anxiety": {"base_freq": 440, "harmonics": "sharp", "tempo": 140},
            "calm": {"base_freq": 220, "harmonics": "smooth", "tempo": 60},
            "focus": {"base_freq": 330, "harmonics": "clear", "tempo": 90},
            "flow": {"base_freq": 396, "harmonics": "rich", "tempo": 100},
            "meditation": {"base_freq": 174, "harmonics": "deep", "tempo": 40},
            "excitement": {"base_freq": 528, "harmonics": "bright", "tempo": 120},
        }

        # State → Texture mappings
        self.state_textures = {
            "anxiety": {"roughness": 0.9, "density": 0.8, "movement": "erratic"},
            "calm": {"roughness": 0.2, "density": 0.3, "movement": "flowing"},
            "focus": {"roughness": 0.4, "density": 0.6, "movement": "directed"},
            "flow": {"roughness": 0.3, "density": 0.7, "movement": "smooth"},
            "meditation": {"roughness": 0.1, "density": 0.2, "movement": "still"},
            "excitement": {"roughness": 0.6, "density": 0.9, "movement": "pulsing"},
        }

    def state_to_visual(
        self,
        state: str,
        eeg_features: np.ndarray | None = None,
        resolution: tuple[int, int] = (256, 256),
    ) -> dict[str, Any]:
        """Convert state to visual representation.

        Args:
            state: Consciousness state name.
            eeg_features: Optional EEG features for modulation.
            resolution: Output image resolution.

        Returns:
            Visual parameters and optional image data.
        """
        # Get base color
        color = self.state_colors.get(state, {
            "hue": 0, "saturation": 0.5, "brightness": 0.5
        })

        # Modulate with EEG features if provided
        if eeg_features is not None:
            # Use feature variance to modulate saturation
            variance = np.var(eeg_features)
            color["saturation"] = np.clip(
                color["saturation"] + 0.2 * (variance - 0.5),
                0, 1
            )

            # Use feature mean to modulate brightness
            mean = np.mean(np.abs(eeg_features))
            color["brightness"] = np.clip(
                color["brightness"] + 0.1 * (mean - 0.5),
                0, 1
            )

        # Get texture
        texture = self.state_textures.get(state, {
            "roughness": 0.5, "density": 0.5, "movement": "random"
        })

        # Generate simple visual pattern
        visual_data = self._generate_visual_pattern(
            color, texture, resolution, eeg_features
        )

        return {
            "color": color,
            "texture": texture,
            "pattern": visual_data,
            "resolution": resolution,
            "state": state,
        }

    def _generate_visual_pattern(
        self,
        color: dict[str, float],
        texture: dict[str, Any],
        resolution: tuple[int, int],
        features: np.ndarray | None,
    ) -> np.ndarray:
        """Generate visual pattern array.

        Args:
            color: Color parameters.
            texture: Texture parameters.
            resolution: Image resolution.
            features: Optional features for modulation.

        Returns:
            Pattern array (H, W, 3) in RGB.
        """
        h, w = resolution

        # Create base gradient
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Add noise based on roughness
        noise = np.random.randn(h, w) * texture["roughness"] * 0.3

        # Create pattern based on movement type
        if texture["movement"] == "erratic":
            pattern = np.sin(10 * xx + noise) * np.cos(10 * yy + noise)
        elif texture["movement"] == "flowing":
            pattern = np.sin(3 * xx + 2 * yy + noise * 0.5)
        elif texture["movement"] == "directed":
            pattern = np.sin(5 * xx + noise * 0.3) * 0.5 + 0.5
        elif texture["movement"] == "pulsing":
            r = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2)
            pattern = np.sin(10 * r + noise)
        else:
            pattern = 0.5 * (xx + yy) + noise * 0.2

        # Normalize pattern
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)

        # Apply color (simplified HSV to RGB)
        hue = color["hue"] / 360
        sat = color["saturation"]
        val = color["brightness"]

        # HSV to RGB approximation
        c = val * sat
        x_color = c * (1 - abs((hue * 6) % 2 - 1))
        m = val - c

        if hue < 1/6:
            r, g, b = c, x_color, 0
        elif hue < 2/6:
            r, g, b = x_color, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x_color
        elif hue < 4/6:
            r, g, b = 0, x_color, c
        elif hue < 5/6:
            r, g, b = x_color, 0, c
        else:
            r, g, b = c, 0, x_color

        # Create RGB image
        rgb = np.zeros((h, w, 3))
        rgb[:, :, 0] = pattern * (r + m) + (1 - pattern) * m
        rgb[:, :, 1] = pattern * (g + m) + (1 - pattern) * m
        rgb[:, :, 2] = pattern * (b + m) + (1 - pattern) * m

        return np.clip(rgb, 0, 1)

    def state_to_audio(
        self,
        state: str,
        eeg_features: np.ndarray | None = None,
        duration: float = 2.0,
        sample_rate: int = 22050,
    ) -> dict[str, Any]:
        """Convert state to audio representation.

        Args:
            state: Consciousness state name.
            eeg_features: Optional EEG features for modulation.
            duration: Audio duration in seconds.
            sample_rate: Sample rate in Hz.

        Returns:
            Audio parameters and waveform data.
        """
        # Get base sound parameters
        sound = self.state_sounds.get(state, {
            "base_freq": 330, "harmonics": "neutral", "tempo": 80
        })

        # Modulate with EEG features
        if eeg_features is not None:
            # Modulate frequency with feature energy
            energy = np.mean(eeg_features ** 2)
            freq_mod = 1 + 0.2 * (energy - 0.5)
            sound["base_freq"] *= freq_mod

        # Generate audio waveform
        waveform = self._generate_audio_waveform(
            sound, duration, sample_rate, eeg_features
        )

        return {
            "parameters": sound,
            "waveform": waveform,
            "duration": duration,
            "sample_rate": sample_rate,
            "state": state,
        }

    def _generate_audio_waveform(
        self,
        sound: dict[str, Any],
        duration: float,
        sample_rate: int,
        features: np.ndarray | None,
    ) -> np.ndarray:
        """Generate audio waveform.

        Args:
            sound: Sound parameters.
            duration: Duration in seconds.
            sample_rate: Sample rate.
            features: Optional features for modulation.

        Returns:
            Audio waveform array.
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)

        freq = sound["base_freq"]
        harmonics = sound["harmonics"]

        # Base tone
        waveform = np.sin(2 * np.pi * freq * t)

        # Add harmonics based on timbre
        if harmonics == "sharp":
            waveform += 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            waveform += 0.3 * np.sin(2 * np.pi * 3 * freq * t)
            waveform += 0.2 * np.sin(2 * np.pi * 5 * freq * t)
        elif harmonics == "smooth":
            waveform += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        elif harmonics == "clear":
            waveform += 0.2 * np.sin(2 * np.pi * 3 * freq * t)
        elif harmonics == "rich":
            waveform += 0.4 * np.sin(2 * np.pi * 2 * freq * t)
            waveform += 0.2 * np.sin(2 * np.pi * 3 * freq * t)
            waveform += 0.1 * np.sin(2 * np.pi * 4 * freq * t)
        elif harmonics == "deep":
            waveform += 0.5 * np.sin(2 * np.pi * 0.5 * freq * t)
        elif harmonics == "bright":
            waveform += 0.4 * np.sin(2 * np.pi * 2 * freq * t)
            waveform += 0.3 * np.sin(2 * np.pi * 4 * freq * t)

        # Apply envelope
        envelope = np.ones(n_samples)
        attack = int(0.1 * n_samples)
        release = int(0.2 * n_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        waveform *= envelope

        # Normalize
        waveform /= np.max(np.abs(waveform)) + 1e-10

        return waveform

    def state_to_text(
        self,
        state: str,
        eeg_features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Convert state to text description.

        Args:
            state: Consciousness state name.
            eeg_features: Optional EEG features for context.

        Returns:
            Text description and metadata.
        """
        # Base descriptions
        descriptions = {
            "anxiety": {
                "primary": "A state of heightened alertness with scattered attention",
                "sensory": "Sharp edges, rapid movements, red-orange hues",
                "physical": "Elevated heart rate, muscle tension, shallow breathing",
            },
            "calm": {
                "primary": "Peaceful equilibrium with gentle awareness",
                "sensory": "Soft blues, flowing curves, gentle waves",
                "physical": "Relaxed muscles, slow steady breathing, low heart rate",
            },
            "focus": {
                "primary": "Directed attention with clear mental clarity",
                "sensory": "Clean greens, straight lines, forward movement",
                "physical": "Alert but relaxed, steady gaze, measured breathing",
            },
            "flow": {
                "primary": "Effortless engagement with complete absorption",
                "sensory": "Purple auroras, smooth transitions, harmonious patterns",
                "physical": "Energized yet relaxed, time distortion, automatic responses",
            },
            "meditation": {
                "primary": "Deep stillness with expanded awareness",
                "sensory": "Deep indigo, infinite space, subtle vibrations",
                "physical": "Profound relaxation, minimal movement, deep breathing",
            },
            "excitement": {
                "primary": "Elevated energy with positive anticipation",
                "sensory": "Bright yellows, pulsing rhythms, upward spirals",
                "physical": "Increased heart rate, heightened senses, quick movements",
            },
        }

        desc = descriptions.get(state, {
            "primary": f"A state of {state}",
            "sensory": "Neutral sensory qualities",
            "physical": "Normal physiological state",
        })

        # Modulate intensity with EEG features
        intensity = "moderate"
        if eeg_features is not None:
            energy = np.mean(eeg_features ** 2)
            if energy > 0.7:
                intensity = "intense"
            elif energy < 0.3:
                intensity = "subtle"

        return {
            "state": state,
            "intensity": intensity,
            "description": desc,
            "full_text": f"{desc['primary']}. Characterized by {desc['sensory'].lower()}. "
                        f"Physically manifests as {desc['physical'].lower()}.",
        }


class EmotionalStateGenerator:
    """Generate emotional state representations.

    Maps EEG features to emotional dimensions using
    the valence-arousal-dominance model.

    Example:
        >>> generator = EmotionalStateGenerator()
        >>> emotional_state = generator.from_eeg(eeg_features)
        >>> print(f"Valence: {emotional_state['valence']:.2f}")
    """

    def __init__(self):
        """Initialize emotional state generator."""
        # EEG band weights for emotion dimensions
        # Based on neuroscience literature
        self.valence_weights = {
            "alpha_asymmetry": 0.4,   # Frontal alpha asymmetry
            "theta_frontal": -0.3,    # Frontal theta (negative valence)
            "beta_frontal": 0.3,      # Frontal beta (approach behavior)
        }

        self.arousal_weights = {
            "beta": 0.4,    # Beta power → arousal
            "gamma": 0.3,   # Gamma power → arousal
            "theta": -0.2,  # Theta → relaxation (inverse)
            "alpha": -0.1,  # Alpha → relaxation (inverse)
        }

        self.dominance_weights = {
            "beta_frontal": 0.4,   # Frontal beta → control
            "alpha_parietal": 0.3, # Parietal alpha → confidence
        }

    def from_eeg(
        self,
        features: np.ndarray | dict[str, float],
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract emotional state from EEG features.

        Args:
            features: EEG feature vector or dictionary.
            feature_names: Names of features if array.

        Returns:
            Emotional state with VAD values.
        """
        # Convert to dictionary if needed
        if isinstance(features, np.ndarray):
            if feature_names is None:
                # Create generic names
                feature_dict = {f"feature_{i}": v for i, v in enumerate(features)}
            else:
                feature_dict = dict(zip(feature_names, features))
        else:
            feature_dict = features

        # Compute dimensions
        valence = self._compute_valence(feature_dict)
        arousal = self._compute_arousal(feature_dict)
        dominance = self._compute_dominance(feature_dict)

        # Map to emotion category
        emotion = self._map_to_emotion(valence, arousal, dominance)

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "emotion": emotion,
            "coordinates": [valence, arousal, dominance],
        }

    def _compute_valence(self, features: dict[str, float]) -> float:
        """Compute valence from features.

        Args:
            features: Feature dictionary.

        Returns:
            Valence value (-1 to 1).
        """
        valence = 0.0
        count = 0

        # Look for relevant features
        for key, value in features.items():
            key_lower = key.lower()

            if "alpha" in key_lower and "asymmetry" in key_lower:
                valence += self.valence_weights["alpha_asymmetry"] * value
                count += 1
            elif "theta" in key_lower and "frontal" in key_lower:
                valence += self.valence_weights["theta_frontal"] * value
                count += 1
            elif "beta" in key_lower and "frontal" in key_lower:
                valence += self.valence_weights["beta_frontal"] * value
                count += 1

        # Default based on general features
        if count == 0:
            values = list(features.values())
            if values:
                valence = np.tanh(np.mean(values) - 0.5)
            else:
                valence = 0.0

        return float(np.clip(valence, -1, 1))

    def _compute_arousal(self, features: dict[str, float]) -> float:
        """Compute arousal from features.

        Args:
            features: Feature dictionary.

        Returns:
            Arousal value (0 to 1).
        """
        arousal = 0.5  # Baseline
        count = 0

        for key, value in features.items():
            key_lower = key.lower()

            if "beta" in key_lower:
                arousal += self.arousal_weights["beta"] * (value - 0.5)
                count += 1
            elif "gamma" in key_lower:
                arousal += self.arousal_weights["gamma"] * (value - 0.5)
                count += 1
            elif "theta" in key_lower:
                arousal += self.arousal_weights["theta"] * (value - 0.5)
                count += 1
            elif "alpha" in key_lower:
                arousal += self.arousal_weights["alpha"] * (value - 0.5)
                count += 1

        return float(np.clip(arousal, 0, 1))

    def _compute_dominance(self, features: dict[str, float]) -> float:
        """Compute dominance from features.

        Args:
            features: Feature dictionary.

        Returns:
            Dominance value (0 to 1).
        """
        dominance = 0.5  # Baseline

        for key, value in features.items():
            key_lower = key.lower()

            if "beta" in key_lower and "frontal" in key_lower:
                dominance += self.dominance_weights["beta_frontal"] * (value - 0.5)
            elif "alpha" in key_lower and "parietal" in key_lower:
                dominance += self.dominance_weights["alpha_parietal"] * (value - 0.5)

        return float(np.clip(dominance, 0, 1))

    def _map_to_emotion(
        self,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> str:
        """Map VAD coordinates to emotion label.

        Args:
            valence: Valence value.
            arousal: Arousal value.
            dominance: Dominance value.

        Returns:
            Emotion label.
        """
        # Simplified emotion mapping
        if valence > 0.3:
            if arousal > 0.6:
                return "excitement" if dominance > 0.5 else "joy"
            else:
                return "contentment" if dominance > 0.5 else "calm"
        elif valence < -0.3:
            if arousal > 0.6:
                return "anger" if dominance > 0.5 else "anxiety"
            else:
                return "sadness" if dominance < 0.5 else "boredom"
        else:
            if arousal > 0.6:
                return "alertness"
            else:
                return "neutral"


class QualiaSynthesizer:
    """Main qualia synthesis interface.

    Combines cross-modal fusion, synesthesia simulation,
    and emotional state generation.

    Example:
        >>> synthesizer = QualiaSynthesizer()
        >>> qualia = synthesizer.synthesize(
        ...     eeg_features=eeg,
        ...     state="flow",
        ... )
    """

    def __init__(self, output_dim: int = 128):
        """Initialize qualia synthesizer.

        Args:
            output_dim: Output embedding dimension.
        """
        self.fusion = CrossModalFusion(output_dim=output_dim)
        self.synesthesia = SynesthesiaSimulator()
        self.emotion_generator = EmotionalStateGenerator()

    def synthesize(
        self,
        eeg_features: np.ndarray,
        state: str | None = None,
        audio_features: np.ndarray | None = None,
        visual_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> QualiaRepresentation:
        """Synthesize qualia from multi-modal inputs.

        Args:
            eeg_features: EEG feature vector.
            state: Optional consciousness state label.
            audio_features: Optional audio features.
            visual_features: Optional visual features.
            text_embedding: Optional text embedding.

        Returns:
            QualiaRepresentation object.
        """
        # Build modality embeddings
        modality_embeddings = {"eeg": eeg_features}
        if audio_features is not None:
            modality_embeddings["audio"] = audio_features
        if visual_features is not None:
            modality_embeddings["visual"] = visual_features
        if text_embedding is not None:
            modality_embeddings["text"] = text_embedding

        # Compute modality weights
        weights = {k: 1.0 / len(modality_embeddings) for k in modality_embeddings}
        weights["eeg"] = 0.4  # EEG gets higher weight
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Fuse embeddings
        unified_embedding = self.fusion.fuse(modality_embeddings, weights)

        # Get emotional state
        emotional = self.emotion_generator.from_eeg(eeg_features)

        # Generate sensory mappings
        sensory_mapping = {}
        if state:
            sensory_mapping["visual"] = self.synesthesia.state_to_visual(
                state, eeg_features
            )
            sensory_mapping["audio"] = self.synesthesia.state_to_audio(
                state, eeg_features
            )
            sensory_mapping["text"] = self.synesthesia.state_to_text(
                state, eeg_features
            )

        return QualiaRepresentation(
            embedding=unified_embedding,
            modality_weights=weights,
            emotional_valence=emotional["valence"],
            arousal=emotional["arousal"],
            dominance=emotional["dominance"],
            sensory_mapping=sensory_mapping,
        )
