"""Consciousness mapper plugin.

Plugin for mapping EEG features to consciousness latent space.
"""

import numpy as np
from pathlib import Path
from typing import Any

from ..base import ConsciousnessPlugin


class ConsciousnessMapperPlugin(ConsciousnessPlugin):
    """Plugin for consciousness state mapping.

    Maps EEG features to a latent space for consciousness
    state analysis and visualization.

    Example:
        >>> plugin = ConsciousnessMapperPlugin()
        >>> plugin.configure(embedding_dim=32)
        >>> result = plugin.process(features, labels=labels)
    """

    name = "consciousness_mapper"
    version = "1.0.0"
    description = "Map EEG features to consciousness latent space"

    def __init__(self):
        """Initialize consciousness mapper plugin."""
        super().__init__()
        self._mapper = None
        self._config = {
            "embedding_dim": 32,
            "use_semantic": False,
            "semantic_model": "all-MiniLM-L6-v2",
        }

    def configure(self, **kwargs) -> None:
        """Configure the plugin.

        Args:
            embedding_dim: Output embedding dimension.
            use_semantic: Use semantic state embeddings.
            semantic_model: Sentence transformer model.
        """
        self._config.update(kwargs)

    def process(
        self,
        data: np.ndarray,
        labels: list[str] | np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process features and map to latent space.

        Args:
            data: Feature matrix (samples, features).
            labels: State labels for fitting.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with embeddings and analysis.
        """
        from src.consciousness.latent_space import ConsciousnessMapper

        # Initialize mapper if needed
        if self._mapper is None:
            self._mapper = ConsciousnessMapper(
                embedding_dim=self._config["embedding_dim"],
                use_semantic=self._config["use_semantic"],
                semantic_model=self._config["semantic_model"],
            )

        # Fit and transform
        if labels is not None:
            embeddings = self._mapper.fit_transform(data, labels)

            # Analyze states
            state_analysis = {}
            for state in self._mapper.states:
                centroid = self._mapper.get_state_centroid(state)
                state_analysis[state] = {
                    "centroid": centroid.tolist() if centroid is not None else None,
                }
        else:
            # Transform only
            if not self._mapper.is_fitted:
                raise RuntimeError("Mapper not fitted. Provide labels for fitting.")
            embeddings = self._mapper.transform(data)
            state_analysis = {}

        return {
            "embeddings": embeddings,
            "embedding_dim": self._config["embedding_dim"],
            "n_samples": len(data),
            "states": self._mapper.states,
            "state_analysis": state_analysis,
        }

    def save_state(self, path: str | Path) -> None:
        """Save plugin state to file.

        Args:
            path: Output file path.
        """
        if self._mapper is not None:
            self._mapper.save(path)

    def load_state(self, path: str | Path) -> None:
        """Load plugin state from file.

        Args:
            path: Input file path.
        """
        from src.consciousness.latent_space import ConsciousnessMapper
        self._mapper = ConsciousnessMapper.load(path)


class AttentionAnalysisPlugin(ConsciousnessPlugin):
    """Plugin for attention pattern analysis.

    Analyzes attention patterns using Global Workspace Theory
    inspired metrics.

    Example:
        >>> plugin = AttentionAnalysisPlugin()
        >>> plugin.configure(fs=256.0)
        >>> result = plugin.process(eeg_data)
    """

    name = "attention_analysis"
    version = "1.0.0"
    description = "Analyze attention patterns with Global Workspace metrics"

    def __init__(self):
        """Initialize attention analysis plugin."""
        super().__init__()
        self._analyzer = None
        self._config = {
            "fs": 256.0,
        }

    def configure(self, **kwargs) -> None:
        """Configure the plugin.

        Args:
            fs: Sampling frequency in Hz.
        """
        self._config.update(kwargs)

    def process(self, data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Process EEG data for attention analysis.

        Args:
            data: EEG data (channels, samples).
            **kwargs: Additional arguments.

        Returns:
            Dictionary with attention metrics.
        """
        from src.consciousness.attention import AttentionAnalyzer

        # Initialize analyzer
        if self._analyzer is None:
            self._analyzer = AttentionAnalyzer(fs=self._config["fs"])

        # Compute metrics
        gw_metrics = self._analyzer.compute_global_workspace(data)
        entropy = self._analyzer.compute_attention_entropy(data)
        distribution = self._analyzer.compute_attention_distribution(data)

        return {
            "global_workspace": {
                "integration": gw_metrics.integration,
                "differentiation": gw_metrics.differentiation,
                "broadcast_strength": gw_metrics.broadcast_strength,
                "workspace_stability": gw_metrics.workspace_stability,
                "ignition_events": gw_metrics.ignition_events,
            },
            "entropy": entropy,
            "distribution": distribution,
        }


class MetaAwarenessPlugin(ConsciousnessPlugin):
    """Plugin for meta-awareness detection.

    Detects introspection and meta-cognitive states from EEG.

    Example:
        >>> plugin = MetaAwarenessPlugin()
        >>> plugin.configure(fs=256.0, channel_names=["TP9", "AF7", "AF8", "TP10"])
        >>> result = plugin.process(eeg_data)
    """

    name = "meta_awareness"
    version = "1.0.0"
    description = "Detect meta-awareness and introspection states"

    def __init__(self):
        """Initialize meta-awareness plugin."""
        super().__init__()
        self._detector = None
        self._config = {
            "fs": 256.0,
            "channel_names": ["TP9", "AF7", "AF8", "TP10"],
        }

    def configure(self, **kwargs) -> None:
        """Configure the plugin.

        Args:
            fs: Sampling frequency in Hz.
            channel_names: EEG channel names.
        """
        self._config.update(kwargs)

    def process(self, data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Process EEG data for meta-awareness detection.

        Args:
            data: EEG data (channels, samples).
            **kwargs: Additional arguments.

        Returns:
            Dictionary with meta-awareness metrics.
        """
        from src.consciousness.meta_awareness import IntrospectionDetector

        # Initialize detector
        if self._detector is None:
            self._detector = IntrospectionDetector(
                fs=self._config["fs"],
                channel_names=self._config["channel_names"],
            )

        # Detect state
        state = self._detector.detect(data)

        return {
            "self_awareness": state.self_awareness,
            "introspection": state.introspection,
            "monitoring": state.monitoring,
            "evaluation": state.evaluation,
            "metacognitive_index": state.metacognitive_index,
            "confidence": state.confidence,
        }
