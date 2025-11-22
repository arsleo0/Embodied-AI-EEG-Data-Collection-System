"""Consciousness layer - AI consciousness analysis modules.

Modules:
    latent_space: EEG to embedding projection and visualization
    attention: Attention pattern analysis
    meta_awareness: Meta-awareness and introspection detection
    qualia: Cross-modal synthesis
    emergence: Complexity and emergence metrics
"""

from .latent_space import (
    ConsciousnessMapper,
    EEGEncoder,
    SemanticLabeler,
    LatentSpaceVisualizer,
    TrajectoryTracker,
    ConsciousnessTrajectory,
    plot_embeddings_2d,
    plot_embeddings_3d,
    plot_trajectory,
    compute_trajectory_metrics,
)
from .attention import (
    AttentionAnalyzer,
    GlobalWorkspaceMetrics,
    AttentionComparator,
    HumanAIAlignment,
    compute_attention_entropy,
    compute_attention_distribution,
    compute_attention_overlap,
    visualize_attention_comparison,
)
from .meta_awareness import (
    IntrospectionDetector,
    MetaCognitiveState,
    UncertaintyQuantifier,
    UncertaintyMetrics,
    detect_self_awareness,
    compute_metacognitive_index,
    compute_epistemic_uncertainty,
    compute_aleatoric_uncertainty,
)

__all__ = [
    # Latent Space
    "ConsciousnessMapper",
    "EEGEncoder",
    "SemanticLabeler",
    "LatentSpaceVisualizer",
    "TrajectoryTracker",
    "ConsciousnessTrajectory",
    "plot_embeddings_2d",
    "plot_embeddings_3d",
    "plot_trajectory",
    "compute_trajectory_metrics",
    # Attention
    "AttentionAnalyzer",
    "GlobalWorkspaceMetrics",
    "AttentionComparator",
    "HumanAIAlignment",
    "compute_attention_entropy",
    "compute_attention_distribution",
    "compute_attention_overlap",
    "visualize_attention_comparison",
    # Meta-Awareness
    "IntrospectionDetector",
    "MetaCognitiveState",
    "UncertaintyQuantifier",
    "UncertaintyMetrics",
    "detect_self_awareness",
    "compute_metacognitive_index",
    "compute_epistemic_uncertainty",
    "compute_aleatoric_uncertainty",
]
