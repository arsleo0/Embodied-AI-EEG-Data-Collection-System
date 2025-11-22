"""Latent space mapping for consciousness states.

This module provides tools for mapping EEG features to embedding spaces:
- EEG to embedding projection
- Consciousness state visualization
- Trajectory tracking over time
"""

from .mapper import (
    ConsciousnessMapper,
    EEGEncoder,
    SemanticLabeler,
)
from .visualizer import (
    LatentSpaceVisualizer,
    plot_embeddings_2d,
    plot_embeddings_3d,
    plot_trajectory,
)
from .trajectory import (
    TrajectoryTracker,
    ConsciousnessTrajectory,
    compute_trajectory_metrics,
)

__all__ = [
    # Mapper
    "ConsciousnessMapper",
    "EEGEncoder",
    "SemanticLabeler",
    # Visualizer
    "LatentSpaceVisualizer",
    "plot_embeddings_2d",
    "plot_embeddings_3d",
    "plot_trajectory",
    # Trajectory
    "TrajectoryTracker",
    "ConsciousnessTrajectory",
    "compute_trajectory_metrics",
]
