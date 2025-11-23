"""Emergence metrics for consciousness research.

This module provides tools for analyzing emergent properties:
- Temporal binding and memory consolidation
- State transitions and phase changes
- Complexity metrics (LZ, Kolmogorov, fractal)
- Integrated Information Theory (IIT) metrics
- Critical phenomena and phase transitions
"""

from .temporal_binding import (
    TemporalBindingTracker,
    TemporalBindingMetrics,
    estimate_specious_present,
    compute_working_memory_span,
)
from .transitions import (
    TransitionDetector,
    StateTransition,
    TransitionAnalysis,
    HysteresisAnalyzer,
    compute_transition_smoothness,
    detect_phase_transition_points,
    compute_transition_entropy,
)
from .complexity import (
    ComplexityAnalyzer,
    ComplexityMetrics,
    compute_lempel_ziv_complexity,
    estimate_kolmogorov_complexity,
    compute_fractal_dimension,
    compute_multiscale_entropy,
    compute_permutation_entropy,
    compute_approximate_entropy,
    compute_spectral_entropy,
)
from .integration_theory import (
    IITAnalyzer,
    IITMetrics,
    compute_phi_approximation,
    compute_causal_density,
    compute_integration_differentiation_balance,
)
from .phase_transitions import (
    PhaseTransitionDetector,
    PhaseTransitionMetrics,
    compute_avalanche_statistics,
    compute_edge_of_chaos,
    construct_phase_diagram,
)

__all__ = [
    # Temporal Binding
    "TemporalBindingTracker",
    "TemporalBindingMetrics",
    "estimate_specious_present",
    "compute_working_memory_span",
    # Transitions
    "TransitionDetector",
    "StateTransition",
    "TransitionAnalysis",
    "HysteresisAnalyzer",
    "compute_transition_smoothness",
    "detect_phase_transition_points",
    "compute_transition_entropy",
    # Complexity
    "ComplexityAnalyzer",
    "ComplexityMetrics",
    "compute_lempel_ziv_complexity",
    "estimate_kolmogorov_complexity",
    "compute_fractal_dimension",
    "compute_multiscale_entropy",
    "compute_permutation_entropy",
    "compute_approximate_entropy",
    "compute_spectral_entropy",
    # IIT
    "IITAnalyzer",
    "IITMetrics",
    "compute_phi_approximation",
    "compute_causal_density",
    "compute_integration_differentiation_balance",
    # Phase Transitions
    "PhaseTransitionDetector",
    "PhaseTransitionMetrics",
    "compute_avalanche_statistics",
    "compute_edge_of_chaos",
    "construct_phase_diagram",
]
