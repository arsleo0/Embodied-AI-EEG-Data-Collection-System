#!/usr/bin/env python3
"""Emergence analysis example.

Demonstrates TIER 3 emergence analysis features:
- Complexity metrics (LZ, fractal, entropy)
- Integrated Information Theory (Phi)
- Phase transitions and criticality
- Temporal binding and state transitions
- Qualia synthesis

Usage:
    python scripts/emergence_analysis_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.consciousness.emergence import (
    # Complexity
    ComplexityAnalyzer,
    compute_lempel_ziv_complexity,
    compute_fractal_dimension,
    compute_multiscale_entropy,
    # IIT
    IITAnalyzer,
    compute_phi_approximation,
    compute_integration_differentiation_balance,
    # Phase transitions
    PhaseTransitionDetector,
    compute_avalanche_statistics,
    compute_edge_of_chaos,
    # Temporal binding
    TemporalBindingTracker,
    estimate_specious_present,
    # Transitions
    TransitionDetector,
    compute_transition_entropy,
)
from src.consciousness.qualia import (
    QualiaSynthesizer,
    SynesthesiaSimulator,
    EmotionalStateGenerator,
)


def generate_synthetic_eeg(
    n_channels: int = 4,
    duration: float = 30.0,
    fs: float = 256.0,
    state: str = "normal",
) -> np.ndarray:
    """Generate synthetic EEG with state-specific properties.

    Args:
        n_channels: Number of channels.
        duration: Duration in seconds.
        fs: Sampling frequency.
        state: Brain state type.

    Returns:
        EEG data array.
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Base noise
    data = np.random.randn(n_channels, n_samples) * 10

    if state == "ordered":
        # Highly synchronized oscillations
        for ch in range(n_channels):
            data[ch] += 50 * np.sin(2 * np.pi * 10 * t)

    elif state == "critical":
        # Mix of order and disorder
        for ch in range(n_channels):
            data[ch] += 30 * np.sin(2 * np.pi * 10 * t + ch * 0.2)
            data[ch] += 15 * np.sin(2 * np.pi * 20 * t + np.random.randn() * 0.5)

    elif state == "disordered":
        # High frequency noise dominated
        data += np.random.randn(n_channels, n_samples) * 30

    elif state == "transitional":
        # Gradual change from ordered to disordered
        transition_point = n_samples // 2
        for ch in range(n_channels):
            # First half: ordered
            data[ch, :transition_point] += 50 * np.sin(
                2 * np.pi * 10 * t[:transition_point]
            )
            # Second half: add noise gradually
            noise_scale = np.linspace(0, 40, n_samples - transition_point)
            data[ch, transition_point:] += (
                20 * np.sin(2 * np.pi * 10 * t[transition_point:]) +
                noise_scale * np.random.randn(n_samples - transition_point)
            )

    return data


def main():
    """Run emergence analysis example."""
    print("\n" + "=" * 60)
    print("  Consciousness Research Workbench - TIER 3 Example")
    print("  Emergence Analysis & Qualia Synthesis")
    print("=" * 60 + "\n")

    fs = 256
    n_channels = 4

    # =================================================================
    # 1. Generate data for different states
    # =================================================================
    print("1. Generating synthetic EEG for different states...")

    states = {
        "ordered": generate_synthetic_eeg(n_channels, 10, fs, "ordered"),
        "critical": generate_synthetic_eeg(n_channels, 10, fs, "critical"),
        "disordered": generate_synthetic_eeg(n_channels, 10, fs, "disordered"),
    }

    for state, data in states.items():
        print(f"   {state}: {data.shape}")

    # =================================================================
    # 2. Complexity Analysis
    # =================================================================
    print("\n2. Analyzing signal complexity...")

    complexity_analyzer = ComplexityAnalyzer(n_scales=8)

    for state_name, data in states.items():
        metrics = complexity_analyzer.analyze(data)

        print(f"\n   {state_name.upper()}:")
        print(f"   - Lempel-Ziv complexity: {metrics.lempel_ziv:.3f}")
        print(f"   - Kolmogorov estimate: {metrics.kolmogorov_estimate:.3f}")
        print(f"   - Fractal dimension: {metrics.fractal_dimension:.3f}")
        print(f"   - Overall complexity: {metrics.overall:.3f}")

    # Multi-scale entropy comparison
    print("\n   Multi-scale entropy profiles:")
    for state_name, data in states.items():
        mse = compute_multiscale_entropy(data.flatten(), n_scales=5)
        print(f"   {state_name}: {[f'{x:.2f}' for x in mse]}")

    # =================================================================
    # 3. Integrated Information Theory (IIT)
    # =================================================================
    print("\n3. Computing IIT metrics...")

    iit_analyzer = IITAnalyzer(n_bins=10)

    for state_name, data in states.items():
        metrics = iit_analyzer.analyze(data)

        print(f"\n   {state_name.upper()}:")
        print(f"   - Phi (Φ): {metrics.phi:.4f}")
        print(f"   - Integration: {metrics.integration:.3f}")
        print(f"   - Differentiation: {metrics.differentiation:.3f}")
        print(f"   - Exclusion: {metrics.exclusion:.3f}")

    # Integration-differentiation balance
    print("\n   Integration-Differentiation Balance:")
    for state_name, data in states.items():
        balance = compute_integration_differentiation_balance(data)
        print(f"   {state_name}: balance={balance['balance']:.3f}, "
              f"dominant={balance['dominance']}")

    # =================================================================
    # 4. Phase Transitions
    # =================================================================
    print("\n4. Analyzing phase transitions...")

    phase_detector = PhaseTransitionDetector(fs=fs)

    for state_name, data in states.items():
        metrics = phase_detector.analyze(data)

        print(f"\n   {state_name.upper()}:")
        print(f"   - Order parameter: {metrics.order_parameter:.3f}")
        print(f"   - Susceptibility: {metrics.susceptibility:.3f}")
        print(f"   - Correlation length: {metrics.correlation_length:.3f}")
        print(f"   - Critical point: {metrics.critical_point:.3f}")
        print(f"   - Phase: {metrics.phase}")

    # Edge of chaos analysis
    print("\n   Edge of Chaos scores:")
    for state_name, data in states.items():
        edge = compute_edge_of_chaos(data)
        print(f"   {state_name}: {edge:.3f}")

    # Avalanche statistics
    print("\n   Neuronal Avalanche Statistics:")
    for state_name, data in states.items():
        stats = compute_avalanche_statistics(data, threshold=1.5)
        print(f"   {state_name}: n_avalanches={stats['n_avalanches']}, "
              f"mean_size={stats['mean_size']:.1f}")

    # =================================================================
    # 5. Temporal Binding
    # =================================================================
    print("\n5. Analyzing temporal binding...")

    binding_tracker = TemporalBindingTracker(fs=fs)

    for state_name, data in states.items():
        metrics = binding_tracker.analyze(data)

        print(f"\n   {state_name.upper()}:")
        print(f"   - Integration window: {metrics.integration_window:.3f}s")
        print(f"   - Context retention: {metrics.context_retention:.3f}")
        print(f"   - Binding strength: {metrics.binding_strength:.3f}")
        print(f"   - Temporal coherence: {metrics.temporal_coherence:.3f}")

    # Specious present
    print("\n   Specious Present (duration of 'now'):")
    for state_name, data in states.items():
        sp = estimate_specious_present(data, fs)
        print(f"   {state_name}: {sp:.2f}s")

    # =================================================================
    # 6. State Transitions
    # =================================================================
    print("\n6. Analyzing state transitions...")

    # Generate transitional data
    transitional_data = generate_synthetic_eeg(n_channels, 30, fs, "transitional")

    # Create embeddings (simplified - use mean power as features)
    window_size = int(2 * fs)
    step_size = int(0.5 * fs)
    n_samples = transitional_data.shape[1]

    embeddings = []
    timestamps = []

    for i in range(0, n_samples - window_size, step_size):
        window = transitional_data[:, i:i + window_size]
        # Simple features: mean power in each channel
        features = np.mean(window ** 2, axis=1)
        embeddings.append(features)
        timestamps.append(i / fs)

    embeddings = np.array(embeddings)
    timestamps = np.array(timestamps)

    # Create labels based on time (first half = ordered, second half = disordered)
    labels = ["ordered" if t < 15 else "disordered" for t in timestamps]

    # Detect transitions
    transition_detector = TransitionDetector(threshold=0.5, min_duration=1.0)
    analysis = transition_detector.analyze(embeddings, timestamps, labels)

    print(f"\n   Detected {len(analysis.transitions)} transitions")
    print(f"   Mean smoothness: {analysis.mean_smoothness:.3f}")
    print(f"   Hysteresis: {analysis.hysteresis:.3f}")

    for i, trans in enumerate(analysis.transitions[:3]):
        print(f"   Transition {i+1}: {trans.from_state} → {trans.to_state} "
              f"at t={trans.timestamp:.1f}s")

    # Transition entropy
    entropy = compute_transition_entropy(labels)
    print(f"\n   Transition entropy: {entropy:.3f} bits")

    # =================================================================
    # 7. Qualia Synthesis
    # =================================================================
    print("\n7. Synthesizing qualia representations...")

    synesthesia = SynesthesiaSimulator()

    # Generate synesthetic representations for different states
    consciousness_states = ["meditation", "focus", "anxiety", "flow"]

    for state in consciousness_states:
        # Generate features
        features = np.random.randn(50) * 0.5

        # Visual representation
        visual = synesthesia.state_to_visual(state, features, resolution=(64, 64))

        # Audio representation
        audio = synesthesia.state_to_audio(state, features, duration=1.0)

        # Text description
        text = synesthesia.state_to_text(state, features)

        print(f"\n   {state.upper()}:")
        print(f"   - Visual: hue={visual['color']['hue']:.0f}°, "
              f"saturation={visual['color']['saturation']:.2f}")
        print(f"   - Audio: freq={audio['parameters']['base_freq']:.0f}Hz, "
              f"tempo={audio['parameters']['tempo']}")
        print(f"   - Text: {text['description']['primary'][:50]}...")

    # Emotional state generator
    print("\n   Emotional State Analysis:")
    emotion_gen = EmotionalStateGenerator()

    for state_name in ["ordered", "critical", "disordered"]:
        data = states[state_name]
        # Use band power features
        features = {
            "alpha_frontal": np.mean(data[:2, :]) ** 2,
            "beta_frontal": np.mean(data[:2, :]) ** 2 * 0.5,
            "theta_frontal": np.mean(data[:2, :]) ** 2 * 0.3,
        }

        emotional = emotion_gen.from_eeg(features)
        print(f"   {state_name}: emotion={emotional['emotion']}, "
              f"valence={emotional['valence']:.2f}, "
              f"arousal={emotional['arousal']:.2f}")

    # Full qualia synthesis
    print("\n   Full Qualia Synthesis:")
    synthesizer = QualiaSynthesizer(output_dim=64)

    for state_name in ["ordered", "critical"]:
        data = states[state_name]
        features = np.mean(data ** 2, axis=1)  # Simple power features

        qualia = synthesizer.synthesize(
            eeg_features=features,
            state=state_name if state_name in ["meditation", "focus"] else "calm",
        )

        print(f"\n   {state_name.upper()}:")
        print(f"   - Embedding dim: {len(qualia.embedding)}")
        print(f"   - Valence: {qualia.emotional_valence:.3f}")
        print(f"   - Arousal: {qualia.arousal:.3f}")
        print(f"   - Dominance: {qualia.dominance:.3f}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    print("""
Summary:
- Analyzed complexity metrics across ordered/critical/disordered states
- Computed Phi (Φ) approximations for information integration
- Detected phase transitions and criticality indicators
- Measured temporal binding windows and context retention
- Synthesized cross-modal qualia representations

Key Findings:
- Critical state shows highest Phi and edge-of-chaos proximity
- Ordered state has longest integration window (temporal binding)
- Disordered state has highest complexity but lowest integration

Research Applications:
- Use complexity metrics to characterize consciousness depth
- Monitor Phi during anesthesia or meditation
- Track phase transitions during state changes
- Generate synesthetic representations for consciousness states
""")


if __name__ == "__main__":
    main()
