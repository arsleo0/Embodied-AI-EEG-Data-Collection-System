#!/usr/bin/env python3
"""Consciousness analysis example.

Demonstrates the TIER 2 consciousness analysis features:
- Latent space mapping and visualization
- Attention pattern analysis
- Meta-awareness detection

Usage:
    python scripts/consciousness_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.analysis.signal import Preprocessor
from src.analysis.features import FeatureExtractor
from src.consciousness.latent_space import (
    ConsciousnessMapper,
    LatentSpaceVisualizer,
    TrajectoryTracker,
    compute_trajectory_metrics,
)
from src.consciousness.attention import (
    AttentionAnalyzer,
    compute_attention_entropy,
    compute_attention_distribution,
)
from src.consciousness.meta_awareness import (
    IntrospectionDetector,
    UncertaintyQuantifier,
    compute_metacognitive_index,
)


def generate_synthetic_eeg(
    n_sessions: int = 30,
    n_channels: int = 4,
    duration: float = 10.0,
    fs: float = 256.0,
    state: str = "neutral",
) -> list[np.ndarray]:
    """Generate synthetic EEG data with state-specific patterns.

    Args:
        n_sessions: Number of sessions.
        n_channels: Number of channels.
        duration: Duration in seconds.
        fs: Sampling frequency.
        state: Consciousness state.

    Returns:
        List of EEG arrays.
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    sessions = []
    for _ in range(n_sessions):
        # Base signal
        signal = np.random.randn(n_channels, n_samples) * 10

        # State-specific characteristics
        if state == "meditation":
            # Strong alpha (10 Hz) + some theta (6 Hz)
            alpha = 30 * np.sin(2 * np.pi * 10 * t)
            theta = 15 * np.sin(2 * np.pi * 6 * t)
            signal += alpha + theta

        elif state == "focus":
            # Strong beta (20 Hz)
            beta = 25 * np.sin(2 * np.pi * 20 * t)
            signal += beta

        elif state == "flow":
            # Alpha + theta balance
            alpha = 25 * np.sin(2 * np.pi * 10 * t)
            theta = 20 * np.sin(2 * np.pi * 6 * t)
            signal += alpha + theta

        elif state == "introspection":
            # Frontal theta + alpha suppression
            theta = 25 * np.sin(2 * np.pi * 6 * t)
            # Enhance frontal channels
            signal[1:3] += theta * 1.5
            signal[0] += theta * 0.5
            signal[3] += theta * 0.5

        # Add noise
        signal += np.random.randn(n_channels, n_samples) * 5

        sessions.append(signal)

    return sessions


def main():
    """Run consciousness analysis example."""
    print("\n" + "=" * 60)
    print("  Consciousness Research Workbench - TIER 2 Example")
    print("=" * 60 + "\n")

    # Parameters
    fs = 256
    n_channels = 4
    channel_names = ["TP9", "AF7", "AF8", "TP10"]

    # =================================================================
    # 1. Generate synthetic data for different consciousness states
    # =================================================================
    print("1. Generating synthetic EEG data...")

    states = ["meditation", "focus", "neutral", "introspection"]
    all_sessions = []
    all_labels = []

    for state in states:
        sessions = generate_synthetic_eeg(
            n_sessions=8,
            n_channels=n_channels,
            duration=10.0,
            fs=fs,
            state=state,
        )
        all_sessions.extend(sessions)
        all_labels.extend([state] * len(sessions))

    print(f"   Generated {len(all_sessions)} sessions")
    print(f"   States: {set(all_labels)}")

    # =================================================================
    # 2. Preprocess and extract features
    # =================================================================
    print("\n2. Preprocessing and extracting features...")

    preprocessor = Preprocessor(fs)
    preprocessor.configure(
        notch_freq=60,
        bandpass=(1, 50),
        remove_artifacts=True,
    )

    extractor = FeatureExtractor(fs, channel_names)

    all_features = []
    for session in all_sessions:
        # Preprocess
        processed = preprocessor.process(session)
        # Extract features
        features = extractor.extract(processed, flatten=True)
        all_features.append(features)

    # Convert to array
    import pandas as pd
    feature_df = pd.DataFrame(all_features)
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    X = feature_df[numeric_cols].values

    print(f"   Extracted {X.shape[1]} features per session")

    # =================================================================
    # 3. Consciousness Latent Space Mapping
    # =================================================================
    print("\n3. Mapping to consciousness latent space...")

    mapper = ConsciousnessMapper(
        embedding_dim=32,
        use_semantic=False,  # Skip semantic for speed
    )

    # Fit and transform
    embeddings = mapper.fit_transform(X, all_labels)

    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Known states: {mapper.states}")

    # Find nearest state for a sample
    sample_embedding = embeddings[0]
    nearest = mapper.find_nearest_state(sample_embedding, top_k=3)
    print(f"\n   Sample embedding nearest states:")
    for state, dist in nearest:
        print(f"   - {state}: distance={dist:.3f}")

    # Compute similarities
    similarities = mapper.compute_state_similarity(sample_embedding)
    print(f"\n   State similarities:")
    for state, sim in similarities.items():
        print(f"   - {state}: {sim:.3f}")

    # =================================================================
    # 4. Latent Space Visualization
    # =================================================================
    print("\n4. Creating latent space visualization...")

    try:
        visualizer = LatentSpaceVisualizer(method="pca", n_components=2)
        coords_2d = visualizer.fit_transform(embeddings)

        print(f"   2D coordinates shape: {coords_2d.shape}")

        # Create figure (won't display without plotly)
        try:
            fig = visualizer.plot_2d(coords_2d, all_labels, title="Consciousness Latent Space")
            # Save to HTML
            output_path = Path("data/outputs/consciousness_latent_space.html")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            print(f"   Saved visualization to: {output_path}")
        except ImportError:
            print("   (Plotly not available, skipping plot)")

    except Exception as e:
        print(f"   Visualization skipped: {e}")

    # =================================================================
    # 5. Trajectory Analysis
    # =================================================================
    print("\n5. Analyzing consciousness trajectory...")

    tracker = TrajectoryTracker()

    # Simulate temporal sequence (first 10 sessions)
    for i in range(10):
        tracker.add_point(
            embeddings[i],
            timestamp=float(i * 10),  # 10 seconds apart
            label=all_labels[i],
        )

    trajectory = tracker.get_trajectory()
    metrics = compute_trajectory_metrics(trajectory)

    print(f"   Trajectory points: {trajectory.n_points}")
    print(f"   Duration: {trajectory.duration:.1f}s")
    print(f"   Total distance: {metrics['total_distance']:.3f}")
    print(f"   Efficiency: {metrics['efficiency']:.3f}")
    print(f"   Mean velocity: {metrics['velocity']['mean']:.3f}")

    if "state_transitions" in metrics:
        print(f"   State transitions: {metrics['state_transitions']['n_transitions']}")

    # =================================================================
    # 6. Attention Pattern Analysis
    # =================================================================
    print("\n6. Analyzing attention patterns...")

    attention_analyzer = AttentionAnalyzer(fs)

    # Analyze first session
    sample_data = all_sessions[0]

    # Global workspace metrics
    gw_metrics = attention_analyzer.compute_global_workspace(sample_data)

    print(f"\n   Global Workspace Metrics:")
    print(f"   - Integration: {gw_metrics.integration:.3f}")
    print(f"   - Differentiation: {gw_metrics.differentiation:.3f}")
    print(f"   - Broadcast strength: {gw_metrics.broadcast_strength:.3f}")
    print(f"   - Workspace stability: {gw_metrics.workspace_stability:.3f}")
    print(f"   - Ignition events: {gw_metrics.ignition_events}")

    # Attention entropy
    entropy = compute_attention_entropy(sample_data, fs)
    print(f"\n   Attention Entropy:")
    print(f"   - Spatial entropy: {entropy['spatial_entropy']:.3f}")
    print(f"   - Temporal entropy: {entropy['temporal_entropy']:.3f}")

    # Attention distribution
    distribution = compute_attention_distribution(sample_data, fs)
    print(f"\n   Attention Distribution:")
    print(f"   - Focus score: {distribution['focus_score']:.3f}")
    print(f"   - Dominant band: {distribution['dominant_band']}")

    # =================================================================
    # 7. Meta-Awareness Detection
    # =================================================================
    print("\n7. Detecting meta-awareness states...")

    introspection_detector = IntrospectionDetector(fs, channel_names)

    # Analyze introspection session
    introspection_idx = all_labels.index("introspection")
    introspection_data = all_sessions[introspection_idx]

    state = introspection_detector.detect(introspection_data)

    print(f"\n   Meta-Cognitive State (introspection session):")
    print(f"   - Self-awareness: {state.self_awareness:.3f}")
    print(f"   - Introspection: {state.introspection:.3f}")
    print(f"   - Monitoring: {state.monitoring:.3f}")
    print(f"   - Evaluation: {state.evaluation:.3f}")
    print(f"   - Metacognitive index: {state.metacognitive_index:.3f}")
    print(f"   - Confidence: {state.confidence:.3f}")

    # Compare with focus session
    focus_idx = all_labels.index("focus")
    focus_data = all_sessions[focus_idx]
    focus_state = introspection_detector.detect(focus_data)

    print(f"\n   Meta-Cognitive State (focus session):")
    print(f"   - Metacognitive index: {focus_state.metacognitive_index:.3f}")

    # =================================================================
    # 8. Uncertainty Quantification
    # =================================================================
    print("\n8. Quantifying prediction uncertainty...")

    # Generate some predictions (simulated)
    predictions = np.random.dirichlet(np.ones(len(states)), size=len(all_sessions))

    quantifier = UncertaintyQuantifier(n_bootstrap=50)
    uncertainty = quantifier.quantify(predictions, X)

    print(f"\n   Uncertainty Metrics:")
    print(f"   - Epistemic: {uncertainty.epistemic:.3f}")
    print(f"   - Aleatoric: {uncertainty.aleatoric:.3f}")
    print(f"   - Total: {uncertainty.total:.3f}")
    print(f"   - Entropy: {uncertainty.entropy:.3f}")
    print(f"   - Reliability: {uncertainty.reliability:.3f}")
    print(f"   - Confidence interval: {uncertainty.confidence_interval}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    print(f"""
Summary:
- Analyzed {len(all_sessions)} EEG sessions across {len(states)} states
- Mapped to {embeddings.shape[1]}-dimensional consciousness space
- Computed Global Workspace metrics for attention analysis
- Detected meta-cognitive states with introspection detector
- Quantified prediction uncertainty

Key Findings:
- Highest metacognition in introspection state: {state.metacognitive_index:.3f}
- Dominant attention band: {distribution['dominant_band']}
- Trajectory efficiency: {metrics['efficiency']:.3f}

Next steps:
- Collect real EEG data with Muse device
- Run consciousness scenarios
- Train models on real neural correlates
- Compare human attention with AI models
""")


if __name__ == "__main__":
    main()
