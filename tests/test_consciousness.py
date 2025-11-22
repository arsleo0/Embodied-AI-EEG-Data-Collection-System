"""Tests for consciousness analysis modules (TIER 2)."""

import numpy as np
import pytest


class TestLatentSpaceMapper:
    """Tests for consciousness latent space mapping."""

    def test_eeg_encoder_fit(self):
        """Test EEG encoder fitting."""
        from src.consciousness.latent_space import EEGEncoder

        encoder = EEGEncoder(embedding_dim=16)
        features = np.random.randn(20, 100)

        encoder.fit(features)

        assert encoder.is_fitted
        assert encoder.embedding_dim == 16

    def test_eeg_encoder_encode(self):
        """Test EEG encoding."""
        from src.consciousness.latent_space import EEGEncoder

        encoder = EEGEncoder(embedding_dim=16)
        features = np.random.randn(20, 100)

        encoder.fit(features)
        embeddings = encoder.encode(features)

        assert embeddings.shape == (20, 16)

    def test_eeg_encoder_not_fitted(self):
        """Test error when encoding without fitting."""
        from src.consciousness.latent_space import EEGEncoder

        encoder = EEGEncoder()
        features = np.random.randn(10, 50)

        with pytest.raises(RuntimeError):
            encoder.encode(features)

    def test_consciousness_mapper_fit_transform(self):
        """Test consciousness mapper fit and transform."""
        from src.consciousness.latent_space import ConsciousnessMapper

        mapper = ConsciousnessMapper(embedding_dim=16, use_semantic=False)
        features = np.random.randn(30, 100)
        labels = ["state_a"] * 10 + ["state_b"] * 10 + ["state_c"] * 10

        embeddings = mapper.fit_transform(features, labels)

        assert embeddings.shape == (30, 16)
        assert mapper.is_fitted
        assert len(mapper.states) == 3

    def test_consciousness_mapper_find_nearest(self):
        """Test finding nearest state."""
        from src.consciousness.latent_space import ConsciousnessMapper

        mapper = ConsciousnessMapper(embedding_dim=8, use_semantic=False)
        features = np.random.randn(20, 50)
        labels = ["a"] * 10 + ["b"] * 10

        mapper.fit(features, labels)
        embeddings = mapper.transform(features)

        nearest = mapper.find_nearest_state(embeddings[0], top_k=2)

        assert len(nearest) == 2
        assert nearest[0][0] in ["a", "b"]
        assert nearest[0][1] >= 0  # Distance is non-negative

    def test_consciousness_mapper_similarity(self):
        """Test state similarity computation."""
        from src.consciousness.latent_space import ConsciousnessMapper

        mapper = ConsciousnessMapper(embedding_dim=8, use_semantic=False)
        features = np.random.randn(20, 50)
        labels = ["x"] * 10 + ["y"] * 10

        mapper.fit(features, labels)
        embeddings = mapper.transform(features)

        similarities = mapper.compute_state_similarity(embeddings[0])

        assert "x" in similarities
        assert "y" in similarities
        assert -1 <= similarities["x"] <= 1  # Cosine similarity range


class TestLatentSpaceVisualizer:
    """Tests for latent space visualization."""

    def test_visualizer_pca(self):
        """Test PCA visualization."""
        from src.consciousness.latent_space import LatentSpaceVisualizer

        visualizer = LatentSpaceVisualizer(method="pca", n_components=2)
        embeddings = np.random.randn(50, 32)

        coords = visualizer.fit_transform(embeddings)

        assert coords.shape == (50, 2)

    def test_visualizer_3d(self):
        """Test 3D visualization."""
        from src.consciousness.latent_space import LatentSpaceVisualizer

        visualizer = LatentSpaceVisualizer(method="pca", n_components=3)
        embeddings = np.random.randn(50, 32)

        coords = visualizer.fit_transform(embeddings)

        assert coords.shape == (50, 3)


class TestTrajectoryTracker:
    """Tests for consciousness trajectory tracking."""

    def test_trajectory_tracker_basic(self):
        """Test basic trajectory tracking."""
        from src.consciousness.latent_space import TrajectoryTracker

        tracker = TrajectoryTracker()

        for i in range(10):
            tracker.add_point(
                np.random.randn(16),
                timestamp=float(i),
                label=f"state_{i % 3}"
            )

        assert tracker.n_points == 10

        trajectory = tracker.get_trajectory()
        assert trajectory.n_points == 10
        assert trajectory.duration == 9.0

    def test_trajectory_metrics(self):
        """Test trajectory metrics computation."""
        from src.consciousness.latent_space import (
            TrajectoryTracker,
            compute_trajectory_metrics,
        )

        tracker = TrajectoryTracker()

        # Create trajectory with known properties
        for i in range(20):
            embedding = np.array([i * 0.1, i * 0.1])  # Linear trajectory
            tracker.add_point(embedding, timestamp=float(i))

        trajectory = tracker.get_trajectory()
        metrics = compute_trajectory_metrics(trajectory)

        assert "velocity" in metrics
        assert "total_distance" in metrics
        assert "efficiency" in metrics
        assert metrics["total_distance"] > 0
        assert 0 <= metrics["efficiency"] <= 1

    def test_trajectory_segment(self):
        """Test trajectory segmentation."""
        from src.consciousness.latent_space import ConsciousnessTrajectory

        embeddings = np.random.randn(20, 8)
        timestamps = np.arange(20).astype(float)
        labels = ["a"] * 10 + ["b"] * 10

        trajectory = ConsciousnessTrajectory(embeddings, timestamps, labels)

        segment = trajectory.get_segment(5.0, 15.0)

        assert segment.n_points == 11
        assert segment.duration == 10.0


class TestAttentionAnalyzer:
    """Tests for attention pattern analysis."""

    def test_global_workspace_metrics(self):
        """Test global workspace metrics computation."""
        from src.consciousness.attention import AttentionAnalyzer

        analyzer = AttentionAnalyzer(fs=256.0)
        data = np.random.randn(4, 2560)  # 10 seconds

        metrics = analyzer.compute_global_workspace(data)

        assert 0 <= metrics.integration <= 1
        assert metrics.differentiation >= 0
        assert 0 <= metrics.broadcast_strength <= 1
        assert 0 <= metrics.workspace_stability <= 1
        assert metrics.ignition_events >= 0

    def test_attention_entropy(self):
        """Test attention entropy computation."""
        from src.consciousness.attention import compute_attention_entropy

        data = np.random.randn(4, 2560)

        entropy = compute_attention_entropy(data, fs=256.0)

        assert "spatial_entropy" in entropy
        assert "temporal_entropy" in entropy
        assert entropy["spatial_entropy"] >= 0

    def test_attention_distribution(self):
        """Test attention distribution computation."""
        from src.consciousness.attention import compute_attention_distribution

        data = np.random.randn(4, 2560)

        distribution = compute_attention_distribution(data, fs=256.0)

        assert "band_distribution" in distribution
        assert "focus_score" in distribution
        assert "dominant_band" in distribution
        assert 0 <= distribution["focus_score"] <= 1


class TestAttentionComparator:
    """Tests for human-AI attention comparison."""

    def test_attention_comparison(self):
        """Test basic attention comparison."""
        from src.consciousness.attention import AttentionComparator

        comparator = AttentionComparator(n_regions=4)

        human_attention = np.random.rand(4)
        ai_attention = np.random.rand(4)

        alignment = comparator.compare(human_attention, ai_attention)

        assert -1 <= alignment.correlation <= 1
        assert 0 <= alignment.overlap <= 1
        assert alignment.divergence >= 0

    def test_temporal_comparison(self):
        """Test temporal attention comparison."""
        from src.consciousness.attention import AttentionComparator

        comparator = AttentionComparator(n_regions=4)

        # Time series attention (10 timesteps, 4 regions)
        human_attention = np.random.rand(10, 4)
        ai_attention = np.random.rand(10, 4)

        alignment = comparator.compare(human_attention, ai_attention)

        assert 0 <= alignment.temporal_sync <= 1 or alignment.temporal_sync >= -1

    def test_attention_overlap(self):
        """Test attention overlap computation."""
        from src.consciousness.attention import compute_attention_overlap

        # Identical distributions
        a = np.array([0.5, 0.3, 0.2])
        overlap = compute_attention_overlap(a, a)
        assert np.isclose(overlap, 1.0)

        # Completely different
        b = np.array([0.0, 0.0, 1.0])
        overlap = compute_attention_overlap(a, b)
        assert overlap < 1.0


class TestIntrospectionDetector:
    """Tests for introspection detection."""

    def test_introspection_detection(self):
        """Test basic introspection detection."""
        from src.consciousness.meta_awareness import IntrospectionDetector

        detector = IntrospectionDetector(fs=256.0)
        data = np.random.randn(4, 2560)

        state = detector.detect(data)

        assert 0 <= state.self_awareness <= 1
        assert 0 <= state.introspection <= 1
        assert 0 <= state.monitoring <= 1
        assert 0 <= state.evaluation <= 1
        assert 0 <= state.metacognitive_index <= 1
        assert 0 <= state.confidence <= 1

    def test_metacognitive_index(self):
        """Test metacognitive index computation."""
        from src.consciousness.meta_awareness import compute_metacognitive_index

        data = np.random.randn(4, 2560)

        index = compute_metacognitive_index(data, fs=256.0)

        assert 0 <= index <= 1

    def test_temporal_tracking(self):
        """Test temporal meta-awareness tracking."""
        from src.consciousness.meta_awareness import IntrospectionDetector

        detector = IntrospectionDetector(fs=256.0)
        data = np.random.randn(4, 5120)  # 20 seconds

        states = detector.track_over_time(data, window_size=2.0, overlap=0.5)

        assert len(states) > 0
        assert all(0 <= s.metacognitive_index <= 1 for s in states)


class TestUncertaintyQuantifier:
    """Tests for uncertainty quantification."""

    def test_uncertainty_quantification(self):
        """Test basic uncertainty quantification."""
        from src.consciousness.meta_awareness import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier(n_bootstrap=50)

        predictions = np.random.rand(100)

        metrics = quantifier.quantify(predictions)

        assert metrics.epistemic >= 0
        assert metrics.aleatoric >= 0
        assert metrics.total >= 0
        assert 0 <= metrics.reliability <= 1

    def test_classification_uncertainty(self):
        """Test uncertainty for classification predictions."""
        from src.consciousness.meta_awareness import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier(n_bootstrap=50)

        # Classification probabilities (100 samples, 3 classes)
        predictions = np.random.dirichlet(np.ones(3), size=100)

        metrics = quantifier.quantify(predictions)

        assert metrics.entropy >= 0
        assert len(metrics.confidence_interval) == 2

    def test_uncertainty_decomposition(self):
        """Test uncertainty decomposition."""
        from src.consciousness.meta_awareness import compute_epistemic_uncertainty, compute_aleatoric_uncertainty

        predictions = np.random.rand(100)

        epistemic = compute_epistemic_uncertainty(predictions, n_bootstrap=50)
        aleatoric = compute_aleatoric_uncertainty(predictions)

        assert epistemic >= 0
        assert aleatoric >= 0


class TestIntegration:
    """Integration tests for consciousness modules."""

    def test_full_pipeline(self):
        """Test complete consciousness analysis pipeline."""
        from src.consciousness.latent_space import ConsciousnessMapper, LatentSpaceVisualizer
        from src.consciousness.attention import AttentionAnalyzer
        from src.consciousness.meta_awareness import IntrospectionDetector

        # Generate test data
        n_sessions = 20
        n_features = 50
        fs = 256.0

        features = np.random.randn(n_sessions, n_features)
        labels = ["meditation"] * 10 + ["focus"] * 10
        eeg_data = np.random.randn(4, int(fs * 5))  # 5 seconds

        # Latent space mapping
        mapper = ConsciousnessMapper(embedding_dim=16, use_semantic=False)
        embeddings = mapper.fit_transform(features, labels)

        # Visualization
        visualizer = LatentSpaceVisualizer(method="pca", n_components=2)
        coords = visualizer.fit_transform(embeddings)

        # Attention analysis
        attention_analyzer = AttentionAnalyzer(fs)
        gw_metrics = attention_analyzer.compute_global_workspace(eeg_data)

        # Meta-awareness
        detector = IntrospectionDetector(fs)
        state = detector.detect(eeg_data)

        # Assertions
        assert embeddings.shape == (n_sessions, 16)
        assert coords.shape == (n_sessions, 2)
        assert gw_metrics.integration >= 0
        assert state.metacognitive_index >= 0

    def test_state_comparison(self):
        """Test comparing different consciousness states."""
        from src.consciousness.meta_awareness import IntrospectionDetector

        detector = IntrospectionDetector(fs=256.0)

        # Create different state data
        # Meditation: strong alpha
        t = np.arange(2560) / 256.0
        meditation_data = np.random.randn(4, 2560) * 5
        meditation_data += 20 * np.sin(2 * np.pi * 10 * t)

        # Focus: strong beta
        focus_data = np.random.randn(4, 2560) * 5
        focus_data += 20 * np.sin(2 * np.pi * 20 * t)

        meditation_state = detector.detect(meditation_data)
        focus_state = detector.detect(focus_data)

        # Both should produce valid results
        assert 0 <= meditation_state.metacognitive_index <= 1
        assert 0 <= focus_state.metacognitive_index <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
