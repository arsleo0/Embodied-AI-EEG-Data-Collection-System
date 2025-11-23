"""Tests for TIER 3 advanced research modules."""

import numpy as np
import pytest


class TestComplexityAnalyzer:
    """Tests for complexity metrics."""

    def test_complexity_analyzer_basic(self):
        """Test basic complexity analysis."""
        from src.consciousness.emergence import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(n_scales=5)
        data = np.random.randn(4, 1024)

        metrics = analyzer.analyze(data)

        assert 0 <= metrics.lempel_ziv <= 1
        assert 0 <= metrics.kolmogorov_estimate <= 1
        assert 1 <= metrics.fractal_dimension <= 2
        assert len(metrics.multiscale_entropy) == 5

    def test_lempel_ziv_complexity(self):
        """Test Lempel-Ziv complexity."""
        from src.consciousness.emergence import compute_lempel_ziv_complexity

        # Random signal should have higher complexity than periodic
        random_signal = np.random.randn(1000)
        periodic_signal = np.sin(np.linspace(0, 20 * np.pi, 1000))

        lz_random = compute_lempel_ziv_complexity(random_signal)
        lz_periodic = compute_lempel_ziv_complexity(periodic_signal)

        assert lz_random > lz_periodic

    def test_fractal_dimension(self):
        """Test fractal dimension calculation."""
        from src.consciousness.emergence import compute_fractal_dimension

        signal = np.random.randn(1000)
        fd = compute_fractal_dimension(signal)

        assert 1 <= fd <= 2

    def test_multiscale_entropy(self):
        """Test multi-scale entropy."""
        from src.consciousness.emergence import compute_multiscale_entropy

        signal = np.random.randn(1000)
        mse = compute_multiscale_entropy(signal, n_scales=5)

        assert len(mse) == 5
        assert all(e >= 0 for e in mse)


class TestIITAnalyzer:
    """Tests for IIT metrics."""

    def test_iit_analyzer_basic(self):
        """Test basic IIT analysis."""
        from src.consciousness.emergence import IITAnalyzer

        analyzer = IITAnalyzer(n_bins=10)
        data = np.random.randn(4, 1000)

        metrics = analyzer.analyze(data)

        assert 0 <= metrics.phi <= 1
        assert 0 <= metrics.integration <= 1
        assert 0 <= metrics.differentiation <= 1
        assert 0 <= metrics.exclusion <= 1

    def test_phi_approximation(self):
        """Test Phi approximation."""
        from src.consciousness.emergence import compute_phi_approximation

        data = np.random.randn(4, 1000)
        phi = compute_phi_approximation(data)

        assert phi >= 0

    def test_integration_differentiation_balance(self):
        """Test integration-differentiation balance."""
        from src.consciousness.emergence import compute_integration_differentiation_balance

        data = np.random.randn(4, 1000)
        balance = compute_integration_differentiation_balance(data)

        assert "integration" in balance
        assert "differentiation" in balance
        assert "balance" in balance
        assert 0 <= balance["balance"] <= 1


class TestPhaseTransitionDetector:
    """Tests for phase transition detection."""

    def test_phase_transition_basic(self):
        """Test basic phase transition analysis."""
        from src.consciousness.emergence import PhaseTransitionDetector

        detector = PhaseTransitionDetector(fs=256)
        data = np.random.randn(4, 2560)  # 10 seconds

        metrics = detector.analyze(data)

        assert 0 <= metrics.order_parameter <= 1
        assert metrics.susceptibility >= 0
        assert 0 <= metrics.correlation_length <= 1
        assert 0 <= metrics.critical_point <= 1
        assert metrics.phase in ["ordered", "critical", "disordered", "transitional"]

    def test_edge_of_chaos(self):
        """Test edge of chaos computation."""
        from src.consciousness.emergence import compute_edge_of_chaos

        data = np.random.randn(4, 2560)
        edge = compute_edge_of_chaos(data)

        assert 0 <= edge <= 1

    def test_avalanche_statistics(self):
        """Test avalanche statistics."""
        from src.consciousness.emergence import compute_avalanche_statistics

        data = np.random.randn(4, 2560)
        stats = compute_avalanche_statistics(data, threshold=1.5)

        assert "n_avalanches" in stats
        assert "mean_size" in stats
        assert stats["n_avalanches"] >= 0


class TestTemporalBindingTracker:
    """Tests for temporal binding analysis."""

    def test_temporal_binding_basic(self):
        """Test basic temporal binding analysis."""
        from src.consciousness.emergence import TemporalBindingTracker

        tracker = TemporalBindingTracker(fs=256)
        data = np.random.randn(4, 2560)

        metrics = tracker.analyze(data)

        assert metrics.integration_window > 0
        assert 0 <= metrics.context_retention <= 1
        assert 0 <= metrics.binding_strength <= 1
        assert 0 <= metrics.temporal_coherence <= 1

    def test_specious_present(self):
        """Test specious present estimation."""
        from src.consciousness.emergence import estimate_specious_present

        data = np.random.randn(4, 2560)
        sp = estimate_specious_present(data, fs=256)

        assert 1.0 <= sp <= 5.0  # Typical range

    def test_working_memory_span(self):
        """Test working memory span computation."""
        from src.consciousness.emergence import compute_working_memory_span

        data = np.random.randn(4, 2560)
        wm = compute_working_memory_span(data, fs=256)

        assert "span_seconds" in wm
        assert "span_items" in wm
        assert wm["span_items"] >= 1


class TestTransitionDetector:
    """Tests for state transition detection."""

    def test_transition_detection_basic(self):
        """Test basic transition detection."""
        from src.consciousness.emergence import TransitionDetector

        detector = TransitionDetector(threshold=0.5)

        embeddings = np.random.randn(20, 16)
        timestamps = np.arange(20).astype(float)
        labels = ["a"] * 10 + ["b"] * 10

        analysis = detector.analyze(embeddings, timestamps, labels)

        assert 0 <= analysis.mean_smoothness <= 1
        assert analysis.hysteresis >= 0

    def test_transition_entropy(self):
        """Test transition entropy computation."""
        from src.consciousness.emergence import compute_transition_entropy

        labels = ["a", "b", "a", "b", "c", "a"]
        entropy = compute_transition_entropy(labels)

        assert entropy >= 0

    def test_phase_transition_points(self):
        """Test phase transition point detection."""
        from src.consciousness.emergence import detect_phase_transition_points

        embeddings = np.random.randn(50, 16)
        points = detect_phase_transition_points(embeddings)

        assert isinstance(points, list)


class TestQualiaSynthesizer:
    """Tests for qualia synthesis."""

    def test_cross_modal_fusion(self):
        """Test cross-modal fusion."""
        from src.consciousness.qualia import CrossModalFusion

        fusion = CrossModalFusion(output_dim=64)

        modalities = {
            "eeg": np.random.randn(32),
            "audio": np.random.randn(24),
        }

        fused = fusion.fuse(modalities)

        assert len(fused) == 64

    def test_synesthesia_simulator(self):
        """Test synesthesia simulation."""
        from src.consciousness.qualia import SynesthesiaSimulator

        simulator = SynesthesiaSimulator()

        # Visual
        visual = simulator.state_to_visual("calm", resolution=(64, 64))
        assert "color" in visual
        assert "pattern" in visual
        assert visual["pattern"].shape == (64, 64, 3)

        # Audio
        audio = simulator.state_to_audio("focus", duration=1.0)
        assert "waveform" in audio
        assert len(audio["waveform"]) > 0

        # Text
        text = simulator.state_to_text("anxiety")
        assert "description" in text

    def test_emotional_state_generator(self):
        """Test emotional state generator."""
        from src.consciousness.qualia import EmotionalStateGenerator

        generator = EmotionalStateGenerator()
        features = np.random.randn(20)

        emotional = generator.from_eeg(features)

        assert -1 <= emotional["valence"] <= 1
        assert 0 <= emotional["arousal"] <= 1
        assert 0 <= emotional["dominance"] <= 1
        assert "emotion" in emotional

    def test_qualia_synthesizer_full(self):
        """Test full qualia synthesis."""
        from src.consciousness.qualia import QualiaSynthesizer

        synthesizer = QualiaSynthesizer(output_dim=64)
        features = np.random.randn(32)

        qualia = synthesizer.synthesize(features, state="focus")

        assert len(qualia.embedding) == 64
        assert -1 <= qualia.emotional_valence <= 1
        assert 0 <= qualia.arousal <= 1


class TestModalityMapper:
    """Tests for modality mapping."""

    def test_eeg_to_audio_mapper(self):
        """Test EEG to audio mapping."""
        from src.consciousness.qualia import EEGToAudioMapper

        mapper = EEGToAudioMapper(n_audio_features=32)

        eeg = np.random.randn(20, 50)
        audio = np.random.randn(20, 32)

        mapper.fit(eeg, audio)
        predicted = mapper.transform(eeg[0])

        assert len(predicted) == 32

    def test_eeg_to_visual_mapper(self):
        """Test EEG to visual mapping."""
        from src.consciousness.qualia import EEGToVisualMapper

        mapper = EEGToVisualMapper(n_visual_features=32)

        eeg = np.random.randn(20, 50)
        visual = np.random.randn(20, 32)

        mapper.fit(eeg, visual)
        predicted = mapper.transform(eeg[0])

        assert len(predicted) == 32

    def test_cross_modal_translator(self):
        """Test cross-modal translator."""
        from src.consciousness.qualia import CrossModalTranslator

        translator = CrossModalTranslator()

        eeg = np.random.randn(20, 50)
        audio = np.random.randn(20, 32)

        translator.fit("eeg", eeg, "audio", audio)
        result = translator.translate("eeg", eeg[0], "audio")

        assert result.source_modality == "eeg"
        assert result.target_modality == "audio"
        assert len(result.features) == 32


class TestIntegration:
    """Integration tests for TIER 3 modules."""

    def test_full_emergence_pipeline(self):
        """Test complete emergence analysis pipeline."""
        from src.consciousness.emergence import (
            ComplexityAnalyzer,
            IITAnalyzer,
            PhaseTransitionDetector,
            TemporalBindingTracker,
        )

        # Generate test data
        data = np.random.randn(4, 2560)  # 10 seconds

        # Complexity
        complexity = ComplexityAnalyzer()
        c_metrics = complexity.analyze(data)

        # IIT
        iit = IITAnalyzer()
        iit_metrics = iit.analyze(data)

        # Phase transitions
        phase = PhaseTransitionDetector()
        p_metrics = phase.analyze(data)

        # Temporal binding
        binding = TemporalBindingTracker()
        b_metrics = binding.analyze(data)

        # Verify all metrics
        assert c_metrics.overall >= 0
        assert iit_metrics.phi >= 0
        assert p_metrics.phase in ["ordered", "critical", "disordered", "transitional"]
        assert b_metrics.integration_window > 0

    def test_qualia_from_eeg(self):
        """Test qualia synthesis from EEG features."""
        from src.consciousness.qualia import (
            QualiaSynthesizer,
            EmotionalStateGenerator,
        )
        from src.consciousness.emergence import (
            ComplexityAnalyzer,
        )

        # Generate EEG-like features
        data = np.random.randn(4, 1024)

        # Compute complexity features
        complexity = ComplexityAnalyzer()
        c_metrics = complexity.analyze(data)

        # Use as features for qualia synthesis
        features = np.array([
            c_metrics.lempel_ziv,
            c_metrics.kolmogorov_estimate,
            c_metrics.fractal_dimension,
            c_metrics.overall,
        ] + list(c_metrics.multiscale_entropy[:6]))

        # Synthesize qualia
        synthesizer = QualiaSynthesizer()
        qualia = synthesizer.synthesize(features, state="focus")

        # Get emotional state
        emotion_gen = EmotionalStateGenerator()
        emotional = emotion_gen.from_eeg(features)

        # Verify
        assert len(qualia.embedding) > 0
        assert emotional["emotion"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
