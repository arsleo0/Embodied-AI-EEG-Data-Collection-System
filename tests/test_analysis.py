"""Tests for analysis modules."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSignalProcessing:
    """Tests for signal processing module."""

    def test_bandpass_filter(self):
        """Test bandpass filter."""
        from src.analysis.signal import bandpass_filter

        # Generate test signal
        fs = 256
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz

        # Apply filter
        filtered = bandpass_filter(signal, 5, 15, fs)

        # Signal should pass through (mostly unchanged amplitude)
        assert len(filtered) == len(signal)
        assert np.max(np.abs(filtered)) > 0.5

    def test_notch_filter(self):
        """Test notch filter."""
        from src.analysis.signal import notch_filter

        # Generate signal with 60 Hz component
        fs = 256
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 60 * t)

        # Apply notch filter
        filtered = notch_filter(signal, 60, fs)

        # 60 Hz should be attenuated
        assert np.max(np.abs(filtered)) < 0.5

    def test_filter_bank(self):
        """Test filter bank chaining."""
        from src.analysis.signal import FilterBank

        # Create filter bank
        fb = FilterBank(fs=256)
        fb.add_notch(60)
        fb.add_bandpass(1, 50)

        # Generate test signal
        signal = np.random.randn(4, 1000)

        # Apply filters
        filtered = fb.apply(signal)

        assert filtered.shape == signal.shape

    def test_artifact_detector(self):
        """Test artifact detection."""
        from src.analysis.signal import ArtifactDetector

        # Generate signal with artifact
        signal = np.random.randn(4, 1000) * 10

        # Add large amplitude artifact (eye blink)
        signal[0, 200:250] = 500

        # Detect artifacts
        detector = ArtifactDetector(fs=256)
        artifacts = detector.detect_all(signal)

        # Should find at least one artifact
        assert len(artifacts) > 0

    def test_preprocessor(self):
        """Test preprocessing pipeline."""
        from src.analysis.signal import Preprocessor

        # Create preprocessor
        preprocessor = Preprocessor(fs=256)
        preprocessor.configure(
            notch_freq=60,
            bandpass=(1, 50),
            baseline_correct=True,
        )

        # Generate test signal
        signal = np.random.randn(4, 1000)

        # Process
        processed = preprocessor.process(signal)

        assert processed.shape == signal.shape

    def test_signal_quality(self):
        """Test signal quality assessment."""
        from src.analysis.signal import SignalQualityAssessor

        # Create assessor
        assessor = SignalQualityAssessor(fs=256)

        # Generate clean signal
        signal = np.random.randn(4, 1000) * 10

        # Assess quality
        report = assessor.generate_report(signal)

        assert "overall_score" in report
        assert "overall_label" in report
        assert report["overall_score"] >= 0


class TestFeatureExtraction:
    """Tests for feature extraction module."""

    def test_compute_psd(self):
        """Test PSD computation."""
        from src.analysis.features import compute_psd

        # Generate signal
        signal = np.random.randn(1000)

        # Compute PSD
        freqs, psd = compute_psd(signal, fs=256)

        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)

    def test_compute_band_power(self):
        """Test band power computation."""
        from src.analysis.features import compute_band_power

        # Generate signal with strong alpha
        fs = 256
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz = alpha

        # Compute band powers
        powers = compute_band_power(signal, fs)

        assert "alpha" in powers
        assert "delta" in powers
        assert "beta" in powers

        # Alpha should be dominant
        assert powers["alpha"] > powers["delta"]

    def test_spectral_entropy(self):
        """Test spectral entropy."""
        from src.analysis.features import spectral_entropy

        # Generate signal
        signal = np.random.randn(1000)

        # Compute entropy
        entropy = spectral_entropy(signal, fs=256)

        assert 0 <= entropy <= 1

    def test_hjorth_parameters(self):
        """Test Hjorth parameters."""
        from src.analysis.features import compute_hjorth_parameters

        # Generate signal
        signal = np.random.randn(1000)

        # Compute Hjorth
        hjorth = compute_hjorth_parameters(signal)

        assert "hjorth_activity" in hjorth
        assert "hjorth_mobility" in hjorth
        assert "hjorth_complexity" in hjorth

    def test_statistical_features(self):
        """Test statistical features."""
        from src.analysis.features import compute_statistical_features

        # Generate signal
        signal = np.random.randn(4, 1000)

        # Compute features
        features = compute_statistical_features(signal)

        assert "mean" in features
        assert "std" in features
        assert "skewness" in features
        assert "kurtosis" in features

    def test_feature_extractor(self):
        """Test complete feature extraction."""
        from src.analysis.features import FeatureExtractor

        # Create extractor
        extractor = FeatureExtractor(fs=256)

        # Generate signal
        signal = np.random.randn(4, 1000)

        # Extract features
        features = extractor.extract(signal)

        assert len(features) > 0
        assert any("spectral" in k for k in features.keys())
        assert any("entropy" in k for k in features.keys())

    def test_coherence(self):
        """Test coherence computation."""
        from src.analysis.features import compute_coherence

        # Generate correlated signals
        signal1 = np.random.randn(1000)
        signal2 = signal1 + np.random.randn(1000) * 0.1
        data = np.vstack([signal1, signal2])

        # Compute coherence
        coh = compute_coherence(data, fs=256)

        assert "alpha" in coh
        # Channels should be coherent
        assert coh["alpha"][0, 1] > 0.5


class TestClassification:
    """Tests for classification module."""

    def test_state_classifier(self):
        """Test state classifier."""
        from src.analysis.classification import StateClassifier

        # Generate synthetic data
        X = np.random.randn(100, 50)
        y = ["meditation"] * 50 + ["focus"] * 50

        # Create and fit classifier
        classifier = StateClassifier(n_estimators=10)
        classifier.fit(X, y)

        assert classifier.is_fitted
        assert len(classifier.classes) == 2

    def test_classifier_predict(self):
        """Test classifier prediction."""
        from src.analysis.classification import StateClassifier

        # Generate data
        X_train = np.random.randn(100, 50)
        y_train = ["meditation"] * 50 + ["focus"] * 50

        # Fit
        classifier = StateClassifier(n_estimators=10)
        classifier.fit(X_train, y_train)

        # Predict
        X_test = np.random.randn(10, 50)
        predictions = classifier.predict(X_test)

        assert len(predictions) == 10
        assert all(p in ["meditation", "focus"] for p in predictions)

    def test_classifier_save_load(self):
        """Test classifier save/load."""
        from src.analysis.classification import StateClassifier
        import tempfile

        # Create and fit
        X = np.random.randn(100, 50)
        y = ["meditation"] * 50 + ["focus"] * 50

        classifier = StateClassifier(n_estimators=10)
        classifier.fit(X, y)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        classifier.save(path)

        # Load
        loaded = StateClassifier.load(path)

        assert loaded.is_fitted
        assert loaded.classes == classifier.classes

        # Clean up
        Path(path).unlink()

    def test_evaluate_classifier(self):
        """Test classifier evaluation."""
        from src.analysis.classification import evaluate_classifier

        # Generate predictions
        y_true = ["meditation"] * 10 + ["focus"] * 10
        y_pred = ["meditation"] * 8 + ["focus"] * 2 + ["focus"] * 9 + ["meditation"] * 1

        # Evaluate
        report = evaluate_classifier(y_true, y_pred)

        assert 0 <= report.accuracy <= 1
        assert report.confusion_matrix.shape == (2, 2)

    def test_train_test_split(self):
        """Test train/test split."""
        from src.analysis.classification import create_train_test_split

        # Generate data
        X = np.random.randn(100, 50)
        y = ["a"] * 50 + ["b"] * 50

        # Split
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=0.2
        )

        assert len(X_train) == 80
        assert len(X_test) == 20


class TestIntegration:
    """Integration tests for the analysis pipeline."""

    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        from src.analysis.signal import Preprocessor
        from src.analysis.features import FeatureExtractor
        from src.analysis.classification import StateClassifier

        # Generate synthetic data
        fs = 256
        n_sessions = 20
        sessions = []
        labels = []

        for _ in range(n_sessions // 2):
            # Meditation (strong alpha)
            t = np.arange(1000) / fs
            signal = np.random.randn(4, 1000) + 20 * np.sin(2 * np.pi * 10 * t)
            sessions.append(signal)
            labels.append("meditation")

            # Focus (strong beta)
            signal = np.random.randn(4, 1000) + 15 * np.sin(2 * np.pi * 20 * t)
            sessions.append(signal)
            labels.append("focus")

        # Preprocess
        preprocessor = Preprocessor(fs)
        processed = [preprocessor.process(s) for s in sessions]

        # Extract features
        extractor = FeatureExtractor(fs)
        features_list = [extractor.extract(s, flatten=True) for s in processed]

        # Convert to array
        import pandas as pd
        df = pd.DataFrame(features_list)
        X = df.select_dtypes(include=[np.number]).values

        # Classify
        classifier = StateClassifier(n_estimators=50)
        classifier.fit(X, labels)

        # Score
        score = classifier.score(X, labels)

        # Should achieve reasonable accuracy on training data
        assert score > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
