#!/usr/bin/env python3
"""End-to-end analysis pipeline example.

Demonstrates the complete workflow from synthetic EEG data
to state classification.

Usage:
    python scripts/analysis_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.analysis.signal import Preprocessor, SignalQualityAssessor
from src.analysis.features import FeatureExtractor
from src.analysis.classification import (
    StateClassifier,
    ClassificationPipeline,
    evaluate_classifier,
    create_train_test_split,
)


def generate_synthetic_eeg(
    n_sessions: int = 20,
    n_channels: int = 4,
    duration: float = 10.0,
    fs: float = 256.0,
    state: str = "neutral",
) -> list[np.ndarray]:
    """Generate synthetic EEG data for testing.

    Different states have different frequency characteristics:
    - meditation: Strong alpha (8-13 Hz)
    - focus: Strong beta (13-30 Hz)
    - neutral: Balanced spectrum

    Args:
        n_sessions: Number of sessions to generate.
        n_channels: Number of channels.
        duration: Duration in seconds.
        fs: Sampling frequency.
        state: Consciousness state.

    Returns:
        List of EEG arrays (channels, samples).
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    sessions = []
    for _ in range(n_sessions):
        # Base signal (pink noise)
        signal = np.random.randn(n_channels, n_samples) * 10

        # Add state-specific characteristics
        if state == "meditation":
            # Strong alpha waves (10 Hz)
            alpha = 30 * np.sin(2 * np.pi * 10 * t)
            signal += alpha

        elif state == "focus":
            # Strong beta waves (20 Hz)
            beta = 20 * np.sin(2 * np.pi * 20 * t)
            signal += beta

        elif state == "flow":
            # Alpha + some theta
            alpha = 25 * np.sin(2 * np.pi * 10 * t)
            theta = 15 * np.sin(2 * np.pi * 6 * t)
            signal += alpha + theta

        # Add some noise
        signal += np.random.randn(n_channels, n_samples) * 5

        sessions.append(signal)

    return sessions


def main():
    """Run end-to-end analysis example."""
    print("\n" + "=" * 60)
    print("  Consciousness Research Workbench - Analysis Example")
    print("=" * 60 + "\n")

    # Parameters
    fs = 256
    n_channels = 4
    channel_names = ["TP9", "AF7", "AF8", "TP10"]

    # =================================================================
    # 1. Generate synthetic data
    # =================================================================
    print("1. Generating synthetic EEG data...")

    # Generate sessions for different states
    states = ["meditation", "focus", "neutral"]
    all_sessions = []
    all_labels = []

    for state in states:
        sessions = generate_synthetic_eeg(
            n_sessions=10,
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
    # 2. Signal quality assessment
    # =================================================================
    print("\n2. Assessing signal quality...")

    quality_assessor = SignalQualityAssessor(fs)
    sample_session = all_sessions[0]

    report = quality_assessor.generate_report(sample_session, channel_names)
    print(f"   Overall quality: {report['overall_label']} ({report['overall_score']:.1f}/100)")

    for ch_name, ch_report in report["channels"].items():
        print(f"   {ch_name}: SNR={ch_report['snr']:.1f}dB, Quality={ch_report['quality_label']}")

    # =================================================================
    # 3. Preprocessing
    # =================================================================
    print("\n3. Preprocessing signals...")

    preprocessor = Preprocessor(fs)
    preprocessor.configure(
        notch_freq=60,
        bandpass=(1, 50),
        remove_artifacts=True,
        baseline_correct=True,
    )

    # Preprocess all sessions
    processed_sessions = []
    for session in all_sessions:
        processed = preprocessor.process(session)
        processed_sessions.append(processed)

    print(f"   Preprocessed {len(processed_sessions)} sessions")
    print(f"   Pipeline: {preprocessor.history[-1]['steps'] if preprocessor.history else 'N/A'}")

    # =================================================================
    # 4. Feature extraction
    # =================================================================
    print("\n4. Extracting features...")

    extractor = FeatureExtractor(fs, channel_names)

    all_features = []
    for session in processed_sessions:
        features = extractor.extract(session, flatten=True)
        all_features.append(features)

    # Convert to array
    import pandas as pd
    df = pd.DataFrame(all_features)

    print(f"   Extracted {len(df.columns)} features per session")
    print(f"   Feature types: spectral, entropy, statistical, connectivity")

    # Show sample features
    sample_features = ["spectral_alpha_abs_TP9", "entropy_spectral_entropy_TP9", "stat_hjorth_mobility_TP9"]
    for feat in sample_features:
        if feat in df.columns:
            print(f"   {feat}: {df[feat].iloc[0]:.4f}")

    # =================================================================
    # 5. Train/test split
    # =================================================================
    print("\n5. Creating train/test split...")

    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values

    X_train, X_test, y_train, y_test = create_train_test_split(
        X, all_labels, test_size=0.3, random_state=42
    )

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # =================================================================
    # 6. Train classifier
    # =================================================================
    print("\n6. Training classifier...")

    classifier = StateClassifier(
        model_type="random_forest",
        n_estimators=100,
        random_state=42,
    )

    classifier.fit(X_train, y_train)

    print(f"   Model: {classifier.model_type}")
    print(f"   Classes: {classifier.classes}")

    # =================================================================
    # 7. Evaluate classifier
    # =================================================================
    print("\n7. Evaluating classifier...")

    # Predictions
    y_pred = classifier.predict(X_test)

    # Evaluation
    report = evaluate_classifier(y_test, y_pred)

    print(f"   Accuracy: {report.accuracy:.2%}")
    print(f"   Precision: {report.precision:.2%}")
    print(f"   Recall: {report.recall:.2%}")
    print(f"   F1 Score: {report.f1:.2%}")

    print("\n   Confusion Matrix:")
    print(f"   {report.confusion_matrix}")

    # =================================================================
    # 8. Feature importance
    # =================================================================
    print("\n8. Top 10 most important features:")

    importance = classifier.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for i, (name, imp) in enumerate(sorted_importance[:10], 1):
        # Map feature index to name
        if name.startswith("feature_"):
            idx = int(name.split("_")[1])
            if idx < len(numeric_cols):
                name = numeric_cols[idx]
        print(f"   {i}. {name}: {imp:.4f}")

    # =================================================================
    # 9. Cross-validation
    # =================================================================
    print("\n9. Cross-validation (5-fold)...")

    cv_results = classifier.cross_validate(X, all_labels, cv=5)

    print(f"   Mean accuracy: {cv_results['mean']:.2%}")
    print(f"   Std: {cv_results['std']:.2%}")

    # =================================================================
    # 10. Save model
    # =================================================================
    print("\n10. Saving classifier...")

    model_path = Path("data/models/state_classifier.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    classifier.save(model_path)
    print(f"   Saved to: {model_path}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)
    print(f"""
Summary:
- Processed {len(all_sessions)} EEG sessions
- Extracted {len(numeric_cols)} features per session
- Trained Random Forest classifier
- Test accuracy: {report.accuracy:.2%}
- Cross-validation: {cv_results['mean']:.2%} (+/- {cv_results['std']:.2%})

Next steps:
- Collect real EEG data with Muse device
- Run scenarios (meditation, focus, neutral)
- Train on real data
- Iterate on features and models
""")


if __name__ == "__main__":
    main()
