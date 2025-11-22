# Consciousness Research Workbench

A unified platform for exploring AI consciousness through grounded human experience. Bridges embodied cognition (EEG data) with computational consciousness models (AI latent spaces, attention patterns, meta-awareness).

## Vision

Map the relationship between human consciousness states and AI internal representations by:
- Collecting multimodal EEG data during real-world scenarios
- Analyzing neural patterns with signal processing and ML
- Comparing human states with AI latent spaces and attention patterns
- Developing metrics for machine consciousness

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              Consciousness Research Workbench                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │           DATA FOUNDATION LAYER                 │            │
│  │  EEG (Muse) • GPS • Audio • Timestamps          │            │
│  │  HDF5 Storage • Scenario Execution              │            │
│  └─────────────────────┬───────────────────────────┘            │
│                        │                                        │
│  ┌─────────────────────▼───────────────────────────┐            │
│  │         PROCESSING & ANALYSIS CORE              │            │
│  │  Signal Processing • Feature Extraction          │            │
│  │  State Classification • Pattern Recognition      │            │
│  └─────────────────────┬───────────────────────────┘            │
│                        │                                        │
│  ┌─────────────────────▼───────────────────────────┐            │
│  │        AI CONSCIOUSNESS MODULES                  │            │
│  │  Latent Space Mapper • Attention Analyzer        │            │
│  │  Meta-Awareness • Qualia • Emergence Metrics     │            │
│  └─────────────────────┬───────────────────────────┘            │
│                        │                                        │
│  ┌─────────────────────▼───────────────────────────┐            │
│  │        INTEGRATIONS & VISUALIZATION             │            │
│  │  Dashboard • Reports • API • 3D Viewer           │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
consciousness-research-workbench/
├── src/
│   ├── foundation/          # TIER 0: Data collection
│   │   ├── eeg/             # EEG device handling (BrainFlow)
│   │   ├── data/            # Storage (HDF5, Parquet)
│   │   ├── scenarios/       # Scenario execution
│   │   ├── sensors/         # GPS, Audio integration
│   │   ├── config/          # Configuration management
│   │   └── utils/           # Utilities
│   │
│   ├── analysis/            # TIER 1: Signal processing
│   │   ├── signal/          # Filtering, preprocessing
│   │   ├── features/        # PSD, coherence, entropy
│   │   └── classification/  # State classification
│   │
│   ├── consciousness/       # TIER 2-3: AI modules
│   │   ├── latent_space/    # EEG → embedding projection
│   │   ├── attention/       # Attention pattern analysis
│   │   ├── meta_awareness/  # Meta-awareness detection
│   │   ├── qualia/          # Cross-modal synthesis
│   │   └── emergence/       # Complexity metrics
│   │
│   ├── integrations/        # TIER 4: User experience
│   │   ├── dashboard/       # Streamlit/Gradio UI
│   │   ├── reports/         # Report generation
│   │   └── api/             # REST/GraphQL
│   │
│   └── plugins/             # Extensibility layer
│       ├── base/            # Abstract interfaces
│       ├── registry/        # Plugin management
│       └── examples/        # Example plugins
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # Detailed architecture
│   └── ROADMAP.md           # Implementation timeline
│
├── experiments/             # Experiment results
├── scenarios/               # Scenario definitions
├── scripts/                 # CLI tools
├── tests/                   # Unit tests
├── data/                    # Local storage (gitignored)
│
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- Windows 10/11 (primary), macOS, or Linux
- Muse 2/S EEG headband (or use synthetic data for testing)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd consciousness-research-workbench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### First Run

```bash
# Test EEG connection (synthetic mode)
python scripts/test_eeg_connection.py --device synthetic

# Run your first scenario
python scripts/run_scenario.py urban_walk

# List available scenarios
python scripts/run_scenario.py --list

# Run analysis example
python scripts/analysis_example.py
```

## TIER 1: Analysis Usage

### Signal Processing

```python
from src.analysis.signal import Preprocessor, SignalQualityAssessor

# Create preprocessor
preprocessor = Preprocessor(fs=256)
preprocessor.configure(
    notch_freq=60,          # Remove power line noise
    bandpass=(1, 50),       # Keep 1-50 Hz
    remove_artifacts=True,  # Detect and remove artifacts
    baseline_correct=True,  # Zero-mean
)

# Process EEG data
clean_eeg = preprocessor.process(raw_eeg)

# Assess signal quality
assessor = SignalQualityAssessor(fs=256)
report = assessor.generate_report(clean_eeg)
print(f"Quality: {report['overall_label']} ({report['overall_score']}/100)")
```

### Feature Extraction

```python
from src.analysis.features import FeatureExtractor, compute_band_power

# Extract all features
extractor = FeatureExtractor(fs=256, channel_names=["TP9", "AF7", "AF8", "TP10"])
features = extractor.extract(eeg_data, flatten=True)

# Or extract specific features
band_powers = compute_band_power(eeg_data, fs=256)
print(f"Alpha power: {band_powers['alpha']}")
print(f"Beta power: {band_powers['beta']}")
```

### State Classification

```python
from src.analysis.classification import StateClassifier, evaluate_classifier

# Create and train classifier
classifier = StateClassifier(model_type="random_forest", n_estimators=100)
classifier.fit(X_train, y_train)

# Predict states
predictions = classifier.predict(X_test)

# Evaluate
report = evaluate_classifier(y_test, predictions)
print(f"Accuracy: {report.accuracy:.2%}")

# Get feature importance
importance = classifier.get_feature_importance()

# Save model
classifier.save("models/state_classifier.pkl")
```

### Complete Pipeline

```python
from src.analysis import ClassificationPipeline

# Create end-to-end pipeline
pipeline = ClassificationPipeline(
    fs=256,
    channel_names=["TP9", "AF7", "AF8", "TP10"],
    classifier_type="random_forest",
)

# Fit on sessions
pipeline.fit(eeg_sessions, labels)

# Predict on new data
predictions = pipeline.predict(new_eeg)

# Save pipeline
pipeline.save("models/pipeline.pkl")
```

## Implementation Tiers

### TIER 0: Foundation (Complete)
- [x] EEG device connection (Muse 2/S via BrainFlow)
- [x] HDF5/Parquet storage
- [x] Configuration system
- [x] Scenario runner with event markers
- [x] GPS and audio skeletons

### TIER 1: Analysis Core (Complete)
- [x] Signal processing (filtering, artifact removal)
- [x] Feature extraction (PSD, coherence, entropy)
- [x] State classification (Random Forest, evaluation)

### TIER 2: AI Consciousness (Planned)
- [ ] Latent space mapper (EEG → embeddings)
- [ ] Attention pattern analyzer
- [ ] Meta-awareness detector

### TIER 3: Advanced Research (Planned)
- [ ] Qualia synthesizer
- [ ] Temporal binding tracker
- [ ] Emergence metrics (IIT)

### TIER 4: User Experience (Planned)
- [ ] Real-time dashboard
- [ ] Report generator
- [ ] 3D latent space viewer

See [ROADMAP.md](docs/ROADMAP.md) for detailed implementation timeline.

## Research Scenarios

| Scenario | Description | Focus Area |
|----------|-------------|------------|
| `urban_walk` | City navigation | Environmental response |
| `creative_flow` | Creative work (Blender) | Flow state detection |
| `social_interaction` | Social contexts | Emotional responses |
| `problem_solving` | Coding, design | Cognitive load |
| `meditation` | Calm baseline | Reference state |

## Plugin System

Extend the workbench with custom modules:

```python
from src.plugins import AnalysisPlugin, PluginMetadata, get_registry

class MyCustomPlugin(AnalysisPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="MyCustomPlugin",
            version="1.0.0",
            author="Your Name",
            description="Custom analysis",
            plugin_type="analysis"
        )

    def initialize(self, config):
        pass

    def cleanup(self):
        pass

    def process(self, data):
        # Your analysis logic
        return results

# Register and use
registry = get_registry()
registry.register(MyCustomPlugin)
plugin = registry.get_plugin("MyCustomPlugin", "analysis")
```

Plugin types:
- **DataCollectorPlugin**: New sensors/data sources
- **AnalysisPlugin**: Signal processing, features
- **ConsciousnessPlugin**: AI introspection
- **VisualizationPlugin**: Custom visualizations
- **ExternalServicePlugin**: Third-party integrations

## Data Privacy

- **Local-only**: All data stays on your machine
- **No telemetry**: Zero external connections
- **Gitignored**: Raw data never committed
- **Anonymization**: Optional device serial stripping

## Configuration

Edit `config.yaml` for device and storage settings:

```yaml
eeg:
  device_type: "muse_2"     # or "synthetic" for testing
  brainflow:
    board_id: 22
    sampling_rate: 256

storage:
  raw:
    format: "hdf5"
    compression: "gzip"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement with tests
4. Submit a pull request

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions.

## Tech Stack

**Core:** Python 3.10+, BrainFlow, NumPy, SciPy
**Storage:** HDF5, Parquet, Pandas
**ML:** Scikit-learn, PyTorch (planned)
**Viz:** Plotly, Matplotlib, Streamlit (planned)
**AI:** Anthropic API, Sentence-transformers (planned)

## References

- [BrainFlow Documentation](https://brainflow.ai/)
- [Muse Developer Resources](https://choosemuse.com/)
- [Integrated Information Theory](https://en.wikipedia.org/wiki/Integrated_information_theory)
- [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory)

## License

MIT License - see LICENSE file for details.

---

**Research Focus**: Mapping the relationship between human consciousness states and AI internal representations through grounded, embodied experience.
