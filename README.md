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

# Run consciousness analysis example (TIER 2)
python scripts/consciousness_example.py
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

## TIER 2: Consciousness Analysis

### Latent Space Mapping

```python
from src.consciousness.latent_space import (
    ConsciousnessMapper,
    LatentSpaceVisualizer,
    TrajectoryTracker,
)

# Map EEG features to consciousness embedding space
mapper = ConsciousnessMapper(embedding_dim=32, use_semantic=False)
embeddings = mapper.fit_transform(features, state_labels)

# Find nearest consciousness state
nearest = mapper.find_nearest_state(embeddings[0], top_k=3)
for state, distance in nearest:
    print(f"{state}: {distance:.3f}")

# Visualize latent space with UMAP/t-SNE
visualizer = LatentSpaceVisualizer(method="umap", n_components=2)
coords = visualizer.fit_transform(embeddings)
fig = visualizer.plot_2d(coords, state_labels, title="Consciousness Space")
fig.write_html("consciousness_space.html")

# Track consciousness trajectory over time
tracker = TrajectoryTracker()
for i, embedding in enumerate(embeddings):
    tracker.add_point(embedding, timestamp=float(i), label=state_labels[i])

trajectory = tracker.get_trajectory()
metrics = tracker.compute_metrics()
print(f"Trajectory efficiency: {metrics['efficiency']:.3f}")
```

### Attention Pattern Analysis

```python
from src.consciousness.attention import (
    AttentionAnalyzer,
    AttentionComparator,
    compute_attention_entropy,
)

# Analyze Global Workspace metrics
analyzer = AttentionAnalyzer(fs=256)
gw_metrics = analyzer.compute_global_workspace(eeg_data)

print(f"Integration: {gw_metrics.integration:.3f}")
print(f"Differentiation: {gw_metrics.differentiation:.3f}")
print(f"Broadcast strength: {gw_metrics.broadcast_strength:.3f}")

# Compute attention entropy
entropy = compute_attention_entropy(eeg_data, fs=256)
print(f"Spatial entropy: {entropy['spatial_entropy']:.3f}")

# Compare human-AI attention patterns
comparator = AttentionComparator(n_regions=4)
alignment = comparator.compare(human_attention, ai_attention)
print(f"Correlation: {alignment.correlation:.3f}")
print(f"Overlap: {alignment.overlap:.3f}")
```

### Meta-Awareness Detection

```python
from src.consciousness.meta_awareness import (
    IntrospectionDetector,
    UncertaintyQuantifier,
    compute_metacognitive_index,
)

# Detect introspection and self-awareness
detector = IntrospectionDetector(fs=256, channel_names=["TP9", "AF7", "AF8", "TP10"])
state = detector.detect(eeg_data)

print(f"Self-awareness: {state.self_awareness:.3f}")
print(f"Introspection: {state.introspection:.3f}")
print(f"Metacognitive index: {state.metacognitive_index:.3f}")

# Quantify prediction uncertainty
quantifier = UncertaintyQuantifier(n_bootstrap=100)
uncertainty = quantifier.quantify(predictions, features)

print(f"Epistemic uncertainty: {uncertainty.epistemic:.3f}")
print(f"Aleatoric uncertainty: {uncertainty.aleatoric:.3f}")
print(f"Reliability: {uncertainty.reliability:.3f}")
```

### Run Consciousness Example

```bash
# Full consciousness analysis example
python scripts/consciousness_example.py
```

## TIER 3: Advanced Research

### Complexity and Emergence Metrics

```python
from src.consciousness.emergence import (
    ComplexityAnalyzer,
    IITAnalyzer,
    PhaseTransitionDetector,
    compute_lempel_ziv_complexity,
    compute_phi_approximation,
)

# Analyze signal complexity
complexity = ComplexityAnalyzer(n_scales=10)
metrics = complexity.analyze(eeg_data)
print(f"Lempel-Ziv: {metrics.lempel_ziv:.3f}")
print(f"Fractal dimension: {metrics.fractal_dimension:.3f}")

# IIT metrics (Phi approximation)
iit = IITAnalyzer(n_bins=10)
iit_metrics = iit.analyze(eeg_data)
print(f"Phi: {iit_metrics.phi:.4f}")
print(f"Integration: {iit_metrics.integration:.3f}")

# Phase transition detection
detector = PhaseTransitionDetector(fs=256)
phase = detector.analyze(eeg_data)
print(f"Phase: {phase.phase}")
print(f"Edge of chaos: {phase.critical_point:.3f}")
```

### Temporal Binding and State Transitions

```python
from src.consciousness.emergence import (
    TemporalBindingTracker,
    TransitionDetector,
    estimate_specious_present,
    compute_transition_entropy,
)

# Analyze temporal integration windows
tracker = TemporalBindingTracker(fs=256)
binding = tracker.analyze(eeg_data)
print(f"Integration window: {binding.integration_window:.3f}s")
print(f"Context retention: {binding.context_retention:.3f}")

# Estimate specious present (duration of "now")
sp = estimate_specious_present(eeg_data, fs=256)
print(f"Specious present: {sp:.2f}s")

# Detect state transitions
detector = TransitionDetector(threshold=0.5)
analysis = detector.analyze(embeddings, timestamps, labels)
print(f"Transitions: {len(analysis.transitions)}")
print(f"Smoothness: {analysis.mean_smoothness:.3f}")
```

### Qualia Synthesis

```python
from src.consciousness.qualia import (
    QualiaSynthesizer,
    SynesthesiaSimulator,
    EmotionalStateGenerator,
)

# Synesthesia: map state to sensory representations
simulator = SynesthesiaSimulator()

# What does "anxiety" look like?
visual = simulator.state_to_visual("anxiety", eeg_features)
print(f"Color: hue={visual['color']['hue']}°")

# What does "flow" sound like?
audio = simulator.state_to_audio("flow", eeg_features)
print(f"Frequency: {audio['parameters']['base_freq']}Hz")

# Generate emotional state from EEG
emotion_gen = EmotionalStateGenerator()
emotional = emotion_gen.from_eeg(features)
print(f"Emotion: {emotional['emotion']}")
print(f"Valence: {emotional['valence']:.2f}")

# Full qualia synthesis
synthesizer = QualiaSynthesizer(output_dim=128)
qualia = synthesizer.synthesize(eeg_features, state="meditation")
print(f"Arousal: {qualia.arousal:.3f}")
```

### Run Emergence Example

```bash
# Full emergence analysis example
python scripts/emergence_analysis_example.py
```

## TIER 4: User Experience & Visualization

### Real-time Dashboard

```bash
# Launch the dashboard
python scripts/launch_dashboard.py

# Launch in demo mode
python scripts/launch_dashboard.py --demo

# Specify port
python scripts/launch_dashboard.py --port 8502
```

The dashboard provides:
- Real-time EEG signal visualization
- Consciousness state monitoring
- Signal quality indicators
- Band power analysis
- Event marker timeline

### Report Generation

```python
from src.integrations.reports import (
    ReportGenerator,
    generate_session_report,
    generate_analysis_report,
)
from src.integrations.reports.templates import SessionTemplate
from src.integrations.reports.exporters import (
    HTMLExporter,
    MarkdownExporter,
    DataExporter,
)

# Generate session report
session_data = {
    "title": "Meditation Session",
    "session_id": "med_001",
    "start_time": datetime.now(),
    "duration": 300.0,
    "device": "Muse 2",
    "scenario": "Meditation",
    "channels": ["TP9", "AF7", "AF8", "TP10"],
    "sample_rate": 256,
    "n_samples": 76800,
    "quality": {"TP9": 0.9, "AF7": 0.85, "AF8": 0.88, "TP10": 0.82},
    "markers": [{"time": 0, "label": "start"}],
}

# HTML export
report_path = generate_session_report(session_data, "report.html", format="html")
print(f"Report saved to: {report_path}")

# JSON export
generate_session_report(session_data, "report.json", format="json")

# Markdown export
exporter = MarkdownExporter()
exporter.export(session_data, "report.md", report_type="session")

# Data export (CSV, Excel)
data_exporter = DataExporter()
data_exporter.to_csv(features, "features.csv")
data_exporter.to_excel(results, "results.xlsx")
```

Generate reports from command line:

```bash
# Generate demo report
python scripts/generate_report.py --demo

# Generate from session data
python scripts/generate_report.py session_data.json -o report.html

# Different formats
python scripts/generate_report.py session_data.json --format markdown
python scripts/generate_report.py session_data.json --format pdf
```

### 3D Latent Space Explorer

```python
from src.integrations.visualization import (
    LatentSpaceViewer,
    create_3d_scatter,
    create_trajectory_plot,
)

# Create viewer
viewer = LatentSpaceViewer()
viewer.set_data(embeddings, labels=state_labels, timestamps=timestamps)

# Generate 3D scatter plot
fig = viewer.create_figure()
fig.show()

# With trajectory
fig = viewer.create_figure(show_trajectory=True)
fig.write_html("latent_space.html")

# Animate trajectory over time
animation = viewer.animate_trajectory(duration=10.0)
animation.write_html("trajectory_animation.html")

# Quick plotting functions
fig = create_3d_scatter(embeddings, labels)
fig = create_trajectory_plot(embeddings, timestamps)
```

Command line visualization:

```bash
# Demo visualization
python scripts/visualize_latent_space.py --demo

# From saved embeddings
python scripts/visualize_latent_space.py embeddings.npz

# Show trajectory
python scripts/visualize_latent_space.py embeddings.npz --trajectory

# Launch interactive app
python scripts/visualize_latent_space.py --app
```

### Interactive Plots

```python
from src.integrations.visualization import (
    TimeSeriesPlot,
    SpectrogramPlot,
    TopoPlot,
    BandPowerPlot,
    create_dashboard_layout,
)

# Time series with zoom/pan
ts_plot = TimeSeriesPlot()
fig = ts_plot.plot(eeg_data, fs=256, channels=['TP9', 'AF7', 'AF8', 'TP10'])
fig.show()

# Interactive spectrogram
spec_plot = SpectrogramPlot()
fig = spec_plot.plot(signal, fs=256, freq_range=(0, 50))

# Topographic map
topo_plot = TopoPlot()
fig = topo_plot.plot({'TP9': 0.8, 'AF7': 0.9, 'AF8': 0.85, 'TP10': 0.75})

# Band power visualization
bp_plot = BandPowerPlot()
fig = bp_plot.plot_bars(band_powers, normalize=True)
fig = bp_plot.plot_radar(band_powers)
fig = bp_plot.plot_comparison([powers1, powers2], labels=["Session 1", "Session 2"])

# Complete dashboard layout
fig = create_dashboard_layout(eeg_data, fs=256, channels=channel_names)
fig.write_html("dashboard.html")
```

## TIER 5: Ecosystem & Extensibility

### REST API Server

```bash
# Start the API server
python scripts/start_api_server.py

# With custom settings
python scripts/start_api_server.py --port 8080 --debug

# Production with workers
python scripts/start_api_server.py --workers 4 --api-key YOUR_KEY
```

API documentation available at `http://localhost:8000/docs`.

### Using the API Client

```python
from src.integrations.api import WorkbenchClient

# Create client
client = WorkbenchClient("http://localhost:8000")

# Create and manage sessions
session = client.create_session("meditation_study", device="muse_2")
client.start_session(session["session_id"])
client.add_marker(session["session_id"], "eyes_closed")
client.stop_session(session["session_id"])

# Run analysis
analysis = client.run_analysis(session["session_id"], analysis_type="full")
print(f"Dominant state: {analysis['results']['dominant_state']}")

# Classify consciousness state
result = client.classify_state(features=[0.3, 0.25, 0.2, 0.15, 0.1])
print(f"State: {result['state']} ({result['confidence']:.0%})")

# Generate report
report = client.generate_report(session["session_id"], format="html")
```

### Experiment Tracking

```python
from src.infrastructure import (
    ExperimentTracker,
    get_tracker,
    log_params,
    log_metrics,
)

# Create tracker
tracker = ExperimentTracker("./experiments")

# Run experiment
with tracker.start_run("consciousness_study") as run:
    tracker.log_params({
        "model": "random_forest",
        "n_estimators": 100,
        "features": "band_powers",
    })

    # Your experiment code
    accuracy = train_and_evaluate()

    tracker.log_metric("accuracy", accuracy)
    tracker.log_artifact("model.pkl")

# Compare runs
comparison = tracker.compare_runs(
    "consciousness_study",
    run_ids=["run1", "run2"],
    metrics=["accuracy", "f1_score"]
)
```

### Data Provenance

```python
from src.infrastructure import (
    ProvenanceTracker,
    get_provenance_tracker,
)

# Track data lineage
tracker = get_provenance_tracker()

# Register raw data
tracker.register_data("raw_eeg", source="muse_device", data=raw_data)

# Track processing
with tracker.track_operation("filter", inputs=["raw_eeg"],
                             parameters={"lowcut": 1, "highcut": 50}):
    filtered = apply_filter(raw_data)
    tracker.register_output("filtered_eeg", filtered)

# Get lineage
lineage = tracker.get_lineage("filtered_eeg")
tracker.export_lineage("filtered_eeg", "lineage.md", format="markdown")
```

### Enhanced Plugin System

```python
from src.plugins import (
    get_loader,
    HookType,
    hook,
    invoke_hook,
)

# Load plugins dynamically
loader = get_loader()
loader.add_plugin_directory("./my_plugins")
loader.discover_plugins()

# Load specific plugin
plugin = loader.load_plugin("MyAnalyzer", config={"threshold": 0.5})

# Use hooks
@hook(HookType.PRE_PROCESS, priority=50)
def normalize_data(data):
    return (data - data.mean()) / data.std()

@hook(HookType.POST_ANALYSIS)
def log_results(results):
    print(f"Analysis complete: {results}")
    return results

# Invoke hooks
result = invoke_hook(HookType.PRE_PROCESS, data=eeg_data)
processed = result.modified_data
```

### Docker Deployment

```bash
# Build and deploy
python scripts/deploy.py build
python scripts/deploy.py up -d

# View logs
python scripts/deploy.py logs -f api

# Stop services
python scripts/deploy.py down
```

Or using docker-compose directly:

```bash
cd deployment
docker-compose up -d

# With MLflow tracking
docker-compose --profile with-mlflow up -d
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

### TIER 2: AI Consciousness (Complete)
- [x] Latent space mapper (EEG → embeddings)
- [x] Attention pattern analyzer (Global Workspace metrics)
- [x] Meta-awareness detector (introspection, uncertainty)

### TIER 3: Advanced Research (Complete)
- [x] Qualia synthesizer (cross-modal, synesthesia)
- [x] Temporal binding tracker (specious present, memory)
- [x] Emergence metrics (IIT, complexity, phase transitions)

### TIER 4: User Experience (Complete)
- [x] Real-time dashboard (Streamlit)
- [x] Report generator (HTML, JSON, Markdown, PDF)
- [x] 3D latent space viewer
- [x] Interactive visualization tools

### TIER 5: Ecosystem & Extensibility (Complete)
- [x] Enhanced plugin system with dynamic loading
- [x] Event hook system for pipeline integration
- [x] Experiment versioning and tracking
- [x] MLflow integration for ML experiments
- [x] Data provenance tracking
- [x] FastAPI REST API with WebSocket streaming
- [x] Docker deployment configuration

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
**Viz:** Plotly, Matplotlib, Streamlit
**Reports:** Jinja2, python-docx, openpyxl
**AI:** Sentence-transformers, UMAP
**API:** FastAPI, Uvicorn, WebSockets, Pydantic
**Deploy:** Docker, Docker Compose

## References

- [BrainFlow Documentation](https://brainflow.ai/)
- [Muse Developer Resources](https://choosemuse.com/)
- [Integrated Information Theory](https://en.wikipedia.org/wiki/Integrated_information_theory)
- [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory)

## License

MIT License - see LICENSE file for details.

---

**Research Focus**: Mapping the relationship between human consciousness states and AI internal representations through grounded, embodied experience.
