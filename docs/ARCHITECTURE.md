# Consciousness Research Workbench - Architecture

## Overview

This document describes the technical architecture of the Consciousness Research Workbench, a platform for exploring AI consciousness through grounded human experience.

## Design Philosophy

### Core Principles

1. **Privacy-First**: All data stays local. No telemetry, no cloud dependencies.
2. **Modular Architecture**: Each component is independently testable and replaceable.
3. **Progressive Disclosure**: Simple CLI for basics, full API for advanced use.
4. **Research-Oriented**: Designed for experimentation, not production deployment.

### Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| BrainFlow over muselsl | Better maintained, cross-platform, multi-device support |
| HDF5 for EEG | Efficient time-series storage, chunking, compression |
| Parquet for metadata | Columnar storage, fast queries, Pandas integration |
| Plugin system | Extensibility without core modification |
| YAML config | Human-readable, hierarchical configuration |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CONSCIOUSNESS RESEARCH WORKBENCH                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    PLUGIN SYSTEM                              │  │
│  │  Registry • Lifecycle Management • Configuration              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                   INTEGRATIONS LAYER                          │  │
│  │  Dashboard (Streamlit) • Reports • API (REST/GraphQL)         │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │               CONSCIOUSNESS MODULES                           │  │
│  │  Latent Space • Attention • Meta-Awareness • Qualia           │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                  ANALYSIS CORE                                │  │
│  │  Signal Processing • Features • Classification                │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                   FOUNDATION LAYER                            │  │
│  │  EEG Devices • Sensors • Storage • Scenarios • Config         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### Foundation Layer (`src/foundation/`)

The base layer providing data collection and storage infrastructure.

#### EEG Module (`foundation/eeg/`)

```python
# Device abstraction hierarchy
BaseEEGDevice (ABC)
├── MuseDevice          # Muse 2/S via BrainFlow
├── SyntheticDevice     # Testing without hardware
└── [Future devices]    # OpenBCI, Emotiv, etc.

# Key classes
EEGConnection          # Connection testing and management
EEGStream              # Real-time data streaming
```

**Data Flow:**
```
Muse Device → BrainFlow → EEGStream → DataLogger → HDF5
```

#### Storage Module (`foundation/data/`)

```python
# Storage backends
HDF5Storage           # Time-series EEG data
├── Chunked writes    # 1-second chunks
├── GZIP compression  # Level 4
└── Resizable datasets

ParquetStorage        # Metadata and logs
├── Session metadata
├── Markers
└── Queryable structure
```

**HDF5 Structure:**
```
session_20240101_120000_abc123.h5
├── eeg                    # (channels, samples) float32
├── timestamps             # (samples,) float64
├── markers/
│   ├── 0/                 # {timestamp, name, value, notes}
│   └── ...
├── metadata/
│   ├── session_id
│   ├── sampling_rate
│   └── channels
└── sensors/
    └── gps                # (points, 5) [ts, lat, lon, alt, acc]
```

#### Scenario Module (`foundation/scenarios/`)

```python
# Scenario execution
ScenarioRunner
├── Load scenario config (YAML)
├── Initialize data logger
├── Handle keyboard markers
├── Collect data chunks
└── Generate session report
```

**Marker System:**
- Default markers: `n` (notable), `p` (positive), `g` (negative), `f` (flow)
- Scenario-specific markers defined in YAML
- Real-time HDF5 persistence

---

### Analysis Layer (`src/analysis/`)

Signal processing and feature extraction.

#### Signal Processing (`analysis/signal/`)

**Planned Components:**
```python
# Preprocessing pipeline
Preprocessor
├── NotchFilter(60Hz)      # Power line noise
├── BandpassFilter(1-50Hz) # EEG range
├── ArtifactDetector       # Eye blinks, muscle
└── Resampler              # Standardize rate
```

#### Feature Extraction (`analysis/features/`)

**Planned Features:**

| Feature | Description | Formula |
|---------|-------------|---------|
| PSD | Power spectral density | Welch method |
| Band Power | Delta, Theta, Alpha, Beta, Gamma | Mean PSD in band |
| Coherence | Channel correlation | Cross-spectral density |
| Entropy | Signal complexity | Sample entropy, spectral entropy |
| Hjorth | Activity, mobility, complexity | Time-domain parameters |

#### Classification (`analysis/classification/`)

**Planned Models:**
```python
StateClassifier
├── RandomForestClassifier   # Baseline
├── XGBoostClassifier        # Feature importance
└── NeuralClassifier         # Deep learning
```

**Target States:**
- Flow/Creativity
- Focus/Concentration
- Meditation/Calm
- Anxiety/Stress
- Neutral/Baseline

---

### Consciousness Layer (`src/consciousness/`)

AI consciousness analysis and human-AI state comparison.

#### Latent Space Mapper (`consciousness/latent_space/`)

**Architecture:**
```python
LatentSpaceMapper
├── EEGEncoder           # EEG → embedding
│   ├── Temporal conv
│   └── Attention pooling
├── ProjectionHead       # Embedding → 2D/3D
│   ├── t-SNE
│   └── UMAP
└── TrajectoryTracker    # State evolution
```

**Research Questions:**
- Can we map EEG states to semantic embeddings?
- Do consciousness trajectories cluster meaningfully?
- How do different scenarios affect latent space geometry?

#### Attention Analyzer (`consciousness/attention/`)

**Planned Features:**
```python
AttentionAnalyzer
├── HookManager          # API attention extraction
├── EntropyCalculator    # Attention distribution
├── GlobalWorkspaceDetector
└── HumanAIComparator
```

**Metrics:**
- Attention entropy
- Global workspace activation
- Cross-modal attention patterns

#### Meta-Awareness Detector (`consciousness/meta_awareness/`)

**Research Focus:**
- Recursive self-reference in LLM outputs
- Uncertainty quantification
- "Know what you don't know" metrics
- Chain-of-thought introspection patterns

#### Qualia Synthesizer (`consciousness/qualia/`)

**Planned Architecture:**
```python
QualiaSynthesizer
├── CrossModalEncoder    # CLIP/ImageBind
├── SynesthesiaMapper    # EEG → sensory
└── EmotionalRenderer    # State → imagery
```

#### Emergence Metrics (`consciousness/emergence/`)

**Planned Metrics:**
- Integrated Information (Phi)
- Complexity measures
- Phase transition detection
- Causal density

---

### Integrations Layer (`src/integrations/`)

User interfaces and external connections.

#### Dashboard (`integrations/dashboard/`)

**Planned Features (Streamlit):**
- Real-time EEG signal viewer
- Scenario control panel
- Session browser
- Feature visualization
- Latent space explorer

#### Reports (`integrations/reports/`)

**Report Types:**
- Session summary
- Feature analysis
- State classification results
- Cross-session comparison

#### API (`integrations/api/`)

**Planned Endpoints:**
```
REST API
├── /sessions           # Session CRUD
├── /scenarios          # Scenario management
├── /analysis           # Run analysis
└── /visualizations     # Generate plots

WebSocket
├── /stream/eeg         # Real-time EEG
└── /stream/markers     # Real-time markers
```

---

## Plugin System

### Architecture

```python
# Plugin base classes
BasePlugin
├── DataCollectorPlugin    # New data sources
├── AnalysisPlugin         # Processing modules
├── ConsciousnessPlugin    # AI analysis
├── VisualizationPlugin    # Custom viz
└── ExternalServicePlugin  # Third-party

# Plugin registry
PluginRegistry
├── register(plugin_class)
├── get_plugin(name, type)
├── list_plugins()
└── cleanup_all()
```

### Plugin Lifecycle

```
1. Register    → Add to registry
2. Initialize  → Load config, acquire resources
3. Execute     → Process data, generate results
4. Cleanup     → Release resources
```

### Creating a Plugin

```python
from src.plugins import AnalysisPlugin, PluginMetadata

class MyPlugin(AnalysisPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="MyPlugin",
            version="1.0.0",
            author="Author",
            description="Description",
            plugin_type="analysis",
            dependencies=["numpy", "scipy"],
            config_schema={
                "param1": {"type": "int", "default": 10}
            }
        )

    def initialize(self, config):
        self.param1 = config.get("param1", 10)

    def cleanup(self):
        pass

    def process(self, data):
        # Analysis logic
        return {"results": processed_data}
```

---

## Data Flow

### Recording Session

```
User → CLI → ScenarioRunner
                   │
                   ├─→ EEGStream.start()
                   │        │
                   │        ▼
                   │   BrainFlow.get_data()
                   │        │
                   ├─→ DataLogger.collect_chunk()
                   │        │
                   │        ▼
                   │   HDF5Storage.append()
                   │
                   ├─→ Keyboard Input
                   │        │
                   │        ▼
                   │   DataLogger.add_marker()
                   │
                   └─→ Session Complete
                            │
                            ▼
                    ParquetStorage.save_metadata()
```

### Analysis Pipeline

```
HDF5 Session → Preprocessor → Feature Extractor → Classifier
                    │                  │               │
                    ▼                  ▼               ▼
              Filtered EEG      Feature Vector    State Labels
                    │                  │               │
                    └──────────────────┴───────────────┘
                                       │
                                       ▼
                              Analysis Results
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                    Dashboard      Reports      API/Export
```

---

## Configuration System

### Hierarchy

```
config.yaml (project root)
    │
    ├── Device configs
    │   └── eeg.device_type, eeg.brainflow.*
    │
    ├── Storage configs
    │   └── storage.raw.*, storage.processed.*
    │
    ├── Sensor configs
    │   └── sensors.gps.*, sensors.audio.*
    │
    └── Scenario configs
        └── scenarios.defaults.*, scenarios.types
```

### Access Pattern

```python
from src.foundation.config import load_config

config = load_config()

# Dot-notation access
device = config.get("eeg.device_type")
rate = config.get("eeg.brainflow.sampling_rate")

# Section access
eeg_config = config.eeg
storage_config = config.storage

# Computed properties
board_id = config.get_board_id()
raw_path = config.get_storage_path("raw")
```

---

## Security Considerations

### Data Privacy

- All data stored locally (no cloud sync)
- Sensitive paths (data/) gitignored
- Optional device serial anonymization
- No telemetry or analytics

### Input Validation

- Config schema validation
- File path sanitization
- Plugin sandboxing (future)

### API Security (Future)

- Local-only by default
- Optional authentication
- Rate limiting

---

## Testing Strategy

### Unit Tests

```
tests/
├── test_connection.py      # EEG connection
├── test_storage.py         # HDF5/Parquet
├── test_scenarios.py       # Scenario runner
├── test_plugins.py         # Plugin system
└── test_analysis.py        # Analysis modules
```

### Integration Tests

- Full session recording with synthetic device
- End-to-end analysis pipeline
- Plugin loading and execution

### Test Data

- Synthetic EEG via BrainFlow
- Sample HDF5 sessions
- Mock scenarios

---

## Performance Considerations

### Memory Management

- Streaming data collection (no full session in memory)
- Chunked HDF5 writes
- Lazy loading for analysis

### Concurrency

- Single-threaded data collection (real-time constraint)
- Multiprocessing for batch analysis
- Async API endpoints (future)

### Scalability

- Sessions stored independently
- Horizontal scaling via multiple experiments
- No database bottleneck

---

## Future Architecture

### Phase 2 Additions

- Real-time signal quality feedback
- WebSocket GPS integration
- Whisper transcription pipeline

### Phase 3 Additions

- Neural network feature extractors
- LLM API hooks for attention analysis
- Multi-session analysis

### Phase 4 Additions

- Web-based dashboard
- Experiment versioning (DVC)
- Collaborative features

---

## References

### EEG Processing

- [MNE-Python](https://mne.tools/)
- [BrainFlow](https://brainflow.ai/)
- [PyEEG](https://github.com/forrestbao/pyeeg)

### Consciousness Theory

- Integrated Information Theory (Tononi)
- Global Workspace Theory (Baars)
- Higher-Order Theories (Rosenthal)

### AI Interpretability

- Attention visualization
- Probing classifiers
- Activation analysis
