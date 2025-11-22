# Embodied AI EEG Data Collection System

A privacy-first, modular system for collecting multimodal EEG data during real-world scenarios. Designed for mapping consciousness and emotional states through grounded, embodied experiences.

## Overview

This system enables:
- **Real-time EEG data streaming** from consumer-grade devices (Muse 2/S)
- **Multimodal data collection**: EEG + GPS + audio notes + timestamps
- **Scenario-based experiments**: Urban walks, creative flow, social interactions
- **Local-only storage**: All data stays on your machine
- **Extensible architecture**: Easy to add new devices, sensors, and analysis pipelines

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Embodied AI EEG System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   EEG       │   │   GPS       │   │   Audio     │           │
│  │   Device    │   │   Phone     │   │   Mic       │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Sensor Abstraction Layer               │       │
│  │   (BrainFlow)    (WebSocket)    (SoundDevice)       │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Scenario Execution Engine              │       │
│  │   • Event markers    • Timestamps    • State mgmt   │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │                  Data Logger                        │       │
│  │   • HDF5 (EEG)   • Parquet (meta)   • WAV (audio)  │       │
│  └─────────────────────────┬───────────────────────────┘       │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Local Storage (Privacy-First)          │       │
│  │   data/raw/  data/processed/  data/exports/         │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
embodied-ai-eeg/
├── src/
│   ├── eeg/                    # EEG device handling
│   │   ├── connection.py       # Connection testing & management
│   │   ├── stream.py           # Real-time data streaming
│   │   └── devices/            # Device-specific implementations
│   │       ├── base.py         # Abstract base class
│   │       └── muse.py         # Muse 2/S implementation
│   │
│   ├── data/                   # Data management
│   │   ├── logger.py           # Main data logger
│   │   ├── storage.py          # HDF5/Parquet handlers
│   │   └── models.py           # Data schemas
│   │
│   ├── scenarios/              # Scenario execution
│   │   ├── runner.py           # CLI scenario runner
│   │   ├── base.py             # Base scenario class
│   │   └── examples/           # Built-in scenarios
│   │
│   ├── sensors/                # Additional sensors
│   │   ├── gps.py              # GPS integration
│   │   └── audio.py            # Audio recording
│   │
│   ├── config/                 # Configuration
│   │   └── settings.py         # Config loader
│   │
│   └── utils/                  # Utilities
│       └── time_sync.py        # Timestamp synchronization
│
├── data/                       # Local data storage (gitignored)
│   ├── raw/                    # Raw recordings
│   ├── processed/              # Analyzed data
│   └── exports/                # Export files
│
├── scenarios/                  # Scenario definitions (YAML)
├── scripts/                    # CLI entry points
├── tests/                      # Unit tests
│
├── config.yaml                 # Main configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10+
- Windows 10/11 (primary), macOS, or Linux
- Muse 2 or Muse S headband
- Bluetooth LE support

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd embodied-ai-eeg

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### BrainFlow Driver Setup (Windows)

For Muse devices on Windows, you may need the BLED112 dongle or native Bluetooth:

```bash
# Check BrainFlow installation
python -c "from brainflow.board_shim import BoardShim; print('BrainFlow OK')"
```

## Quick Start

### 1. Test EEG Connection

```bash
# Test connection to your Muse device
python scripts/test_eeg_connection.py

# With specific device
python scripts/test_eeg_connection.py --device muse_2
```

### 2. Run a Scenario

```bash
# Start a scenario session
python scripts/run_scenario.py urban_walk

# List available scenarios
python scripts/run_scenario.py --list

# Custom duration
python scripts/run_scenario.py meditation --duration 300
```

### 3. View Collected Data

Data is stored in `data/raw/` as HDF5 files:

```python
import h5py

with h5py.File('data/raw/session_20240101_120000.h5', 'r') as f:
    eeg_data = f['eeg'][:]
    timestamps = f['timestamps'][:]
    markers = f['markers'][:]
```

## Configuration

Edit `config.yaml` to customize:

- **EEG device settings**: Device type, channels, sampling rate
- **Storage paths**: Where data is saved
- **Sensor settings**: GPS, audio configuration
- **Processing parameters**: Filters, frequency bands

### Key Configuration Options

```yaml
eeg:
  device_type: "muse_2"           # muse_2, muse_s, synthetic
  brainflow:
    board_id: 22                  # BrainFlow board ID
    sampling_rate: 256            # Hz

storage:
  raw:
    format: "hdf5"
    compression: "gzip"

sensors:
  gps:
    enabled: true
    source: "phone"
  audio:
    enabled: true
    sample_rate: 16000
```

## Scenarios

Scenarios define structured data collection sessions:

| Scenario | Description | Focus |
|----------|-------------|-------|
| `urban_walk` | City navigation with varying stimuli | Environmental response |
| `creative_flow` | Creative work (e.g., Blender) | Flow state detection |
| `social_interaction` | Social contexts (café, market) | Emotional responses |
| `problem_solving` | Coding, design decisions | Cognitive load |
| `meditation` | Calm baseline comparison | Reference state |

### Custom Scenarios

Create new scenarios in `scenarios/` as YAML files:

```yaml
# scenarios/my_scenario.yaml
name: "My Custom Scenario"
description: "Description of the scenario"
duration: 600  # 10 minutes, or null for manual stop
markers:
  - name: "start_task"
    key: "s"
  - name: "end_task"
    key: "e"
  - name: "notable_event"
    key: "n"
```

## Data Format

### HDF5 Structure (EEG Data)

```
session_YYYYMMDD_HHMMSS.h5
├── eeg                    # Shape: (samples, channels)
├── timestamps             # Unix timestamps for each sample
├── markers                # Event markers with timestamps
├── metadata
│   ├── device_type
│   ├── sampling_rate
│   ├── channels
│   └── scenario
└── sensors
    ├── gps                # GPS coordinates
    └── audio_segments     # References to audio files
```

### Parquet Files (Metadata)

Session metadata and scenario logs are stored as Parquet for easy querying:

```python
import pandas as pd
df = pd.read_parquet('data/processed/sessions.parquet')
```

## Privacy & Security

- **Local-only**: No data leaves your machine
- **No telemetry**: Zero external connections
- **Gitignored data**: Raw data never committed
- **Optional anonymization**: Strip device serials, hash session IDs

## Development Roadmap

### Phase 1 (Current)
- [x] Project structure
- [x] Configuration system
- [ ] EEG connection tester
- [ ] Data logger skeleton
- [ ] CLI scenario runner

### Phase 2
- [ ] Real-time signal quality indicator
- [ ] GPS phone integration
- [ ] Audio recording + Whisper transcription
- [ ] Basic visualization (Plotly)

### Phase 3
- [ ] Web dashboard
- [ ] LLM integration (GPT-4o analysis)
- [ ] Emotion latent space mapping
- [ ] Cross-session analysis

## Troubleshooting

### Muse Not Connecting

1. Ensure Muse is in pairing mode (hold button until light pulses)
2. Check Bluetooth is enabled
3. Try synthetic board first: `--device synthetic`

### Import Errors

```bash
# Ensure virtual environment is activated
pip install -r requirements.txt --upgrade
```

### Permission Issues (Linux)

```bash
# Add user to dialout group for Bluetooth
sudo usermod -a -G dialout $USER
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [BrainFlow](https://brainflow.ai/) - EEG device SDK
- [Muse](https://choosemuse.com/) - Consumer EEG headband
- Research inspired by embodied cognition and grounded AI principles
