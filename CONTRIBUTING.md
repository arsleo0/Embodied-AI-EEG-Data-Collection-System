# Contributing to Consciousness Research Workbench

Thank you for your interest in contributing! This document provides guidelines
for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others. We aim to foster an inclusive
and welcoming community.

## Getting Started

1. **Fork the repository**

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/consciousness-research-workbench.git
   cd consciousness-research-workbench
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 88 characters (Black default)
- Use descriptive variable and function names

### Documentation

- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Update README.md for new features
- Add inline comments for complex logic

Example docstring:
```python
def process_eeg(data: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """Process raw EEG data.

    Applies filtering, artifact removal, and normalization.

    Args:
        data: Raw EEG data (channels, samples).
        fs: Sampling frequency in Hz.

    Returns:
        Processed EEG data.

    Raises:
        ValueError: If data has incorrect shape.
    """
```

### Testing

- Write tests for new functionality
- Maintain test coverage above 80%
- Use pytest for testing
- Place tests in the `tests/` directory

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Commit Messages

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes

Example:
```
feat: Add temporal binding window tracker

Implements specious present estimation and context retention
metrics for consciousness research.
```

## Pull Request Process

1. **Update documentation** for any new features

2. **Add tests** for new functionality

3. **Run the test suite** and ensure all tests pass

4. **Update CHANGELOG.md** with your changes

5. **Submit the PR** with a clear description:
   - What changes were made
   - Why the changes are needed
   - How to test the changes

6. **Address review feedback** promptly

## Plugin Development

### Creating a Plugin

1. Choose the appropriate plugin type:
   - `DataCollectorPlugin` - New data sources
   - `AnalysisPlugin` - Signal processing
   - `ConsciousnessPlugin` - AI analysis
   - `VisualizationPlugin` - Custom visualizations
   - `ExternalServicePlugin` - Third-party integrations

2. Implement required methods:
   ```python
   from src.plugins import AnalysisPlugin, PluginMetadata

   class MyPlugin(AnalysisPlugin):
       @property
       def metadata(self):
           return PluginMetadata(
               name="MyPlugin",
               version="1.0.0",
               author="Your Name",
               description="Plugin description",
               plugin_type="analysis"
           )

       def initialize(self, config):
           # Setup code
           pass

       def cleanup(self):
           # Cleanup code
           pass

       def process(self, data):
           # Analysis code
           return results
   ```

3. Register with the plugin system:
   ```python
   from src.plugins import get_registry

   registry = get_registry()
   registry.register(MyPlugin)
   ```

### Using Hooks

Integrate with the data pipeline using hooks:

```python
from src.plugins import hook, HookType

@hook(HookType.PRE_PROCESS, priority=50)
def my_preprocessor(data):
    # Modify data before processing
    return modified_data
```

## Architecture

### Project Structure

```
src/
├── foundation/      # TIER 0: Data collection
├── analysis/        # TIER 1: Signal processing
├── consciousness/   # TIER 2-3: AI modules
├── integrations/    # TIER 4-5: UI & API
├── infrastructure/  # Experiment tracking
└── plugins/         # Plugin system
```

### Design Principles

- **Modularity**: Each component should be independent
- **Extensibility**: Use plugins for new functionality
- **Reproducibility**: Track experiments and data provenance
- **Documentation**: Code should be self-documenting

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered

## Questions?

- Check existing issues and discussions
- Open a new discussion for questions
- Contact maintainers for private matters

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.

---

Thank you for contributing to consciousness research!
