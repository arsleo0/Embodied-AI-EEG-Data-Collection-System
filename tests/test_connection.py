"""Tests for EEG connection functionality."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Config, load_config
from src.eeg.connection import EEGConnection, list_available_devices


class TestConfig:
    """Tests for configuration loading."""

    def test_load_config(self):
        """Test loading configuration from file."""
        config = load_config()
        assert config is not None
        assert config.get("eeg.device_type") is not None

    def test_get_board_id(self):
        """Test getting BrainFlow board ID."""
        config = load_config()
        board_id = config.get_board_id()
        assert isinstance(board_id, int)

    def test_get_storage_path(self):
        """Test getting storage paths."""
        config = load_config()
        raw_path = config.get_storage_path("raw")
        assert isinstance(raw_path, Path)


class TestEEGConnection:
    """Tests for EEG connection."""

    def test_list_devices(self):
        """Test listing available devices."""
        devices = list_available_devices()
        assert "synthetic" in devices
        assert "muse_2" in devices

    def test_synthetic_connection(self):
        """Test connection with synthetic board."""
        config = load_config()
        config.set("eeg.device_type", "synthetic")

        connection = EEGConnection(config)
        results = connection.test_connection(duration=2.0)

        assert results["success"] is True
        assert results["samples_received"] > 0
        assert results["device_info"]["device_type"] == "synthetic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
