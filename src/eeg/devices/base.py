"""Abstract base class for EEG devices."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEEGDevice(ABC):
    """Abstract base class for EEG device implementations."""

    def __init__(self, config: dict[str, Any]):
        """Initialize device with configuration.

        Args:
            config: Device configuration dictionary.
        """
        self.config = config
        self._is_connected = False
        self._is_streaming = False

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        """Check if device is streaming data."""
        return self._is_streaming

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the EEG device.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the EEG device."""
        pass

    @abstractmethod
    def start_stream(self) -> None:
        """Start data streaming from device."""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop data streaming from device."""
        pass

    @abstractmethod
    def get_data(self, num_samples: int | None = None) -> np.ndarray:
        """Get EEG data from device buffer.

        Args:
            num_samples: Number of samples to retrieve. None for all available.

        Returns:
            EEG data array of shape (channels, samples).
        """
        pass

    @abstractmethod
    def get_sampling_rate(self) -> int:
        """Get device sampling rate in Hz.

        Returns:
            Sampling rate.
        """
        pass

    @abstractmethod
    def get_channel_names(self) -> list[str]:
        """Get EEG channel names.

        Returns:
            List of channel names.
        """
        pass

    @abstractmethod
    def get_device_info(self) -> dict[str, Any]:
        """Get device information.

        Returns:
            Dictionary with device info.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_streaming:
            self.stop_stream()
        self.disconnect()
        return False
