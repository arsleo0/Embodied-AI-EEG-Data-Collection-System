"""Real-time EEG data streaming utilities."""

import time
from collections.abc import Generator
from typing import Any

import numpy as np

from ..config.settings import Config, load_config
from .connection import EEGConnection
from .devices.base import BaseEEGDevice


class EEGStream:
    """Real-time EEG data stream manager."""

    def __init__(self, config: Config | None = None):
        """Initialize stream manager.

        Args:
            config: Configuration instance.
        """
        self.config = config or load_config()
        self._connection = EEGConnection(self.config)
        self._device: BaseEEGDevice | None = None
        self._is_running = False

    def start(self) -> None:
        """Start the EEG stream.

        Raises:
            RuntimeError: If stream fails to start.
        """
        if self._is_running:
            return

        self._device = self._connection.get_device()
        self._device.connect()
        self._device.start_stream()
        self._is_running = True

    def stop(self) -> None:
        """Stop the EEG stream."""
        if self._device:
            if self._device.is_streaming:
                self._device.stop_stream()
            if self._device.is_connected:
                self._device.disconnect()
        self._is_running = False
        self._device = None

    def get_data(self, num_samples: int | None = None) -> np.ndarray:
        """Get data from stream buffer.

        Args:
            num_samples: Number of samples to get. None for all available.

        Returns:
            EEG data array.
        """
        if not self._is_running or not self._device:
            raise RuntimeError("Stream not running")

        return self._device.get_data(num_samples)

    def stream_chunks(
        self,
        chunk_duration: float = 1.0,
        max_duration: float | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream data in chunks.

        Args:
            chunk_duration: Duration of each chunk in seconds.
            max_duration: Maximum total duration. None for indefinite.

        Yields:
            Dictionary with:
                - data: EEG data array
                - timestamp: Chunk timestamp
                - duration: Actual chunk duration
        """
        if not self._is_running:
            self.start()

        start_time = time.time()
        chunk_samples = int(chunk_duration * self._device.get_sampling_rate())

        try:
            while True:
                # Check duration limit
                elapsed = time.time() - start_time
                if max_duration and elapsed >= max_duration:
                    break

                # Wait for chunk to fill
                time.sleep(chunk_duration)

                # Get chunk data
                data = self.get_data(chunk_samples)

                yield {
                    "data": data,
                    "timestamp": time.time(),
                    "duration": chunk_duration,
                    "samples": data.shape[1] if len(data.shape) > 1 else 0,
                }

        except KeyboardInterrupt:
            pass

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._is_running

    @property
    def sampling_rate(self) -> int:
        """Get stream sampling rate."""
        if self._device:
            return self._device.get_sampling_rate()
        return 0

    @property
    def channel_names(self) -> list[str]:
        """Get channel names."""
        if self._device:
            return self._device.get_channel_names()
        return []

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
