"""Muse EEG device implementation using BrainFlow."""

from typing import Any

import numpy as np

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False

from .base import BaseEEGDevice


class MuseDevice(BaseEEGDevice):
    """Muse 2/S EEG device implementation using BrainFlow."""

    # Muse channel names
    MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

    # BrainFlow board IDs for Muse devices
    BOARD_IDS = {
        "muse_2": 22,        # Muse 2 (native Bluetooth)
        "muse_s": 21,        # Muse S (native Bluetooth)
        "muse_2_bled": 38,   # Muse 2 with BLED112 dongle
        "muse_s_bled": 39,   # Muse S with BLED112 dongle
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize Muse device.

        Args:
            config: Device configuration with keys:
                - device_type: "muse_2", "muse_s", etc.
                - serial_number: Device serial (optional)
                - timeout: Connection timeout in seconds
        """
        super().__init__(config)

        if not BRAINFLOW_AVAILABLE:
            raise ImportError(
                "BrainFlow is required for Muse devices. "
                "Install with: pip install brainflow"
            )

        self._board: BoardShim | None = None
        self._board_id = self._get_board_id()
        self._params = self._create_params()

    def _get_board_id(self) -> int:
        """Get BrainFlow board ID for device type.

        Returns:
            Board ID integer.
        """
        device_type = self.config.get("device_type", "muse_2")
        return self.BOARD_IDS.get(device_type, 22)

    def _create_params(self) -> "BrainFlowInputParams":
        """Create BrainFlow input parameters.

        Returns:
            Configured BrainFlowInputParams.
        """
        params = BrainFlowInputParams()

        # Set serial number if provided
        serial = self.config.get("serial_number")
        if serial:
            params.serial_number = serial

        # Set timeout
        timeout = self.config.get("timeout", 15)
        params.timeout = timeout

        return params

    def connect(self) -> bool:
        """Connect to Muse device.

        Returns:
            True if connection successful.

        Raises:
            RuntimeError: If connection fails.
        """
        if self._is_connected:
            return True

        try:
            # Enable BrainFlow logging for debugging
            BoardShim.enable_dev_board_logger()

            # Create board instance
            self._board = BoardShim(self._board_id, self._params)

            # Prepare session (connect to device)
            self._board.prepare_session()

            self._is_connected = True
            return True

        except Exception as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect to Muse: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Muse device."""
        if self._board and self._is_connected:
            try:
                if self._is_streaming:
                    self.stop_stream()
                self._board.release_session()
            except Exception:
                pass  # Ignore errors during disconnect
            finally:
                self._is_connected = False
                self._board = None

    def start_stream(self) -> None:
        """Start data streaming from Muse.

        Raises:
            RuntimeError: If not connected or stream fails to start.
        """
        if not self._is_connected:
            raise RuntimeError("Device not connected")

        if self._is_streaming:
            return

        try:
            self._board.start_stream()
            self._is_streaming = True
        except Exception as e:
            raise RuntimeError(f"Failed to start stream: {e}") from e

    def stop_stream(self) -> None:
        """Stop data streaming from Muse."""
        if self._board and self._is_streaming:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            finally:
                self._is_streaming = False

    def get_data(self, num_samples: int | None = None) -> np.ndarray:
        """Get EEG data from device buffer.

        Args:
            num_samples: Number of samples to retrieve. None for all available.

        Returns:
            EEG data array of shape (channels, samples).

        Raises:
            RuntimeError: If not streaming.
        """
        if not self._is_streaming:
            raise RuntimeError("Device not streaming")

        if num_samples is None:
            data = self._board.get_board_data()
        else:
            data = self._board.get_current_board_data(num_samples)

        # Extract only EEG channels
        eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        eeg_data = data[eeg_channels, :]

        return eeg_data

    def get_sampling_rate(self) -> int:
        """Get Muse sampling rate.

        Returns:
            Sampling rate in Hz (256 for Muse).
        """
        return BoardShim.get_sampling_rate(self._board_id)

    def get_channel_names(self) -> list[str]:
        """Get Muse EEG channel names.

        Returns:
            List of channel names.
        """
        return self.MUSE_CHANNELS.copy()

    def get_device_info(self) -> dict[str, Any]:
        """Get Muse device information.

        Returns:
            Dictionary with device info.
        """
        return {
            "device_type": self.config.get("device_type", "muse_2"),
            "board_id": self._board_id,
            "sampling_rate": self.get_sampling_rate(),
            "channels": self.get_channel_names(),
            "num_channels": len(self.get_channel_names()),
            "is_connected": self._is_connected,
            "is_streaming": self._is_streaming,
        }

    def get_signal_quality(self) -> dict[str, float]:
        """Get signal quality for each channel.

        Returns:
            Dictionary mapping channel names to quality scores (0-1).
        """
        if not self._is_streaming:
            return {ch: 0.0 for ch in self.MUSE_CHANNELS}

        # Get recent data for quality assessment
        data = self.get_data(num_samples=256)  # 1 second of data

        quality = {}
        for i, channel in enumerate(self.MUSE_CHANNELS):
            if i < data.shape[0]:
                # Simple quality metric: inverse of standard deviation
                # High variance often indicates poor contact
                std = np.std(data[i, :])
                # Normalize: good signal typically has std between 10-100 µV
                if std < 1:
                    quality[channel] = 0.0  # No signal
                elif std > 500:
                    quality[channel] = 0.0  # Too noisy (poor contact)
                else:
                    # Map to 0-1 range
                    quality[channel] = min(1.0, max(0.0, 1.0 - (std - 50) / 450))
            else:
                quality[channel] = 0.0

        return quality


class SyntheticDevice(BaseEEGDevice):
    """Synthetic EEG device for testing without hardware."""

    def __init__(self, config: dict[str, Any]):
        """Initialize synthetic device.

        Args:
            config: Device configuration.
        """
        super().__init__(config)

        if not BRAINFLOW_AVAILABLE:
            raise ImportError("BrainFlow is required. Install with: pip install brainflow")

        self._board: BoardShim | None = None
        self._board_id = -1  # Synthetic board
        self._params = BrainFlowInputParams()

    def connect(self) -> bool:
        """Connect to synthetic board."""
        if self._is_connected:
            return True

        try:
            BoardShim.enable_dev_board_logger()
            self._board = BoardShim(self._board_id, self._params)
            self._board.prepare_session()
            self._is_connected = True
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to create synthetic board: {e}") from e

    def disconnect(self) -> None:
        """Disconnect synthetic board."""
        if self._board and self._is_connected:
            try:
                if self._is_streaming:
                    self.stop_stream()
                self._board.release_session()
            except Exception:
                pass
            finally:
                self._is_connected = False

    def start_stream(self) -> None:
        """Start synthetic data stream."""
        if not self._is_connected:
            raise RuntimeError("Device not connected")
        if not self._is_streaming:
            self._board.start_stream()
            self._is_streaming = True

    def stop_stream(self) -> None:
        """Stop synthetic data stream."""
        if self._board and self._is_streaming:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            finally:
                self._is_streaming = False

    def get_data(self, num_samples: int | None = None) -> np.ndarray:
        """Get synthetic EEG data."""
        if not self._is_streaming:
            raise RuntimeError("Device not streaming")

        if num_samples is None:
            data = self._board.get_board_data()
        else:
            data = self._board.get_current_board_data(num_samples)

        eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        return data[eeg_channels, :]

    def get_sampling_rate(self) -> int:
        """Get synthetic sampling rate."""
        return BoardShim.get_sampling_rate(self._board_id)

    def get_channel_names(self) -> list[str]:
        """Get synthetic channel names."""
        num_channels = len(BoardShim.get_eeg_channels(self._board_id))
        return [f"CH{i+1}" for i in range(num_channels)]

    def get_device_info(self) -> dict[str, Any]:
        """Get synthetic device info."""
        return {
            "device_type": "synthetic",
            "board_id": self._board_id,
            "sampling_rate": self.get_sampling_rate(),
            "channels": self.get_channel_names(),
            "is_connected": self._is_connected,
            "is_streaming": self._is_streaming,
        }
