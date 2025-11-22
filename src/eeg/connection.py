"""EEG device connection testing and management."""

import time
from typing import Any

from ..config.settings import Config, load_config
from .devices.base import BaseEEGDevice
from .devices.muse import MuseDevice, SyntheticDevice


class EEGConnection:
    """Manages EEG device connections and provides connection testing."""

    def __init__(self, config: Config | None = None):
        """Initialize connection manager.

        Args:
            config: Configuration instance. Loads default if None.
        """
        self.config = config or load_config()
        self._device: BaseEEGDevice | None = None

    def get_device(self) -> BaseEEGDevice:
        """Get or create EEG device instance.

        Returns:
            EEG device instance based on configuration.

        Raises:
            ValueError: If device type is not supported.
        """
        if self._device is not None:
            return self._device

        device_type = self.config.get("eeg.device_type", "synthetic")

        device_config = {
            "device_type": device_type,
            "serial_number": self.config.get("eeg.muse.serial_number"),
            "timeout": self.config.get("eeg.brainflow.timeout", 15),
        }

        if device_type in ("muse_2", "muse_s", "muse_2_bled", "muse_s_bled"):
            self._device = MuseDevice(device_config)
        elif device_type == "synthetic":
            self._device = SyntheticDevice(device_config)
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

        return self._device

    def test_connection(self, duration: float = 5.0) -> dict[str, Any]:
        """Test connection to EEG device.

        Args:
            duration: Duration to stream data for testing (seconds).

        Returns:
            Test results dictionary with:
                - success: Whether test passed
                - device_info: Device information
                - samples_received: Number of samples received
                - actual_rate: Calculated sampling rate
                - errors: List of any errors encountered
        """
        results = {
            "success": False,
            "device_info": {},
            "samples_received": 0,
            "actual_rate": 0.0,
            "errors": [],
        }

        device = None
        try:
            # Get device instance
            device = self.get_device()

            # Test connection
            print(f"Connecting to {device.config.get('device_type', 'unknown')}...")
            connected = device.connect()

            if not connected:
                results["errors"].append("Connection failed")
                return results

            results["device_info"] = device.get_device_info()
            print(f"Connected! Device info: {results['device_info']}")

            # Test streaming
            print(f"Starting stream for {duration} seconds...")
            device.start_stream()

            # Wait and collect data
            time.sleep(duration)

            # Get data
            data = device.get_data()
            results["samples_received"] = data.shape[1] if len(data.shape) > 1 else 0

            # Calculate actual sampling rate
            expected_samples = duration * device.get_sampling_rate()
            results["actual_rate"] = results["samples_received"] / duration

            # Check if we got reasonable amount of data
            rate_tolerance = 0.1  # 10% tolerance
            expected_rate = device.get_sampling_rate()
            rate_diff = abs(results["actual_rate"] - expected_rate) / expected_rate

            if rate_diff > rate_tolerance:
                results["errors"].append(
                    f"Sampling rate deviation: expected {expected_rate} Hz, "
                    f"got {results['actual_rate']:.1f} Hz"
                )

            # Stop streaming
            device.stop_stream()
            print(f"Received {results['samples_received']} samples")

            # Test passed if we got data
            results["success"] = results["samples_received"] > 0

        except Exception as e:
            results["errors"].append(str(e))

        finally:
            # Clean up
            if device and device.is_connected:
                device.disconnect()
            self._device = None

        return results

    def quick_check(self) -> bool:
        """Quick connection check without streaming.

        Returns:
            True if device can connect.
        """
        device = None
        try:
            device = self.get_device()
            connected = device.connect()
            return connected
        except Exception:
            return False
        finally:
            if device and device.is_connected:
                device.disconnect()
            self._device = None


def test_connection(
    device_type: str | None = None,
    duration: float = 5.0,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Convenience function to test EEG connection.

    Args:
        device_type: Override device type from config.
        duration: Test duration in seconds.
        config_path: Path to config file.

    Returns:
        Test results dictionary.
    """
    config = load_config(config_path)

    if device_type:
        config.set("eeg.device_type", device_type)

    connection = EEGConnection(config)
    return connection.test_connection(duration)


def list_available_devices() -> list[str]:
    """List available/supported EEG device types.

    Returns:
        List of device type names.
    """
    return [
        "muse_2",
        "muse_s",
        "muse_2_bled",
        "muse_s_bled",
        "synthetic",
    ]
