"""GPS integration for location tracking."""

import time
from typing import Any

from ..data.models import GPSPoint


class GPSReceiver:
    """GPS receiver for location tracking.

    Phase 1: Skeleton implementation for manual coordinate logging.
    Phase 2: WebSocket receiver for phone GPS.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize GPS receiver.

        Args:
            config: GPS configuration.
        """
        self.config = config or {}
        self._is_connected = False
        self._last_point: GPSPoint | None = None

    def connect(self) -> bool:
        """Connect to GPS source.

        Returns:
            True if connected successfully.
        """
        source = self.config.get("source", "manual")

        if source == "manual":
            # Manual mode always "connects"
            self._is_connected = True
            return True

        elif source == "phone":
            # TODO: Implement WebSocket connection to phone
            # This will be implemented in Phase 2
            self._is_connected = True
            return True

        return False

    def disconnect(self) -> None:
        """Disconnect from GPS source."""
        self._is_connected = False

    def get_current_location(self) -> GPSPoint | None:
        """Get current GPS location.

        Returns:
            Current GPS point or None if not available.
        """
        if not self._is_connected:
            return None

        # In manual mode, return last set point
        return self._last_point

    def set_location(
        self,
        latitude: float,
        longitude: float,
        altitude: float | None = None,
        accuracy: float | None = None,
    ) -> GPSPoint:
        """Manually set current location.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
            altitude: Altitude in meters.
            accuracy: Accuracy in meters.

        Returns:
            Created GPS point.
        """
        self._last_point = GPSPoint(
            timestamp=time.time(),
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            accuracy=accuracy,
        )
        return self._last_point

    @property
    def is_connected(self) -> bool:
        """Check if GPS is connected."""
        return self._is_connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


# Placeholder for future WebSocket GPS implementation
class PhoneGPSReceiver(GPSReceiver):
    """WebSocket-based GPS receiver for phone location.

    To be implemented in Phase 2.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize phone GPS receiver."""
        super().__init__(config)
        self._port = self.config.get("phone", {}).get("port", 8765)

    async def start_server(self) -> None:
        """Start WebSocket server for phone connections.

        TODO: Implement in Phase 2.
        """
        pass

    async def stop_server(self) -> None:
        """Stop WebSocket server.

        TODO: Implement in Phase 2.
        """
        pass
