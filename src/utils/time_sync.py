"""Timestamp synchronization utilities."""

import time
from datetime import datetime, timezone
from typing import Any

import pytz


class TimestampManager:
    """Manages timestamps and synchronization across sensors."""

    def __init__(self, timezone_name: str = "local"):
        """Initialize timestamp manager.

        Args:
            timezone_name: Timezone name or "local" for system timezone.
        """
        if timezone_name == "local":
            self.tz = datetime.now().astimezone().tzinfo
        else:
            self.tz = pytz.timezone(timezone_name)

        self._offset = 0.0  # Offset from system time (for NTP sync)

    def now(self) -> float:
        """Get current timestamp with any offset applied.

        Returns:
            Unix timestamp.
        """
        return time.time() + self._offset

    def now_iso(self) -> str:
        """Get current timestamp in ISO 8601 format.

        Returns:
            ISO 8601 formatted string.
        """
        dt = datetime.fromtimestamp(self.now(), tz=self.tz)
        return dt.isoformat()

    def now_datetime(self) -> datetime:
        """Get current datetime with timezone.

        Returns:
            Timezone-aware datetime.
        """
        return datetime.fromtimestamp(self.now(), tz=self.tz)

    def to_iso(self, timestamp: float) -> str:
        """Convert Unix timestamp to ISO 8601.

        Args:
            timestamp: Unix timestamp.

        Returns:
            ISO 8601 formatted string.
        """
        dt = datetime.fromtimestamp(timestamp, tz=self.tz)
        return dt.isoformat()

    def from_iso(self, iso_string: str) -> float:
        """Convert ISO 8601 string to Unix timestamp.

        Args:
            iso_string: ISO 8601 formatted string.

        Returns:
            Unix timestamp.
        """
        dt = datetime.fromisoformat(iso_string)
        return dt.timestamp()

    def set_offset(self, offset: float) -> None:
        """Set time offset (e.g., from NTP sync).

        Args:
            offset: Offset in seconds to add to system time.
        """
        self._offset = offset

    def get_session_id(self) -> str:
        """Generate session ID from current timestamp.

        Returns:
            Session ID string.
        """
        return datetime.fromtimestamp(self.now(), tz=self.tz).strftime("%Y%m%d_%H%M%S")

    def format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like "1h 23m 45s".
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Get manager state as dictionary.

        Returns:
            State dictionary.
        """
        return {
            "timezone": str(self.tz),
            "offset": self._offset,
            "current_time": self.now_iso(),
        }
