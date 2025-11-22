"""Data models and schemas for EEG data collection."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Marker:
    """Event marker during recording."""

    timestamp: float  # Unix timestamp
    name: str  # Marker name/type
    value: Any = None  # Optional value
    notes: str = ""  # Optional notes


@dataclass
class GPSPoint:
    """GPS coordinate point."""

    timestamp: float
    latitude: float
    longitude: float
    altitude: float | None = None
    accuracy: float | None = None  # Meters


@dataclass
class AudioSegment:
    """Reference to recorded audio segment."""

    timestamp: float
    duration: float  # Seconds
    file_path: str
    transcript: str | None = None


@dataclass
class SessionMetadata:
    """Metadata for a recording session."""

    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    scenario_name: str | None = None
    device_type: str = ""
    sampling_rate: int = 0
    channels: list[str] = field(default_factory=list)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "scenario_name": self.scenario_name,
            "device_type": self.device_type,
            "sampling_rate": self.sampling_rate,
            "channels": self.channels,
            "notes": self.notes,
            "tags": self.tags,
        }


@dataclass
class SessionData:
    """Complete session data container."""

    metadata: SessionMetadata
    markers: list[Marker] = field(default_factory=list)
    gps_points: list[GPSPoint] = field(default_factory=list)
    audio_segments: list[AudioSegment] = field(default_factory=list)
