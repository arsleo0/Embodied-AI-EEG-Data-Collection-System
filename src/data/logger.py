"""Main data logger for EEG collection sessions."""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config.settings import Config, load_config
from ..eeg.stream import EEGStream
from .models import SessionMetadata, Marker, GPSPoint, AudioSegment
from .storage import HDF5Storage, ParquetStorage


class Session:
    """Represents an active recording session."""

    def __init__(
        self,
        session_id: str,
        scenario_name: str | None = None,
        config: Config | None = None,
    ):
        """Initialize session.

        Args:
            session_id: Unique session identifier.
            scenario_name: Name of scenario being run.
            config: Configuration instance.
        """
        self.session_id = session_id
        self.scenario_name = scenario_name
        self.config = config or load_config()

        self.start_time = datetime.now()
        self.end_time: datetime | None = None

        self.markers: list[Marker] = []
        self.gps_points: list[GPSPoint] = []
        self.audio_segments: list[AudioSegment] = []

        self._is_active = True

    def add_marker(self, name: str, value: Any = None, notes: str = "") -> Marker:
        """Add event marker to session.

        Args:
            name: Marker name.
            value: Optional value.
            notes: Optional notes.

        Returns:
            Created marker.
        """
        marker = Marker(
            timestamp=time.time(),
            name=name,
            value=value,
            notes=notes,
        )
        self.markers.append(marker)
        return marker

    def add_gps(
        self,
        latitude: float,
        longitude: float,
        altitude: float | None = None,
        accuracy: float | None = None,
    ) -> GPSPoint:
        """Add GPS point to session.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
            altitude: Altitude in meters.
            accuracy: Accuracy in meters.

        Returns:
            Created GPS point.
        """
        point = GPSPoint(
            timestamp=time.time(),
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            accuracy=accuracy,
        )
        self.gps_points.append(point)
        return point

    def end(self) -> None:
        """End the session."""
        self.end_time = datetime.now()
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._is_active

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def get_metadata(self) -> SessionMetadata:
        """Get session metadata.

        Returns:
            SessionMetadata instance.
        """
        return SessionMetadata(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=self.end_time,
            scenario_name=self.scenario_name,
        )


class DataLogger:
    """Main data logger for EEG collection."""

    def __init__(self, config: Config | None = None):
        """Initialize data logger.

        Args:
            config: Configuration instance.
        """
        self.config = config or load_config()

        # Initialize storage backends
        raw_path = self.config.get_storage_path("raw")
        processed_path = self.config.get_storage_path("processed")

        self.hdf5_storage = HDF5Storage(
            raw_path,
            compression=self.config.get("storage.raw.compression", "gzip"),
            compression_level=self.config.get("storage.raw.compression_level", 4),
        )
        self.parquet_storage = ParquetStorage(processed_path)

        # Current session state
        self._session: Session | None = None
        self._stream: EEGStream | None = None
        self._file_path: Path | None = None
        self._is_recording = False

    def start_session(
        self,
        scenario_name: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Start a new recording session.

        Args:
            scenario_name: Name of scenario.
            session_id: Custom session ID. Auto-generated if None.

        Returns:
            Active session instance.

        Raises:
            RuntimeError: If session already active.
        """
        if self._is_recording:
            raise RuntimeError("Session already in progress")

        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create session
        self._session = Session(session_id, scenario_name, self.config)

        # Initialize EEG stream
        self._stream = EEGStream(self.config)
        self._stream.start()

        # Update session metadata with device info
        metadata = SessionMetadata(
            session_id=session_id,
            start_time=self._session.start_time,
            scenario_name=scenario_name,
            device_type=self.config.get("eeg.device_type", "unknown"),
            sampling_rate=self._stream.sampling_rate,
            channels=self._stream.channel_names,
        )

        # Create HDF5 file
        self._file_path = self.hdf5_storage.create_session_file(session_id, metadata)

        self._is_recording = True

        # Add start marker
        self._session.add_marker("session_start")

        return self._session

    def stop_session(self) -> Session | None:
        """Stop current recording session.

        Returns:
            Completed session or None if no active session.
        """
        if not self._is_recording or not self._session:
            return None

        # Add end marker
        self._session.add_marker("session_end")

        # Stop streaming
        if self._stream:
            # Get any remaining data
            try:
                data = self._stream.get_data()
                if data.size > 0:
                    timestamps = np.linspace(
                        time.time() - data.shape[1] / self._stream.sampling_rate,
                        time.time(),
                        data.shape[1],
                    )
                    self.hdf5_storage.append_eeg_data(self._file_path, data, timestamps)
            except Exception:
                pass

            self._stream.stop()
            self._stream = None

        # End session
        self._session.end()

        # Save metadata
        self.parquet_storage.save_session_metadata(self._session.get_metadata())

        # Save markers
        if self._session.markers:
            self.parquet_storage.save_markers(
                self._session.session_id,
                self._session.markers,
            )

        self._is_recording = False

        session = self._session
        self._session = None
        self._file_path = None

        return session

    def add_marker(self, name: str, value: Any = None, notes: str = "") -> Marker | None:
        """Add marker to current session.

        Args:
            name: Marker name.
            value: Optional value.
            notes: Optional notes.

        Returns:
            Created marker or None if no active session.
        """
        if not self._session:
            return None

        marker = self._session.add_marker(name, value, notes)

        # Also save to HDF5
        if self._file_path:
            self.hdf5_storage.add_marker(self._file_path, marker)

        return marker

    def add_gps(
        self,
        latitude: float,
        longitude: float,
        altitude: float | None = None,
        accuracy: float | None = None,
    ) -> GPSPoint | None:
        """Add GPS point to current session.

        Args:
            latitude: Latitude.
            longitude: Longitude.
            altitude: Altitude.
            accuracy: Accuracy.

        Returns:
            Created GPS point or None.
        """
        if not self._session:
            return None

        point = self._session.add_gps(latitude, longitude, altitude, accuracy)

        # Also save to HDF5
        if self._file_path:
            self.hdf5_storage.add_gps_point(self._file_path, point)

        return point

    def collect_chunk(self, chunk_duration: float = 1.0) -> int:
        """Collect and save a chunk of EEG data.

        Args:
            chunk_duration: Duration of chunk in seconds.

        Returns:
            Number of samples collected.
        """
        if not self._is_recording or not self._stream:
            return 0

        try:
            # Wait for data to accumulate
            time.sleep(chunk_duration)

            # Get data from stream
            samples = int(chunk_duration * self._stream.sampling_rate)
            data = self._stream.get_data(samples)

            if data.size == 0:
                return 0

            # Generate timestamps
            end_time = time.time()
            start_time = end_time - data.shape[1] / self._stream.sampling_rate
            timestamps = np.linspace(start_time, end_time, data.shape[1])

            # Save to HDF5
            self.hdf5_storage.append_eeg_data(self._file_path, data, timestamps)

            return data.shape[1]

        except Exception as e:
            print(f"Error collecting chunk: {e}")
            return 0

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def current_session(self) -> Session | None:
        """Get current session."""
        return self._session

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_recording:
            self.stop_session()
        return False
