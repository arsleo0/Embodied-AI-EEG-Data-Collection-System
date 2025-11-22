"""Data storage handlers for HDF5 and Parquet formats."""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from .models import SessionMetadata, Marker, GPSPoint, AudioSegment


class HDF5Storage:
    """HDF5 storage handler for EEG time-series data."""

    def __init__(
        self,
        base_path: str | Path,
        compression: str = "gzip",
        compression_level: int = 4,
    ):
        """Initialize HDF5 storage.

        Args:
            base_path: Base directory for HDF5 files.
            compression: Compression algorithm (gzip, lzf, szip).
            compression_level: Compression level (1-9 for gzip).
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.compression_level = compression_level

    def create_session_file(self, session_id: str, metadata: SessionMetadata) -> Path:
        """Create a new HDF5 file for a session.

        Args:
            session_id: Unique session identifier.
            metadata: Session metadata.

        Returns:
            Path to created file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}_{session_id[:8]}.h5"
        file_path = self.base_path / filename

        with h5py.File(file_path, "w") as f:
            # Create metadata group
            meta_group = f.create_group("metadata")
            for key, value in metadata.to_dict().items():
                if value is not None:
                    if isinstance(value, list):
                        # Store lists as datasets
                        if value:
                            meta_group.create_dataset(key, data=value)
                    else:
                        meta_group.attrs[key] = value

            # Create empty datasets for EEG data (will be resized)
            f.create_dataset(
                "eeg",
                shape=(len(metadata.channels), 0),
                maxshape=(len(metadata.channels), None),
                dtype="float32",
                compression=self.compression,
                compression_opts=self.compression_level,
                chunks=(len(metadata.channels), metadata.sampling_rate),
            )

            # Timestamps dataset
            f.create_dataset(
                "timestamps",
                shape=(0,),
                maxshape=(None,),
                dtype="float64",
                compression=self.compression,
            )

            # Markers group
            f.create_group("markers")

            # Sensors group
            f.create_group("sensors")

        return file_path

    def append_eeg_data(
        self,
        file_path: str | Path,
        data: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        """Append EEG data to session file.

        Args:
            file_path: Path to HDF5 file.
            data: EEG data array (channels, samples).
            timestamps: Timestamp array.
        """
        with h5py.File(file_path, "a") as f:
            eeg_dataset = f["eeg"]
            ts_dataset = f["timestamps"]

            # Current size
            current_samples = eeg_dataset.shape[1]
            new_samples = data.shape[1]

            # Resize datasets
            eeg_dataset.resize(current_samples + new_samples, axis=1)
            ts_dataset.resize(current_samples + new_samples, axis=0)

            # Append data
            eeg_dataset[:, current_samples:] = data
            ts_dataset[current_samples:] = timestamps

    def add_marker(self, file_path: str | Path, marker: Marker) -> None:
        """Add event marker to session file.

        Args:
            file_path: Path to HDF5 file.
            marker: Marker to add.
        """
        with h5py.File(file_path, "a") as f:
            markers_group = f["markers"]
            marker_id = str(len(markers_group))

            marker_group = markers_group.create_group(marker_id)
            marker_group.attrs["timestamp"] = marker.timestamp
            marker_group.attrs["name"] = marker.name
            if marker.value is not None:
                marker_group.attrs["value"] = str(marker.value)
            marker_group.attrs["notes"] = marker.notes

    def add_gps_point(self, file_path: str | Path, point: GPSPoint) -> None:
        """Add GPS point to session file.

        Args:
            file_path: Path to HDF5 file.
            point: GPS point to add.
        """
        with h5py.File(file_path, "a") as f:
            if "gps" not in f["sensors"]:
                # Create GPS dataset
                f["sensors"].create_dataset(
                    "gps",
                    shape=(0, 5),  # timestamp, lat, lon, alt, accuracy
                    maxshape=(None, 5),
                    dtype="float64",
                )

            gps_dataset = f["sensors/gps"]
            current_size = gps_dataset.shape[0]
            gps_dataset.resize(current_size + 1, axis=0)

            gps_dataset[current_size] = [
                point.timestamp,
                point.latitude,
                point.longitude,
                point.altitude or 0.0,
                point.accuracy or 0.0,
            ]

    def read_session(self, file_path: str | Path) -> dict[str, Any]:
        """Read complete session data from file.

        Args:
            file_path: Path to HDF5 file.

        Returns:
            Dictionary with session data.
        """
        with h5py.File(file_path, "r") as f:
            result = {
                "eeg": f["eeg"][:],
                "timestamps": f["timestamps"][:],
                "metadata": dict(f["metadata"].attrs),
                "markers": [],
            }

            # Read markers
            for marker_id in f["markers"]:
                marker_group = f["markers"][marker_id]
                result["markers"].append({
                    "timestamp": marker_group.attrs["timestamp"],
                    "name": marker_group.attrs["name"],
                    "notes": marker_group.attrs.get("notes", ""),
                })

            # Read GPS if available
            if "gps" in f["sensors"]:
                result["gps"] = f["sensors/gps"][:]

        return result


class ParquetStorage:
    """Parquet storage handler for metadata and logs."""

    def __init__(self, base_path: str | Path):
        """Initialize Parquet storage.

        Args:
            base_path: Base directory for Parquet files.
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "pyarrow is required. Install with: pip install pyarrow"
            )

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_session_metadata(
        self,
        metadata: SessionMetadata,
        filename: str = "sessions.parquet",
    ) -> Path:
        """Save session metadata to Parquet file.

        Args:
            metadata: Session metadata to save.
            filename: Output filename.

        Returns:
            Path to Parquet file.
        """
        file_path = self.base_path / filename

        # Convert to table
        data = metadata.to_dict()
        table = pa.Table.from_pydict({k: [v] for k, v in data.items()})

        if file_path.exists():
            # Append to existing file
            existing = pq.read_table(file_path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, file_path)
        return file_path

    def save_markers(
        self,
        session_id: str,
        markers: list[Marker],
        filename: str | None = None,
    ) -> Path:
        """Save markers to Parquet file.

        Args:
            session_id: Session identifier.
            markers: List of markers.
            filename: Output filename. Auto-generated if None.

        Returns:
            Path to Parquet file.
        """
        if filename is None:
            filename = f"markers_{session_id[:8]}.parquet"

        file_path = self.base_path / filename

        data = {
            "session_id": [session_id] * len(markers),
            "timestamp": [m.timestamp for m in markers],
            "name": [m.name for m in markers],
            "value": [str(m.value) if m.value else "" for m in markers],
            "notes": [m.notes for m in markers],
        }

        table = pa.Table.from_pydict(data)
        pq.write_table(table, file_path)

        return file_path

    def read_sessions(self, filename: str = "sessions.parquet") -> list[dict]:
        """Read all session metadata.

        Args:
            filename: Parquet filename.

        Returns:
            List of session metadata dictionaries.
        """
        file_path = self.base_path / filename

        if not file_path.exists():
            return []

        table = pq.read_table(file_path)
        return table.to_pylist()
