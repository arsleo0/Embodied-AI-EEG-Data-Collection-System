"""Data provenance tracking for research reproducibility.

Tracks data lineage, processing history, and maintains
chain of custody for research data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import hashlib
import logging
import uuid


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStep:
    """A single processing step in data lineage.

    Attributes:
        step_id: Unique step identifier.
        operation: Operation name.
        timestamp: When step was executed.
        inputs: Input data identifiers.
        outputs: Output data identifiers.
        parameters: Operation parameters.
        software: Software/version used.
        duration: Processing duration.
        notes: Additional notes.
    """

    step_id: str
    operation: str
    timestamp: datetime
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    software: str = ""
    duration: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "software": self.software,
            "duration": self.duration,
            "notes": self.notes,
        }


@dataclass
class DataLineage:
    """Complete lineage for a data object.

    Attributes:
        data_id: Data object identifier.
        created_at: Creation time.
        source: Original data source.
        processing_steps: List of processing steps.
        metadata: Additional metadata.
        checksum: Data checksum.
    """

    data_id: str
    created_at: datetime
    source: str
    processing_steps: list[ProcessingStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_id": self.data_id,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "processing_steps": [s.to_dict() for s in self.processing_steps],
            "metadata": self.metadata,
            "checksum": self.checksum,
        }


class ProvenanceTracker:
    """Track data provenance and lineage.

    Maintains a complete record of data transformations
    for reproducibility and audit trails.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> tracker.register_data("raw_eeg", source="muse_device")
        >>> with tracker.track_operation("filter", inputs=["raw_eeg"]):
        ...     filtered = apply_filter(raw_data)
        >>> tracker.register_output("filtered_eeg", filtered)
    """

    def __init__(self, storage_dir: str | Path = "./provenance"):
        """Initialize provenance tracker.

        Args:
            storage_dir: Directory for provenance data.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._lineages: dict[str, DataLineage] = {}
        self._active_step: ProcessingStep | None = None
        self._step_start: datetime | None = None

        self._load_lineages()

    def _load_lineages(self) -> None:
        """Load existing lineage data."""
        for lineage_file in self.storage_dir.glob("*.json"):
            try:
                with open(lineage_file) as f:
                    data = json.load(f)
                    lineage = DataLineage(
                        data_id=data["data_id"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        source=data["source"],
                        metadata=data.get("metadata", {}),
                        checksum=data.get("checksum", ""),
                    )
                    for step_data in data.get("processing_steps", []):
                        step = ProcessingStep(
                            step_id=step_data["step_id"],
                            operation=step_data["operation"],
                            timestamp=datetime.fromisoformat(step_data["timestamp"]),
                            inputs=step_data.get("inputs", []),
                            outputs=step_data.get("outputs", []),
                            parameters=step_data.get("parameters", {}),
                            software=step_data.get("software", ""),
                            duration=step_data.get("duration", 0.0),
                            notes=step_data.get("notes", ""),
                        )
                        lineage.processing_steps.append(step)
                    self._lineages[lineage.data_id] = lineage
            except Exception as e:
                logger.error(f"Error loading lineage {lineage_file}: {e}")

    def _save_lineage(self, data_id: str) -> None:
        """Save lineage to disk.

        Args:
            data_id: Data identifier.
        """
        if data_id not in self._lineages:
            return

        lineage_file = self.storage_dir / f"{data_id}.json"
        with open(lineage_file, "w") as f:
            json.dump(self._lineages[data_id].to_dict(), f, indent=2)

    def register_data(
        self,
        data_id: str,
        source: str,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new data object.

        Args:
            data_id: Data identifier.
            source: Data source description.
            data: Actual data (for checksum).
            metadata: Additional metadata.

        Returns:
            Data ID.
        """
        checksum = ""
        if data is not None:
            checksum = self._compute_checksum(data)

        lineage = DataLineage(
            data_id=data_id,
            created_at=datetime.now(),
            source=source,
            metadata=metadata or {},
            checksum=checksum,
        )

        self._lineages[data_id] = lineage
        self._save_lineage(data_id)

        logger.info(f"Registered data: {data_id}")
        return data_id

    def _compute_checksum(self, data: Any) -> str:
        """Compute checksum for data.

        Args:
            data: Data to checksum.

        Returns:
            SHA256 checksum.
        """
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                return hashlib.sha256(data.tobytes()).hexdigest()
            else:
                return hashlib.sha256(str(data).encode()).hexdigest()
        except Exception:
            return ""

    def track_operation(
        self,
        operation: str,
        inputs: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        software: str = "",
    ) -> "ProvenanceTracker":
        """Start tracking an operation.

        Args:
            operation: Operation name.
            inputs: Input data IDs.
            parameters: Operation parameters.
            software: Software/version used.

        Returns:
            Self for context manager usage.
        """
        self._step_start = datetime.now()
        self._active_step = ProcessingStep(
            step_id=str(uuid.uuid4())[:8],
            operation=operation,
            timestamp=self._step_start,
            inputs=inputs or [],
            parameters=parameters or {},
            software=software,
        )
        return self

    def __enter__(self) -> "ProvenanceTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._active_step and self._step_start:
            self._active_step.duration = (
                datetime.now() - self._step_start
            ).total_seconds()
            self._active_step = None
            self._step_start = None

    def register_output(
        self,
        output_id: str,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an output from the current operation.

        Args:
            output_id: Output data ID.
            data: Output data.
            metadata: Additional metadata.
        """
        if not self._active_step:
            logger.warning("No active operation")
            return

        # Compute checksum
        checksum = ""
        if data is not None:
            checksum = self._compute_checksum(data)

        # Create lineage for output
        lineage = DataLineage(
            data_id=output_id,
            created_at=datetime.now(),
            source=f"operation:{self._active_step.operation}",
            metadata=metadata or {},
            checksum=checksum,
        )

        # Add processing step
        self._active_step.outputs.append(output_id)
        if self._step_start:
            self._active_step.duration = (
                datetime.now() - self._step_start
            ).total_seconds()
        lineage.processing_steps.append(self._active_step)

        # Copy steps from input lineages
        for input_id in self._active_step.inputs:
            if input_id in self._lineages:
                for step in self._lineages[input_id].processing_steps:
                    if step.step_id != self._active_step.step_id:
                        lineage.processing_steps.insert(0, step)

        self._lineages[output_id] = lineage
        self._save_lineage(output_id)

        logger.debug(f"Registered output: {output_id}")

    def get_lineage(self, data_id: str) -> DataLineage | None:
        """Get lineage for a data object.

        Args:
            data_id: Data identifier.

        Returns:
            DataLineage if found.
        """
        return self._lineages.get(data_id)

    def get_ancestors(self, data_id: str) -> list[str]:
        """Get all ancestor data IDs.

        Args:
            data_id: Data identifier.

        Returns:
            List of ancestor IDs.
        """
        ancestors = []
        lineage = self._lineages.get(data_id)

        if lineage:
            for step in lineage.processing_steps:
                for input_id in step.inputs:
                    if input_id not in ancestors:
                        ancestors.append(input_id)
                        ancestors.extend(self.get_ancestors(input_id))

        return list(set(ancestors))

    def get_descendants(self, data_id: str) -> list[str]:
        """Get all descendant data IDs.

        Args:
            data_id: Data identifier.

        Returns:
            List of descendant IDs.
        """
        descendants = []

        for other_id, lineage in self._lineages.items():
            for step in lineage.processing_steps:
                if data_id in step.inputs:
                    if other_id not in descendants:
                        descendants.append(other_id)
                        descendants.extend(self.get_descendants(other_id))
                    break

        return list(set(descendants))

    def verify_checksum(self, data_id: str, data: Any) -> bool:
        """Verify data integrity.

        Args:
            data_id: Data identifier.
            data: Data to verify.

        Returns:
            True if checksum matches.
        """
        lineage = self._lineages.get(data_id)
        if not lineage or not lineage.checksum:
            return True

        current_checksum = self._compute_checksum(data)
        return current_checksum == lineage.checksum

    def export_lineage(
        self,
        data_id: str,
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """Export lineage to file.

        Args:
            data_id: Data identifier.
            output_path: Output file path.
            format: Export format ("json", "markdown").
        """
        lineage = self._lineages.get(data_id)
        if not lineage:
            logger.error(f"Lineage not found: {data_id}")
            return

        output_path = Path(output_path)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(lineage.to_dict(), f, indent=2)
        elif format == "markdown":
            md = self._lineage_to_markdown(lineage)
            output_path.write_text(md)

        logger.info(f"Exported lineage to: {output_path}")

    def _lineage_to_markdown(self, lineage: DataLineage) -> str:
        """Convert lineage to Markdown.

        Args:
            lineage: DataLineage object.

        Returns:
            Markdown string.
        """
        md = f"# Data Lineage: {lineage.data_id}\n\n"
        md += f"**Source:** {lineage.source}\n"
        md += f"**Created:** {lineage.created_at.isoformat()}\n"
        if lineage.checksum:
            md += f"**Checksum:** {lineage.checksum[:16]}...\n"
        md += "\n## Processing Steps\n\n"

        for i, step in enumerate(lineage.processing_steps, 1):
            md += f"### {i}. {step.operation}\n\n"
            md += f"- **Time:** {step.timestamp.isoformat()}\n"
            md += f"- **Duration:** {step.duration:.3f}s\n"
            if step.inputs:
                md += f"- **Inputs:** {', '.join(step.inputs)}\n"
            if step.outputs:
                md += f"- **Outputs:** {', '.join(step.outputs)}\n"
            if step.parameters:
                md += f"- **Parameters:** {step.parameters}\n"
            if step.software:
                md += f"- **Software:** {step.software}\n"
            md += "\n"

        return md


# Global tracker
_global_provenance: ProvenanceTracker | None = None


def get_provenance_tracker(storage_dir: str | Path = "./provenance") -> ProvenanceTracker:
    """Get the global provenance tracker.

    Args:
        storage_dir: Storage directory.

    Returns:
        Global ProvenanceTracker instance.
    """
    global _global_provenance
    if _global_provenance is None:
        _global_provenance = ProvenanceTracker(storage_dir)
    return _global_provenance
