"""Experiment tracking and versioning for reproducible research.

Provides tools for tracking experiments, parameters, metrics,
and artifacts for reproducibility.
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
class ExperimentRun:
    """A single experiment run.

    Attributes:
        run_id: Unique run identifier.
        experiment_name: Parent experiment name.
        start_time: Run start time.
        end_time: Run end time.
        status: Run status.
        parameters: Run parameters.
        metrics: Logged metrics.
        artifacts: Artifact paths.
        tags: Run tags.
        notes: User notes.
    """

    run_id: str
    experiment_name: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "running"
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "notes": self.notes,
        }


class ExperimentTracker:
    """Track experiments for reproducibility.

    Provides tools for logging parameters, metrics, and artifacts
    across experiment runs.

    Example:
        >>> tracker = ExperimentTracker("./experiments")
        >>> with tracker.start_run("my_experiment") as run:
        ...     tracker.log_params({"lr": 0.01})
        ...     tracker.log_metric("accuracy", 0.95)
    """

    def __init__(self, tracking_dir: str | Path = "./experiments"):
        """Initialize experiment tracker.

        Args:
            tracking_dir: Directory for storing experiment data.
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self._experiments: dict[str, list[ExperimentRun]] = {}
        self._active_run: ExperimentRun | None = None

        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load existing experiments from disk."""
        for exp_dir in self.tracking_dir.iterdir():
            if exp_dir.is_dir():
                exp_name = exp_dir.name
                self._experiments[exp_name] = []

                for run_file in exp_dir.glob("*/run.json"):
                    try:
                        with open(run_file) as f:
                            data = json.load(f)
                            run = ExperimentRun(
                                run_id=data["run_id"],
                                experiment_name=data["experiment_name"],
                                start_time=datetime.fromisoformat(data["start_time"]),
                                end_time=(
                                    datetime.fromisoformat(data["end_time"])
                                    if data.get("end_time") else None
                                ),
                                status=data.get("status", "completed"),
                                parameters=data.get("parameters", {}),
                                metrics=data.get("metrics", {}),
                                artifacts=data.get("artifacts", []),
                                tags=data.get("tags", {}),
                                notes=data.get("notes", ""),
                            )
                            self._experiments[exp_name].append(run)
                    except Exception as e:
                        logger.error(f"Error loading run {run_file}: {e}")

    def create_experiment(self, name: str, tags: dict[str, str] | None = None) -> str:
        """Create a new experiment.

        Args:
            name: Experiment name.
            tags: Experiment tags.

        Returns:
            Experiment name.
        """
        exp_dir = self.tracking_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if name not in self._experiments:
            self._experiments[name] = []

        # Save experiment metadata
        meta_file = exp_dir / "experiment.json"
        with open(meta_file, "w") as f:
            json.dump({
                "name": name,
                "created_at": datetime.now().isoformat(),
                "tags": tags or {},
            }, f, indent=2)

        logger.info(f"Created experiment: {name}")
        return name

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> "ExperimentTracker":
        """Start a new run.

        Args:
            experiment_name: Experiment to run under.
            run_name: Optional run name.
            tags: Run tags.

        Returns:
            Self for context manager usage.
        """
        if experiment_name not in self._experiments:
            self.create_experiment(experiment_name)

        run_id = run_name or str(uuid.uuid4())[:8]

        self._active_run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            start_time=datetime.now(),
            tags=tags or {},
        )

        # Create run directory
        run_dir = self.tracking_dir / experiment_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Started run: {experiment_name}/{run_id}")
        return self

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.end_run(status="failed" if exc_type else "completed")

    def end_run(self, status: str = "completed") -> None:
        """End the current run.

        Args:
            status: Final run status.
        """
        if not self._active_run:
            return

        self._active_run.end_time = datetime.now()
        self._active_run.status = status

        # Save run data
        run_dir = (
            self.tracking_dir
            / self._active_run.experiment_name
            / self._active_run.run_id
        )
        run_file = run_dir / "run.json"

        with open(run_file, "w") as f:
            json.dump(self._active_run.to_dict(), f, indent=2)

        self._experiments[self._active_run.experiment_name].append(self._active_run)

        logger.info(
            f"Ended run: {self._active_run.experiment_name}/"
            f"{self._active_run.run_id} ({status})"
        )
        self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for current run.

        Args:
            params: Parameters to log.
        """
        if not self._active_run:
            logger.warning("No active run")
            return

        self._active_run.parameters.update(params)
        logger.debug(f"Logged params: {list(params.keys())}")

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if not self._active_run:
            logger.warning("No active run")
            return

        if name not in self._active_run.metrics:
            self._active_run.metrics[name] = []

        timestamp = datetime.now().timestamp()
        self._active_run.metrics[name].append((timestamp, value))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Metrics to log.
            step: Optional step number.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, local_path: str | Path, artifact_name: str | None = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to artifact file.
            artifact_name: Name in artifacts directory.
        """
        if not self._active_run:
            logger.warning("No active run")
            return

        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"Artifact not found: {local_path}")
            return

        artifact_name = artifact_name or local_path.name
        run_dir = (
            self.tracking_dir
            / self._active_run.experiment_name
            / self._active_run.run_id
        )
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        dest_path = artifacts_dir / artifact_name

        # Copy file
        import shutil
        shutil.copy2(local_path, dest_path)

        self._active_run.artifacts.append(artifact_name)
        logger.debug(f"Logged artifact: {artifact_name}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key.
            value: Tag value.
        """
        if not self._active_run:
            return
        self._active_run.tags[key] = value

    def get_run(self, experiment_name: str, run_id: str) -> ExperimentRun | None:
        """Get a specific run.

        Args:
            experiment_name: Experiment name.
            run_id: Run ID.

        Returns:
            ExperimentRun if found.
        """
        if experiment_name not in self._experiments:
            return None

        for run in self._experiments[experiment_name]:
            if run.run_id == run_id:
                return run
        return None

    def list_experiments(self) -> list[str]:
        """List all experiments.

        Returns:
            List of experiment names.
        """
        return list(self._experiments.keys())

    def list_runs(self, experiment_name: str) -> list[ExperimentRun]:
        """List runs for an experiment.

        Args:
            experiment_name: Experiment name.

        Returns:
            List of runs.
        """
        return self._experiments.get(experiment_name, [])

    def compare_runs(
        self,
        experiment_name: str,
        run_ids: list[str],
        metrics: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare multiple runs.

        Args:
            experiment_name: Experiment name.
            run_ids: Run IDs to compare.
            metrics: Specific metrics to compare.

        Returns:
            Comparison dictionary.
        """
        comparison = {}

        for run_id in run_ids:
            run = self.get_run(experiment_name, run_id)
            if not run:
                continue

            run_data = {
                "parameters": run.parameters,
                "metrics": {},
            }

            # Get final metric values
            for metric_name, values in run.metrics.items():
                if metrics and metric_name not in metrics:
                    continue
                if values:
                    run_data["metrics"][metric_name] = values[-1][1]

            comparison[run_id] = run_data

        return comparison

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "max",
    ) -> ExperimentRun | None:
        """Get the best run based on a metric.

        Args:
            experiment_name: Experiment name.
            metric: Metric to optimize.
            mode: "max" or "min".

        Returns:
            Best run.
        """
        runs = self.list_runs(experiment_name)
        if not runs:
            return None

        best_run = None
        best_value = None

        for run in runs:
            if metric not in run.metrics or not run.metrics[metric]:
                continue

            value = run.metrics[metric][-1][1]  # Last value

            if best_value is None:
                best_value = value
                best_run = run
            elif mode == "max" and value > best_value:
                best_value = value
                best_run = run
            elif mode == "min" and value < best_value:
                best_value = value
                best_run = run

        return best_run


# Global tracker
_global_tracker: ExperimentTracker | None = None


def get_tracker(tracking_dir: str | Path = "./experiments") -> ExperimentTracker:
    """Get the global experiment tracker.

    Args:
        tracking_dir: Tracking directory.

    Returns:
        Global ExperimentTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ExperimentTracker(tracking_dir)
    return _global_tracker


def create_experiment(name: str, tags: dict[str, str] | None = None) -> str:
    """Create experiment using global tracker.

    Args:
        name: Experiment name.
        tags: Experiment tags.

    Returns:
        Experiment name.
    """
    return get_tracker().create_experiment(name, tags)


def log_params(params: dict[str, Any]) -> None:
    """Log parameters using global tracker.

    Args:
        params: Parameters to log.
    """
    get_tracker().log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics using global tracker.

    Args:
        metrics: Metrics to log.
        step: Optional step number.
    """
    get_tracker().log_metrics(metrics, step)
