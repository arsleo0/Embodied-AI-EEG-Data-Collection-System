"""MLflow integration for experiment tracking.

Provides MLflow-based experiment tracking, metric logging,
model versioning, and artifact management.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """MLflow configuration.

    Attributes:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Default experiment name.
        artifact_location: Artifact storage location.
        registry_uri: Model registry URI.
    """

    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "consciousness-research"
    artifact_location: str | None = None
    registry_uri: str | None = None


class MLflowTracker:
    """MLflow-based experiment tracker.

    Provides integration with MLflow for comprehensive
    experiment tracking and model management.

    Example:
        >>> tracker = MLflowTracker()
        >>> with tracker.start_run("my_run"):
        ...     tracker.log_params({"lr": 0.01})
        ...     tracker.log_metric("loss", 0.5)
    """

    def __init__(self, config: MLflowConfig | None = None):
        """Initialize MLflow tracker.

        Args:
            config: MLflow configuration.
        """
        self.config = config or MLflowConfig()
        self._mlflow = None
        self._active_run = None

        self._setup()

    def _setup(self) -> None:
        """Setup MLflow connection."""
        try:
            import mlflow
            self._mlflow = mlflow

            mlflow.set_tracking_uri(self.config.tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                )

            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLflow initialized: {self.config.tracking_uri}")

        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            self._mlflow = None

    @property
    def is_available(self) -> bool:
        """Check if MLflow is available."""
        return self._mlflow is not None

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> "MLflowTracker":
        """Start a new MLflow run.

        Args:
            run_name: Run name.
            tags: Run tags.
            nested: Whether this is a nested run.

        Returns:
            Self for context manager usage.
        """
        if not self.is_available:
            return self

        self._active_run = self._mlflow.start_run(
            run_name=run_name,
            nested=nested,
        )

        if tags:
            for key, value in tags.items():
                self._mlflow.set_tag(key, value)

        return self

    def __enter__(self) -> "MLflowTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.end_run()

    def end_run(self) -> None:
        """End the current run."""
        if self.is_available and self._active_run:
            self._mlflow.end_run()
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters.

        Args:
            params: Parameters to log.
        """
        if not self.is_available:
            return

        # Convert values to strings for MLflow
        str_params = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(str_params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Step number.
        """
        if not self.is_available:
            return

        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Metrics to log.
            step: Step number.
        """
        if not self.is_available:
            return

        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact.

        Args:
            local_path: Local file path.
            artifact_path: Path within artifacts.
        """
        if not self.is_available:
            return

        self._mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """Log a directory of artifacts.

        Args:
            local_dir: Local directory path.
            artifact_path: Path within artifacts.
        """
        if not self.is_available:
            return

        self._mlflow.log_artifacts(str(local_dir), artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str | None = None,
    ) -> None:
        """Log a model.

        Args:
            model: Model to log.
            artifact_path: Path within artifacts.
            registered_model_name: Name for model registry.
        """
        if not self.is_available:
            return

        # Try different model flavors
        try:
            from mlflow.sklearn import log_model
            log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )
        except Exception:
            # Fall back to pickle
            self._mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag.

        Args:
            key: Tag key.
            value: Tag value.
        """
        if not self.is_available:
            return

        self._mlflow.set_tag(key, value)

    def get_run_id(self) -> str | None:
        """Get current run ID.

        Returns:
            Run ID if active.
        """
        if self._active_run:
            return self._active_run.info.run_id
        return None

    def search_runs(
        self,
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for runs.

        Args:
            filter_string: MLflow filter string.
            order_by: Columns to order by.
            max_results: Maximum results.

        Returns:
            List of run dictionaries.
        """
        if not self.is_available:
            return []

        experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
        if not experiment:
            return []

        runs = self._mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results,
        )

        return runs.to_dict(orient="records")

    def load_model(self, model_uri: str) -> Any:
        """Load a logged model.

        Args:
            model_uri: Model URI.

        Returns:
            Loaded model.
        """
        if not self.is_available:
            return None

        return self._mlflow.sklearn.load_model(model_uri)

    def get_artifact_uri(self, artifact_path: str = "") -> str | None:
        """Get artifact URI.

        Args:
            artifact_path: Path within artifacts.

        Returns:
            Full artifact URI.
        """
        if not self.is_available or not self._active_run:
            return None

        base_uri = self._active_run.info.artifact_uri
        if artifact_path:
            return f"{base_uri}/{artifact_path}"
        return base_uri


def setup_mlflow(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "consciousness-research",
) -> MLflowTracker:
    """Setup and return an MLflow tracker.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Experiment name.

    Returns:
        Configured MLflowTracker.
    """
    config = MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    return MLflowTracker(config)
