"""Infrastructure for experiment management and tracking.

Provides versioning, MLflow integration, and data provenance
tracking for reproducible consciousness research.
"""

from .versioning import (
    ExperimentTracker,
    ExperimentRun,
    get_tracker,
    create_experiment,
    log_params,
    log_metrics,
)
from .mlflow_integration import (
    MLflowTracker,
    setup_mlflow,
)
from .provenance import (
    ProvenanceTracker,
    DataLineage,
    ProcessingStep,
    get_provenance_tracker,
)

__version__ = "1.0.0"

__all__ = [
    # Versioning
    "ExperimentTracker",
    "ExperimentRun",
    "get_tracker",
    "create_experiment",
    "log_params",
    "log_metrics",
    # MLflow
    "MLflowTracker",
    "setup_mlflow",
    # Provenance
    "ProvenanceTracker",
    "DataLineage",
    "ProcessingStep",
    "get_provenance_tracker",
]
