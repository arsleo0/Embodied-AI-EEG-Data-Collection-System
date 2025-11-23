"""Interactive visualization for consciousness research.

Provides 3D latent space exploration and interactive
plotting capabilities for EEG analysis.
"""

from .latent_viewer import (
    LatentSpaceViewer,
    LatentSpaceApp,
    create_3d_scatter,
    create_trajectory_plot,
    create_cluster_visualization,
)
from .interactive_plots import (
    InteractivePlotter,
    TimeSeriesPlot,
    SpectrogramPlot,
    TopoPlot,
    BandPowerPlot,
    create_dashboard_layout,
)

__all__ = [
    # Latent space viewer
    "LatentSpaceViewer",
    "LatentSpaceApp",
    "create_3d_scatter",
    "create_trajectory_plot",
    "create_cluster_visualization",
    # Interactive plots
    "InteractivePlotter",
    "TimeSeriesPlot",
    "SpectrogramPlot",
    "TopoPlot",
    "BandPowerPlot",
    "create_dashboard_layout",
]
