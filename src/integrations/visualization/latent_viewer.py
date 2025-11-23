"""3D latent space viewer for consciousness research.

Provides interactive 3D visualization of latent space
embeddings, trajectories, and clustering.
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import numpy as np


@dataclass
class ViewerConfig:
    """Configuration for latent space viewer.

    Attributes:
        title: Window title.
        width: Window width.
        height: Window height.
        point_size: Default point size.
        colormap: Default colormap.
        background_color: Background color.
        show_axes: Whether to show axes.
        show_grid: Whether to show grid.
    """

    title: str = "Latent Space Explorer"
    width: int = 1200
    height: int = 800
    point_size: int = 5
    colormap: str = "viridis"
    background_color: str = "white"
    show_axes: bool = True
    show_grid: bool = True


class LatentSpaceViewer:
    """Interactive 3D latent space viewer.

    Provides tools for exploring consciousness state
    embeddings in 3D space with various visualizations.

    Example:
        >>> viewer = LatentSpaceViewer()
        >>> viewer.set_data(embeddings, labels)
        >>> fig = viewer.create_figure()
    """

    def __init__(self, config: ViewerConfig | None = None):
        """Initialize latent space viewer.

        Args:
            config: Viewer configuration.
        """
        self.config = config or ViewerConfig()
        self.embeddings: np.ndarray | None = None
        self.labels: list[str] | None = None
        self.timestamps: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}

    def set_data(
        self,
        embeddings: np.ndarray,
        labels: list[str] | None = None,
        timestamps: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set data for visualization.

        Args:
            embeddings: N x 3 array of embeddings.
            labels: Optional labels for each point.
            timestamps: Optional timestamps.
            metadata: Additional metadata.
        """
        if embeddings.shape[1] != 3:
            raise ValueError("Embeddings must have 3 dimensions")

        self.embeddings = embeddings
        self.labels = labels
        self.timestamps = timestamps
        self.metadata = metadata or {}

    def create_figure(
        self,
        color_by: str = "label",
        show_trajectory: bool = False,
    ) -> Any:
        """Create plotly 3D figure.

        Args:
            color_by: What to color points by ("label", "time", "density").
            show_trajectory: Whether to show trajectory lines.

        Returns:
            Plotly figure object.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            raise ImportError("Plotly required: pip install plotly")

        if self.embeddings is None:
            raise ValueError("No data set. Call set_data() first.")

        # Determine colors
        if color_by == "label" and self.labels:
            colors = self.labels
            color_discrete_map = None
        elif color_by == "time" and self.timestamps is not None:
            colors = self.timestamps
            color_discrete_map = None
        else:
            # Color by density
            colors = self._compute_density()
            color_discrete_map = None

        # Create scatter plot
        fig = go.Figure()

        # Add points
        if self.labels:
            unique_labels = list(set(self.labels))
            for label in unique_labels:
                mask = np.array(self.labels) == label
                fig.add_trace(go.Scatter3d(
                    x=self.embeddings[mask, 0],
                    y=self.embeddings[mask, 1],
                    z=self.embeddings[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=self.config.point_size,
                        opacity=0.7,
                    ),
                    name=label,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "x: %{x:.3f}<br>"
                        "y: %{y:.3f}<br>"
                        "z: %{z:.3f}"
                        "<extra></extra>"
                    ),
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=self.embeddings[:, 0],
                y=self.embeddings[:, 1],
                z=self.embeddings[:, 2],
                mode='markers',
                marker=dict(
                    size=self.config.point_size,
                    color=colors if isinstance(colors, np.ndarray) else None,
                    colorscale=self.config.colormap,
                    opacity=0.7,
                ),
                hovertemplate=(
                    "x: %{x:.3f}<br>"
                    "y: %{y:.3f}<br>"
                    "z: %{z:.3f}"
                    "<extra></extra>"
                ),
            ))

        # Add trajectory
        if show_trajectory and self.timestamps is not None:
            sorted_idx = np.argsort(self.timestamps)
            fig.add_trace(go.Scatter3d(
                x=self.embeddings[sorted_idx, 0],
                y=self.embeddings[sorted_idx, 1],
                z=self.embeddings[sorted_idx, 2],
                mode='lines',
                line=dict(
                    color='rgba(128, 128, 128, 0.5)',
                    width=2,
                ),
                name='Trajectory',
                showlegend=True,
            ))

        # Update layout
        fig.update_layout(
            title=self.config.title,
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                bgcolor=self.config.background_color,
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
        )

        return fig

    def _compute_density(self) -> np.ndarray:
        """Compute local density for coloring.

        Returns:
            Density values for each point.
        """
        from scipy.spatial.distance import cdist

        if self.embeddings is None:
            return np.array([])

        # Compute pairwise distances
        distances = cdist(self.embeddings, self.embeddings)

        # Local density = number of neighbors within radius
        radius = np.percentile(distances, 10)
        density = np.sum(distances < radius, axis=1)

        return density

    def add_clusters(
        self,
        cluster_labels: np.ndarray,
        show_centroids: bool = True,
        show_hulls: bool = False,
    ) -> Any:
        """Add cluster visualization.

        Args:
            cluster_labels: Cluster label for each point.
            show_centroids: Whether to show cluster centroids.
            show_hulls: Whether to show convex hulls.

        Returns:
            Updated plotly figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly required: pip install plotly")

        if self.embeddings is None:
            raise ValueError("No data set.")

        fig = self.create_figure(color_by="label")

        # Compute centroids
        unique_clusters = np.unique(cluster_labels)

        if show_centroids:
            centroids = []
            for cluster in unique_clusters:
                mask = cluster_labels == cluster
                centroid = np.mean(self.embeddings[mask], axis=0)
                centroids.append(centroid)

            centroids = np.array(centroids)

            fig.add_trace(go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='diamond',
                    color='black',
                    opacity=0.8,
                ),
                name='Centroids',
            ))

        return fig

    def animate_trajectory(
        self,
        duration: float = 10.0,
        fps: int = 30,
    ) -> Any:
        """Create animated trajectory visualization.

        Args:
            duration: Animation duration in seconds.
            fps: Frames per second.

        Returns:
            Animated plotly figure.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly required: pip install plotly")

        if self.embeddings is None or self.timestamps is None:
            raise ValueError("Need embeddings and timestamps for animation")

        # Sort by time
        sorted_idx = np.argsort(self.timestamps)
        embeddings = self.embeddings[sorted_idx]

        n_frames = int(duration * fps)
        step = max(1, len(embeddings) // n_frames)

        # Create frames
        frames = []
        for i in range(0, len(embeddings), step):
            frame_data = embeddings[:i+1]
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(
                        x=frame_data[:, 0],
                        y=frame_data[:, 1],
                        z=frame_data[:, 2],
                        mode='lines+markers',
                        marker=dict(size=4),
                        line=dict(width=2),
                    )
                ],
                name=str(i),
            ))

        # Initial figure
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[embeddings[0, 0]],
                    y=[embeddings[0, 1]],
                    z=[embeddings[0, 2]],
                    mode='markers',
                    marker=dict(size=6),
                )
            ],
            frames=frames,
        )

        # Add animation controls
        fig.update_layout(
            title="Consciousness Trajectory Animation",
            width=self.config.width,
            height=self.config.height,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 1000/fps},
                                "transition": {"duration": 0},
                            }],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                            }],
                        ),
                    ],
                )
            ],
        )

        return fig

    def export_html(self, output_path: str | Path) -> Path:
        """Export interactive figure to HTML.

        Args:
            output_path: Output file path.

        Returns:
            Path to exported file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = self.create_figure()
        fig.write_html(str(output_path))

        return output_path


class LatentSpaceApp:
    """Streamlit app for latent space exploration.

    Provides an interactive web interface for exploring
    consciousness state embeddings.

    Example:
        >>> app = LatentSpaceApp()
        >>> app.run()
    """

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize latent space app.

        Args:
            data_dir: Directory containing embedding data.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.viewer = LatentSpaceViewer()

    def run(self) -> None:
        """Run the Streamlit app."""
        try:
            import streamlit as st
        except ImportError:
            raise ImportError("Streamlit required: pip install streamlit")

        st.set_page_config(
            page_title="Latent Space Explorer",
            page_icon="🧠",
            layout="wide",
        )

        st.title("🧠 Latent Space Explorer")

        # Sidebar controls
        with st.sidebar:
            st.header("Controls")

            # Data source
            data_source = st.selectbox(
                "Data Source",
                ["Upload", "Generate Sample"],
            )

            if data_source == "Upload":
                uploaded = st.file_uploader(
                    "Upload embeddings (NPZ)",
                    type=["npz"],
                )
                if uploaded:
                    data = np.load(uploaded)
                    embeddings = data.get('embeddings', data.get('arr_0'))
                    labels = data.get('labels', None)
                    if labels is not None:
                        labels = labels.tolist()
                    timestamps = data.get('timestamps', None)

                    self.viewer.set_data(embeddings, labels, timestamps)

            else:
                # Generate sample data
                n_points = st.slider("Number of points", 100, 1000, 300)
                n_clusters = st.slider("Number of clusters", 2, 6, 3)

                if st.button("Generate"):
                    embeddings, labels = self._generate_sample_data(
                        n_points, n_clusters
                    )
                    timestamps = np.linspace(0, 100, n_points)
                    self.viewer.set_data(embeddings, labels, timestamps)

            st.divider()

            # Visualization options
            st.subheader("Visualization")

            point_size = st.slider("Point size", 1, 15, 5)
            self.viewer.config.point_size = point_size

            show_trajectory = st.checkbox("Show trajectory", value=False)

            colormap = st.selectbox(
                "Colormap",
                ["viridis", "plasma", "inferno", "magma", "cividis"],
            )
            self.viewer.config.colormap = colormap

        # Main content
        if self.viewer.embeddings is not None:
            # Create and display figure
            fig = self.viewer.create_figure(
                show_trajectory=show_trajectory
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Points", len(self.viewer.embeddings))

            with col2:
                if self.viewer.labels:
                    st.metric("Unique States", len(set(self.viewer.labels)))

            with col3:
                spread = np.std(self.viewer.embeddings)
                st.metric("Spread (std)", f"{spread:.3f}")

            # Export options
            st.divider()
            st.subheader("Export")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export to HTML"):
                    output_path = self.viewer.export_html("latent_space.html")
                    st.success(f"Exported to {output_path}")

        else:
            st.info("Upload data or generate sample to begin.")

    def _generate_sample_data(
        self,
        n_points: int,
        n_clusters: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Generate sample embedding data.

        Args:
            n_points: Number of points.
            n_clusters: Number of clusters.

        Returns:
            Tuple of embeddings and labels.
        """
        # Generate clustered data
        points_per_cluster = n_points // n_clusters
        embeddings = []
        labels = []

        cluster_names = ["focus", "relaxed", "anxious", "creative", "flow", "drowsy"]

        for i in range(n_clusters):
            center = np.random.randn(3) * 2
            cluster_points = center + np.random.randn(points_per_cluster, 3) * 0.5
            embeddings.append(cluster_points)
            labels.extend([cluster_names[i % len(cluster_names)]] * points_per_cluster)

        embeddings = np.vstack(embeddings)

        return embeddings, labels


def create_3d_scatter(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    title: str = "3D Scatter Plot",
    point_size: int = 5,
) -> Any:
    """Create a simple 3D scatter plot.

    Args:
        embeddings: N x 3 array.
        labels: Optional point labels.
        title: Plot title.
        point_size: Marker size.

    Returns:
        Plotly figure.
    """
    viewer = LatentSpaceViewer()
    viewer.config.title = title
    viewer.config.point_size = point_size
    viewer.set_data(embeddings, labels)
    return viewer.create_figure()


def create_trajectory_plot(
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    title: str = "Consciousness Trajectory",
) -> Any:
    """Create a trajectory plot through latent space.

    Args:
        embeddings: N x 3 array.
        timestamps: Time values.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    viewer = LatentSpaceViewer()
    viewer.config.title = title
    viewer.set_data(embeddings, timestamps=timestamps)
    return viewer.create_figure(show_trajectory=True)


def create_cluster_visualization(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Cluster Visualization",
) -> Any:
    """Create a cluster visualization.

    Args:
        embeddings: N x 3 array.
        cluster_labels: Cluster assignment for each point.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    viewer = LatentSpaceViewer()
    viewer.config.title = title

    # Convert cluster labels to string labels
    labels = [f"Cluster {i}" for i in cluster_labels]
    viewer.set_data(embeddings, labels)

    return viewer.add_clusters(cluster_labels, show_centroids=True)
