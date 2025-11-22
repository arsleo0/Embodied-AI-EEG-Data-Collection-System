"""Visualization tools for consciousness latent space.

Provides 2D/3D visualizations of EEG embeddings using
t-SNE, UMAP, and PCA with interactive Plotly plots.
"""

from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class LatentSpaceVisualizer:
    """Visualize embeddings in 2D/3D latent space.

    Supports multiple dimensionality reduction methods and
    creates interactive visualizations with Plotly.

    Example:
        >>> visualizer = LatentSpaceVisualizer(method="umap")
        >>> coords = visualizer.fit_transform(embeddings)
        >>> fig = visualizer.plot_2d(coords, labels)
    """

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize visualizer.

        Args:
            method: Reduction method ("umap", "tsne", "pca").
            n_components: Output dimensions (2 or 3).
            random_state: Random seed.
            **kwargs: Additional arguments for reducer.
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs

        self._reducer = None
        self._is_fitted = False

        # Initialize reducer
        self._init_reducer()

    def _init_reducer(self) -> None:
        """Initialize the dimensionality reducer."""
        if self.method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "umap-learn required. Install with: pip install umap-learn"
                )
            self._reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.kwargs.get("n_neighbors", 15),
                min_dist=self.kwargs.get("min_dist", 0.1),
                metric=self.kwargs.get("metric", "euclidean"),
            )

        elif self.method == "tsne":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for t-SNE")
            self._reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=self.kwargs.get("perplexity", 30),
                learning_rate=self.kwargs.get("learning_rate", "auto"),
                init=self.kwargs.get("init", "pca"),
            )

        elif self.method == "pca":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for PCA")
            self._reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, embeddings: np.ndarray) -> "LatentSpaceVisualizer":
        """Fit reducer to embeddings.

        Args:
            embeddings: Embedding matrix (samples, features).

        Returns:
            Self for chaining.
        """
        if self.method in ["umap", "pca"]:
            self._reducer.fit(embeddings)
        # t-SNE doesn't have separate fit
        self._is_fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to low-dimensional space.

        Args:
            embeddings: Embedding matrix.

        Returns:
            Reduced coordinates (samples, n_components).
        """
        if self.method == "tsne":
            # t-SNE always does fit_transform
            return self._reducer.fit_transform(embeddings)

        return self._reducer.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            embeddings: Embedding matrix.

        Returns:
            Reduced coordinates.
        """
        self._is_fitted = True
        return self._reducer.fit_transform(embeddings)

    def plot_2d(
        self,
        coords: np.ndarray,
        labels: list[str] | np.ndarray | None = None,
        title: str = "Consciousness Latent Space",
        color_map: dict[str, str] | None = None,
        marker_size: int = 8,
        opacity: float = 0.7,
        show_centroids: bool = True,
    ) -> Any:
        """Create interactive 2D scatter plot.

        Args:
            coords: 2D coordinates (samples, 2).
            labels: State labels for coloring.
            title: Plot title.
            color_map: Custom colors for states.
            marker_size: Point size.
            opacity: Point opacity.
            show_centroids: Show state centroids.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly required. Install with: pip install plotly"
            )

        if coords.shape[1] != 2:
            raise ValueError("Coordinates must be 2D")

        # Create figure
        fig = go.Figure()

        if labels is not None:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

            # Default color map
            if color_map is None:
                colors = px.colors.qualitative.Set2
                color_map = {
                    label: colors[i % len(colors)]
                    for i, label in enumerate(unique_labels)
                }

            # Plot each state
            for label in unique_labels:
                mask = labels == label
                fig.add_trace(go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=label,
                    marker=dict(
                        size=marker_size,
                        color=color_map.get(label, "gray"),
                        opacity=opacity,
                    ),
                    hovertemplate=f"{label}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>",
                ))

                # Add centroid
                if show_centroids:
                    centroid = np.mean(coords[mask], axis=0)
                    fig.add_trace(go.Scatter(
                        x=[centroid[0]],
                        y=[centroid[1]],
                        mode="markers+text",
                        name=f"{label} (centroid)",
                        text=[label],
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=color_map.get(label, "gray"),
                            symbol="x",
                            line=dict(width=2, color="black"),
                        ),
                        showlegend=False,
                    ))
        else:
            # Single color for all points
            fig.add_trace(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    opacity=opacity,
                ),
            ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=f"{self.method.upper()} 1",
            yaxis_title=f"{self.method.upper()} 2",
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
            ),
        )

        return fig

    def plot_3d(
        self,
        coords: np.ndarray,
        labels: list[str] | np.ndarray | None = None,
        title: str = "Consciousness Latent Space (3D)",
        color_map: dict[str, str] | None = None,
        marker_size: int = 5,
        opacity: float = 0.7,
    ) -> Any:
        """Create interactive 3D scatter plot.

        Args:
            coords: 3D coordinates (samples, 3).
            labels: State labels for coloring.
            title: Plot title.
            color_map: Custom colors for states.
            marker_size: Point size.
            opacity: Point opacity.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        if coords.shape[1] != 3:
            raise ValueError("Coordinates must be 3D")

        # Create figure
        fig = go.Figure()

        if labels is not None:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

            if color_map is None:
                colors = px.colors.qualitative.Set2
                color_map = {
                    label: colors[i % len(colors)]
                    for i, label in enumerate(unique_labels)
                }

            for label in unique_labels:
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode="markers",
                    name=label,
                    marker=dict(
                        size=marker_size,
                        color=color_map.get(label, "gray"),
                        opacity=opacity,
                    ),
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    opacity=opacity,
                ),
            ))

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"{self.method.upper()} 1",
                yaxis_title=f"{self.method.upper()} 2",
                zaxis_title=f"{self.method.upper()} 3",
            ),
            template="plotly_white",
        )

        return fig

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        coords: np.ndarray,
        labels: list[str] | None,
        path: str | Path,
    ) -> None:
        """Save embeddings and coordinates to file.

        Args:
            embeddings: Original embeddings.
            coords: Reduced coordinates.
            labels: State labels.
            path: Output file path (.npz).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            embeddings=embeddings,
            coords=coords,
            labels=np.array(labels) if labels else None,
            method=self.method,
            n_components=self.n_components,
        )

    @staticmethod
    def load_embeddings(path: str | Path) -> dict[str, Any]:
        """Load saved embeddings.

        Args:
            path: Input file path.

        Returns:
            Dictionary with embeddings, coords, labels, metadata.
        """
        data = np.load(path, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "coords": data["coords"],
            "labels": data["labels"] if data["labels"] is not None else None,
            "method": str(data["method"]),
            "n_components": int(data["n_components"]),
        }


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray | None = None,
    method: str = "umap",
    title: str = "Consciousness Latent Space",
    **kwargs,
) -> Any:
    """Convenience function to visualize embeddings in 2D.

    Args:
        embeddings: Embedding matrix (samples, features).
        labels: State labels.
        method: Reduction method.
        title: Plot title.
        **kwargs: Additional visualizer arguments.

    Returns:
        Plotly figure.
    """
    visualizer = LatentSpaceVisualizer(method=method, n_components=2, **kwargs)
    coords = visualizer.fit_transform(embeddings)
    return visualizer.plot_2d(coords, labels, title)


def plot_embeddings_3d(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray | None = None,
    method: str = "umap",
    title: str = "Consciousness Latent Space (3D)",
    **kwargs,
) -> Any:
    """Convenience function to visualize embeddings in 3D.

    Args:
        embeddings: Embedding matrix (samples, features).
        labels: State labels.
        method: Reduction method.
        title: Plot title.
        **kwargs: Additional visualizer arguments.

    Returns:
        Plotly figure.
    """
    visualizer = LatentSpaceVisualizer(method=method, n_components=3, **kwargs)
    coords = visualizer.fit_transform(embeddings)
    return visualizer.plot_3d(coords, labels, title)


def plot_trajectory(
    coords: np.ndarray,
    timestamps: np.ndarray | None = None,
    labels: list[str] | np.ndarray | None = None,
    title: str = "Consciousness Trajectory",
    show_direction: bool = True,
    line_width: float = 1.5,
) -> Any:
    """Plot consciousness trajectory over time.

    Args:
        coords: 2D or 3D coordinates in temporal order.
        timestamps: Time values for each point.
        labels: State labels.
        title: Plot title.
        show_direction: Show trajectory direction arrows.
        line_width: Line width.

    Returns:
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required")

    is_3d = coords.shape[1] == 3

    # Color by time or label
    if timestamps is not None:
        colors = timestamps
        colorscale = "Viridis"
        colorbar_title = "Time"
    else:
        colors = np.arange(len(coords))
        colorscale = "Viridis"
        colorbar_title = "Step"

    if is_3d:
        fig = go.Figure()

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines+markers",
            marker=dict(
                size=4,
                color=colors,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
            ),
            line=dict(
                width=line_width,
                color="gray",
            ),
            name="Trajectory",
        ))

        # Start and end markers
        fig.add_trace(go.Scatter3d(
            x=[coords[0, 0]],
            y=[coords[0, 1]],
            z=[coords[0, 2]],
            mode="markers+text",
            marker=dict(size=10, color="green", symbol="circle"),
            text=["Start"],
            name="Start",
        ))

        fig.add_trace(go.Scatter3d(
            x=[coords[-1, 0]],
            y=[coords[-1, 1]],
            z=[coords[-1, 2]],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="circle"),
            text=["End"],
            name="End",
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
                zaxis_title="Dim 3",
            ),
        )
    else:
        fig = go.Figure()

        # Trajectory line
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=colors,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
            ),
            line=dict(
                width=line_width,
                color="gray",
            ),
            name="Trajectory",
        ))

        # Start marker
        fig.add_trace(go.Scatter(
            x=[coords[0, 0]],
            y=[coords[0, 1]],
            mode="markers+text",
            marker=dict(size=12, color="green", symbol="circle"),
            text=["Start"],
            textposition="top center",
            name="Start",
            showlegend=True,
        ))

        # End marker
        fig.add_trace(go.Scatter(
            x=[coords[-1, 0]],
            y=[coords[-1, 1]],
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="circle"),
            text=["End"],
            textposition="top center",
            name="End",
            showlegend=True,
        ))

        # Direction arrows
        if show_direction and len(coords) > 10:
            # Add arrows at intervals
            n_arrows = min(10, len(coords) // 5)
            indices = np.linspace(0, len(coords) - 2, n_arrows, dtype=int)

            for i in indices:
                fig.add_annotation(
                    x=coords[i + 1, 0],
                    y=coords[i + 1, 1],
                    ax=coords[i, 0],
                    ay=coords[i, 1],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor="rgba(0,0,0,0.3)",
                )

        fig.update_layout(
            title=title,
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            template="plotly_white",
        )

    return fig


def create_comparison_plot(
    coords_list: list[np.ndarray],
    labels_list: list[str],
    title: str = "Embedding Comparison",
) -> Any:
    """Create side-by-side comparison of multiple embeddings.

    Args:
        coords_list: List of coordinate arrays.
        labels_list: Labels for each embedding set.
        title: Overall title.

    Returns:
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required")

    n_plots = len(coords_list)
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=labels_list,
    )

    colors = px.colors.qualitative.Set2

    for i, coords in enumerate(coords_list):
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                ),
                name=labels_list[i],
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False,
    )

    return fig
