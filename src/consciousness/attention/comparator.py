"""Human-AI attention comparison tools.

Compare attention patterns between human EEG data and
AI model attention weights for consciousness research.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class HumanAIAlignment:
    """Results of human-AI attention alignment analysis.

    Attributes:
        correlation: Correlation between attention patterns.
        overlap: Attention overlap score.
        divergence: KL divergence between distributions.
        temporal_sync: Temporal synchronization score.
        spatial_alignment: Spatial alignment across regions.
    """

    correlation: float
    overlap: float
    divergence: float
    temporal_sync: float
    spatial_alignment: dict[str, float]


class AttentionComparator:
    """Compare human EEG attention with AI model attention.

    Provides tools for analyzing alignment between human
    neural attention and AI transformer attention patterns.

    Example:
        >>> comparator = AttentionComparator()
        >>> alignment = comparator.compare(
        ...     human_attention=eeg_attention,
        ...     ai_attention=transformer_attention
        ... )
        >>> print(f"Correlation: {alignment.correlation:.3f}")
    """

    def __init__(
        self,
        n_regions: int = 4,
        region_names: list[str] | None = None,
    ):
        """Initialize comparator.

        Args:
            n_regions: Number of spatial regions.
            region_names: Names for regions (e.g., channel names).
        """
        self.n_regions = n_regions
        self.region_names = region_names or [f"Region_{i}" for i in range(n_regions)]

    def compare(
        self,
        human_attention: np.ndarray,
        ai_attention: np.ndarray,
        normalize: bool = True,
    ) -> HumanAIAlignment:
        """Compare human and AI attention patterns.

        Args:
            human_attention: Human attention (regions,) or (time, regions).
            ai_attention: AI attention, same shape as human.
            normalize: Normalize attention distributions.

        Returns:
            HumanAIAlignment results.
        """
        # Ensure same shape
        if human_attention.shape != ai_attention.shape:
            raise ValueError(
                f"Shape mismatch: human {human_attention.shape} vs AI {ai_attention.shape}"
            )

        # Normalize if needed
        if normalize:
            human_attention = self._normalize(human_attention)
            ai_attention = self._normalize(ai_attention)

        # Compute metrics
        correlation = self._compute_correlation(human_attention, ai_attention)
        overlap = self._compute_overlap(human_attention, ai_attention)
        divergence = self._compute_kl_divergence(human_attention, ai_attention)

        # Temporal sync if 2D
        if human_attention.ndim == 2:
            temporal_sync = self._compute_temporal_sync(human_attention, ai_attention)
        else:
            temporal_sync = 1.0

        # Spatial alignment per region
        spatial_alignment = self._compute_spatial_alignment(
            human_attention, ai_attention
        )

        return HumanAIAlignment(
            correlation=correlation,
            overlap=overlap,
            divergence=divergence,
            temporal_sync=temporal_sync,
            spatial_alignment=spatial_alignment,
        )

    def _normalize(self, attention: np.ndarray) -> np.ndarray:
        """Normalize attention to probability distribution.

        Args:
            attention: Raw attention values.

        Returns:
            Normalized attention (sums to 1 per time step).
        """
        if attention.ndim == 1:
            total = np.sum(attention) + 1e-10
            return attention / total
        else:
            # Normalize each time step
            totals = np.sum(attention, axis=1, keepdims=True) + 1e-10
            return attention / totals

    def _compute_correlation(
        self,
        human: np.ndarray,
        ai: np.ndarray,
    ) -> float:
        """Compute Pearson correlation.

        Args:
            human: Human attention.
            ai: AI attention.

        Returns:
            Correlation coefficient.
        """
        if human.ndim == 1:
            corr = np.corrcoef(human, ai)[0, 1]
        else:
            # Mean correlation across time
            correlations = []
            for t in range(len(human)):
                if np.std(human[t]) > 0 and np.std(ai[t]) > 0:
                    corr = np.corrcoef(human[t], ai[t])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            corr = np.mean(correlations) if correlations else 0

        return float(corr) if not np.isnan(corr) else 0.0

    def _compute_overlap(
        self,
        human: np.ndarray,
        ai: np.ndarray,
    ) -> float:
        """Compute attention overlap (intersection).

        Args:
            human: Human attention distribution.
            ai: AI attention distribution.

        Returns:
            Overlap score (0-1).
        """
        if human.ndim == 1:
            overlap = np.sum(np.minimum(human, ai))
        else:
            overlaps = [np.sum(np.minimum(human[t], ai[t])) for t in range(len(human))]
            overlap = np.mean(overlaps)

        return float(overlap)

    def _compute_kl_divergence(
        self,
        human: np.ndarray,
        ai: np.ndarray,
    ) -> float:
        """Compute KL divergence from human to AI.

        Args:
            human: Human attention distribution.
            ai: AI attention distribution.

        Returns:
            KL divergence (lower = more similar).
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10

        if human.ndim == 1:
            human = human + eps
            ai = ai + eps
            divergence = np.sum(human * np.log(human / ai))
        else:
            divergences = []
            for t in range(len(human)):
                h = human[t] + eps
                a = ai[t] + eps
                divergences.append(np.sum(h * np.log(h / a)))
            divergence = np.mean(divergences)

        return float(divergence)

    def _compute_temporal_sync(
        self,
        human: np.ndarray,
        ai: np.ndarray,
    ) -> float:
        """Compute temporal synchronization.

        Measures how well attention changes are synchronized
        between human and AI over time.

        Args:
            human: Human attention (time, regions).
            ai: AI attention (time, regions).

        Returns:
            Synchronization score (0-1).
        """
        if len(human) < 2:
            return 1.0

        # Compute attention "velocity" (change over time)
        human_vel = np.diff(human, axis=0)
        ai_vel = np.diff(ai, axis=0)

        # Correlation of velocities
        correlations = []
        for t in range(len(human_vel)):
            if np.std(human_vel[t]) > 0 and np.std(ai_vel[t]) > 0:
                corr = np.corrcoef(human_vel[t], ai_vel[t])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def _compute_spatial_alignment(
        self,
        human: np.ndarray,
        ai: np.ndarray,
    ) -> dict[str, float]:
        """Compute alignment for each spatial region.

        Args:
            human: Human attention.
            ai: AI attention.

        Returns:
            Per-region alignment scores.
        """
        if human.ndim == 1:
            n_regions = len(human)
        else:
            n_regions = human.shape[1]

        alignment = {}
        for i in range(n_regions):
            region_name = self.region_names[i] if i < len(self.region_names) else f"Region_{i}"

            if human.ndim == 1:
                # Single time point
                diff = abs(human[i] - ai[i])
                align_score = 1 - min(diff, 1)
            else:
                # Mean alignment over time
                diffs = np.abs(human[:, i] - ai[:, i])
                align_score = float(1 - np.mean(np.minimum(diffs, 1)))

            alignment[region_name] = align_score

        return alignment

    def analyze_over_time(
        self,
        human_attention: np.ndarray,
        ai_attention: np.ndarray,
        window_size: int = 10,
    ) -> dict[str, np.ndarray]:
        """Analyze alignment over time with sliding window.

        Args:
            human_attention: Human attention (time, regions).
            ai_attention: AI attention (time, regions).
            window_size: Window size for analysis.

        Returns:
            Time series of alignment metrics.
        """
        n_steps = len(human_attention)
        n_windows = n_steps - window_size + 1

        correlations = []
        overlaps = []
        divergences = []

        for i in range(n_windows):
            h_window = human_attention[i:i + window_size]
            a_window = ai_attention[i:i + window_size]

            result = self.compare(h_window, a_window, normalize=True)
            correlations.append(result.correlation)
            overlaps.append(result.overlap)
            divergences.append(result.divergence)

        return {
            "correlation": np.array(correlations),
            "overlap": np.array(overlaps),
            "divergence": np.array(divergences),
            "timestamps": np.arange(n_windows),
        }


def compute_attention_overlap(
    attention_a: np.ndarray,
    attention_b: np.ndarray,
) -> float:
    """Compute overlap between two attention distributions.

    Args:
        attention_a: First attention distribution.
        attention_b: Second attention distribution.

    Returns:
        Overlap score (0-1).
    """
    # Normalize
    a = attention_a / (np.sum(attention_a) + 1e-10)
    b = attention_b / (np.sum(attention_b) + 1e-10)

    # Intersection
    return float(np.sum(np.minimum(a, b)))


def visualize_attention_comparison(
    human_attention: np.ndarray,
    ai_attention: np.ndarray,
    region_names: list[str] | None = None,
    title: str = "Human-AI Attention Comparison",
) -> Any:
    """Create visualization comparing human and AI attention.

    Args:
        human_attention: Human attention (regions,) or (time, regions).
        ai_attention: AI attention, same shape.
        region_names: Names for regions.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for visualization")

    is_temporal = human_attention.ndim == 2

    if is_temporal:
        # Create subplot with heatmaps and time series
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Human Attention", "AI Attention",
                "Attention Over Time", "Alignment"
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
        )

        n_regions = human_attention.shape[1]
        if region_names is None:
            region_names = [f"R{i}" for i in range(n_regions)]

        # Heatmaps
        fig.add_trace(
            go.Heatmap(
                z=human_attention.T,
                y=region_names,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=ai_attention.T,
                y=region_names,
                colorscale="Viridis",
                showscale=False,
            ),
            row=1, col=2
        )

        # Time series (mean attention)
        times = np.arange(len(human_attention))
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.mean(human_attention, axis=1),
                name="Human",
                line=dict(color="blue"),
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.mean(ai_attention, axis=1),
                name="AI",
                line=dict(color="red"),
            ),
            row=2, col=1
        )

        # Alignment over time
        alignment = np.sum(np.minimum(human_attention, ai_attention), axis=1)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=alignment,
                name="Overlap",
                fill="tozeroy",
            ),
            row=2, col=2
        )

    else:
        # Bar chart comparison
        n_regions = len(human_attention)
        if region_names is None:
            region_names = [f"R{i}" for i in range(n_regions)]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Human",
            x=region_names,
            y=human_attention,
            marker_color="blue",
        ))

        fig.add_trace(go.Bar(
            name="AI",
            x=region_names,
            y=ai_attention,
            marker_color="red",
        ))

        fig.update_layout(barmode="group")

    fig.update_layout(
        title=title,
        template="plotly_white",
    )

    return fig


def compute_attention_similarity_matrix(
    attention_patterns: list[np.ndarray],
    labels: list[str] | None = None,
) -> tuple[np.ndarray, Any]:
    """Compute similarity matrix between multiple attention patterns.

    Args:
        attention_patterns: List of attention arrays.
        labels: Labels for each pattern.

    Returns:
        Tuple of (similarity_matrix, plotly_figure).
    """
    n = len(attention_patterns)

    # Compute pairwise similarities
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                similarity[i, j] = 1.0
            else:
                overlap = compute_attention_overlap(
                    attention_patterns[i].flatten(),
                    attention_patterns[j].flatten()
                )
                similarity[i, j] = overlap

    # Create visualization
    fig = None
    if PLOTLY_AVAILABLE:
        if labels is None:
            labels = [f"Pattern {i}" for i in range(n)]

        fig = go.Figure(data=go.Heatmap(
            z=similarity,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0.5,
        ))

        fig.update_layout(
            title="Attention Pattern Similarity",
            template="plotly_white",
        )

    return similarity, fig
