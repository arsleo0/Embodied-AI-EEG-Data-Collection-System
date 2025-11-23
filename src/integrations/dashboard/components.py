"""Reusable dashboard components.

Modular components for building consciousness research dashboards.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class ComponentConfig:
    """Configuration for dashboard components."""
    height: int = 300
    width: int | None = None
    theme: str = "plotly_white"
    show_legend: bool = True


class EEGSignalPlot:
    """Real-time EEG signal visualization component.

    Displays multi-channel EEG signals with configurable
    time window and channel display.

    Example:
        >>> plot = EEGSignalPlot(n_channels=4, fs=256)
        >>> fig = plot.create_figure(data, timestamps)
    """

    def __init__(
        self,
        n_channels: int = 4,
        fs: float = 256.0,
        channel_names: list[str] | None = None,
        config: ComponentConfig | None = None,
    ):
        """Initialize EEG signal plot.

        Args:
            n_channels: Number of channels.
            fs: Sampling frequency.
            channel_names: Channel names.
            config: Component configuration.
        """
        self.n_channels = n_channels
        self.fs = fs
        self.channel_names = channel_names or [
            f"Ch{i}" for i in range(n_channels)
        ]
        self.config = config or ComponentConfig()

        # Color palette
        self.colors = px.colors.qualitative.Set2 if PLOTLY_AVAILABLE else []

    def create_figure(
        self,
        data: np.ndarray,
        timestamps: np.ndarray | None = None,
        title: str = "EEG Signals",
    ) -> Any:
        """Create EEG signal figure.

        Args:
            data: EEG data (channels, samples).
            timestamps: Time values.
            title: Plot title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        if timestamps is None:
            timestamps = np.arange(data.shape[1]) / self.fs

        fig = go.Figure()

        # Calculate offset for channel separation
        channel_offset = np.max(np.abs(data)) * 2.5

        for i in range(min(self.n_channels, data.shape[0])):
            signal = data[i] - i * channel_offset
            color = self.colors[i % len(self.colors)]

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=signal,
                name=self.channel_names[i],
                line=dict(width=1, color=color),
                hovertemplate=f"{self.channel_names[i]}<br>" +
                              "Time: %{x:.2f}s<br>" +
                              "Value: %{customdata:.1f}μV<extra></extra>",
                customdata=data[i],
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=self.config.height,
            template=self.config.theme,
            showlegend=self.config.show_legend,
            yaxis=dict(showticklabels=False),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        return fig

    def create_spectrogram(
        self,
        data: np.ndarray,
        channel: int = 0,
        title: str = "Spectrogram",
    ) -> Any:
        """Create spectrogram for a channel.

        Args:
            data: EEG data.
            channel: Channel index.
            title: Plot title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        from scipy import signal as sig

        # Compute spectrogram
        f, t, Sxx = sig.spectrogram(
            data[channel], self.fs,
            nperseg=256, noverlap=128
        )

        # Limit frequency range
        freq_mask = f <= 50
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        fig = go.Figure(go.Heatmap(
            x=t,
            y=f,
            z=10 * np.log10(Sxx + 1e-10),
            colorscale="Viridis",
            colorbar=dict(title="Power (dB)"),
        ))

        fig.update_layout(
            title=f"{title} - {self.channel_names[channel]}",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=self.config.height,
            template=self.config.theme,
        )

        return fig


class ConsciousnessStateDisplay:
    """Display current consciousness state.

    Shows the detected state, confidence, and related metrics.

    Example:
        >>> display = ConsciousnessStateDisplay()
        >>> fig = display.create_gauge(state="Focus", confidence=0.85)
    """

    def __init__(self, config: ComponentConfig | None = None):
        """Initialize state display.

        Args:
            config: Component configuration.
        """
        self.config = config or ComponentConfig(height=200)

        # State colors
        self.state_colors = {
            "Meditation": "#3498db",
            "Focus": "#2ecc71",
            "Flow": "#9b59b6",
            "Neutral": "#95a5a6",
            "Anxiety": "#e74c3c",
            "Relaxation": "#1abc9c",
        }

    def create_gauge(
        self,
        state: str,
        confidence: float,
        title: str = "Current State",
    ) -> Any:
        """Create confidence gauge for state.

        Args:
            state: Detected state name.
            confidence: Confidence score (0-1).
            title: Display title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        color = self.state_colors.get(state, "#95a5a6")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            title={"text": f"{title}<br><b>{state}</b>"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#f8f9fa"},
                    {"range": [30, 70], "color": "#e9ecef"},
                    {"range": [70, 100], "color": "#dee2e6"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80
                }
            }
        ))

        fig.update_layout(
            height=self.config.height,
            template=self.config.theme,
        )

        return fig

    def create_state_comparison(
        self,
        state_probabilities: dict[str, float],
        title: str = "State Probabilities",
    ) -> Any:
        """Create bar chart comparing state probabilities.

        Args:
            state_probabilities: Dict mapping states to probabilities.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        states = list(state_probabilities.keys())
        probs = list(state_probabilities.values())
        colors = [self.state_colors.get(s, "#95a5a6") for s in states]

        fig = go.Figure(go.Bar(
            x=states,
            y=probs,
            marker_color=colors,
            text=[f"{p:.0%}" for p in probs],
            textposition="outside",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="State",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=self.config.height,
            template=self.config.theme,
        )

        return fig


class SignalQualityMeter:
    """Signal quality indicator component.

    Displays quality metrics for each channel with
    color-coded status indicators.

    Example:
        >>> meter = SignalQualityMeter(channel_names=["TP9", "AF7"])
        >>> fig = meter.create_figure(quality_scores)
    """

    def __init__(
        self,
        channel_names: list[str],
        config: ComponentConfig | None = None,
    ):
        """Initialize quality meter.

        Args:
            channel_names: Names of channels.
            config: Component configuration.
        """
        self.channel_names = channel_names
        self.config = config or ComponentConfig(height=150)

    def create_figure(
        self,
        quality_scores: dict[str, float] | np.ndarray,
        title: str = "Signal Quality",
    ) -> Any:
        """Create quality meter figure.

        Args:
            quality_scores: Quality score per channel (0-1).
            title: Display title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        if isinstance(quality_scores, np.ndarray):
            quality_scores = {
                ch: q for ch, q in zip(self.channel_names, quality_scores)
            }

        channels = list(quality_scores.keys())
        scores = [quality_scores[ch] * 100 for ch in channels]

        # Color based on quality
        colors = []
        for score in scores:
            if score >= 80:
                colors.append("#28a745")  # Green
            elif score >= 60:
                colors.append("#ffc107")  # Yellow
            else:
                colors.append("#dc3545")  # Red

        fig = go.Figure(go.Bar(
            x=channels,
            y=scores,
            marker_color=colors,
            text=[f"{s:.0f}%" for s in scores],
            textposition="outside",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Channel",
            yaxis_title="Quality (%)",
            yaxis=dict(range=[0, 110]),
            height=self.config.height,
            template=self.config.theme,
        )

        return fig


class FeatureVisualization:
    """Visualize extracted features.

    Display band powers, feature importance, and other metrics.

    Example:
        >>> viz = FeatureVisualization()
        >>> fig = viz.create_band_power_chart(band_powers)
    """

    def __init__(self, config: ComponentConfig | None = None):
        """Initialize feature visualization.

        Args:
            config: Component configuration.
        """
        self.config = config or ComponentConfig()

        # Band colors
        self.band_colors = {
            "delta": "#1f77b4",
            "theta": "#ff7f0e",
            "alpha": "#2ca02c",
            "beta": "#d62728",
            "gamma": "#9467bd",
        }

    def create_band_power_chart(
        self,
        band_powers: dict[str, float],
        title: str = "Band Power Distribution",
    ) -> Any:
        """Create band power pie/bar chart.

        Args:
            band_powers: Power per frequency band.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        bands = list(band_powers.keys())
        powers = list(band_powers.values())
        colors = [self.band_colors.get(b.lower(), "#95a5a6") for b in bands]

        fig = go.Figure(go.Pie(
            labels=bands,
            values=powers,
            marker_colors=colors,
            hole=0.4,
            textinfo="label+percent",
        ))

        fig.update_layout(
            title=title,
            height=self.config.height,
            template=self.config.theme,
        )

        return fig

    def create_feature_radar(
        self,
        features: dict[str, float],
        title: str = "Feature Profile",
    ) -> Any:
        """Create radar chart of features.

        Args:
            features: Feature name to value mapping.
            title: Chart title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        names = list(features.keys())
        values = list(features.values())

        # Close the radar
        names = names + [names[0]]
        values = values + [values[0]]

        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=names,
            fill='toself',
            name='Features'
        ))

        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(range=[0, 1])
            ),
            height=self.config.height,
            template=self.config.theme,
        )

        return fig


class MarkerTimeline:
    """Event marker timeline component.

    Displays session events and markers on a timeline.

    Example:
        >>> timeline = MarkerTimeline()
        >>> fig = timeline.create_figure(markers, duration)
    """

    def __init__(self, config: ComponentConfig | None = None):
        """Initialize marker timeline.

        Args:
            config: Component configuration.
        """
        self.config = config or ComponentConfig(height=200)

    def create_figure(
        self,
        markers: list[dict[str, Any]],
        duration: float,
        title: str = "Event Timeline",
    ) -> Any:
        """Create marker timeline figure.

        Args:
            markers: List of marker dicts with 'time' and 'label'.
            duration: Total duration in seconds.
            title: Timeline title.

        Returns:
            Plotly figure.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")

        fig = go.Figure()

        # Timeline base
        fig.add_trace(go.Scatter(
            x=[0, duration],
            y=[0, 0],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
        ))

        # Markers
        for marker in markers:
            time = marker.get('time', 0)
            label = marker.get('label', 'Event')

            fig.add_trace(go.Scatter(
                x=[time],
                y=[0],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=[label],
                textposition='top center',
                showlegend=False,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis=dict(visible=False),
            height=self.config.height,
            template=self.config.theme,
        )

        return fig
