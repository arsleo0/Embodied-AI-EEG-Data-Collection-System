"""Interactive plotting for EEG data visualization.

Provides interactive, zoomable plots for time series,
spectrograms, topographic maps, and band power.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PlotConfig:
    """Configuration for interactive plots.

    Attributes:
        width: Plot width in pixels.
        height: Plot height in pixels.
        colormap: Default colormap.
        template: Plotly template.
    """

    width: int = 800
    height: int = 400
    colormap: str = "viridis"
    template: str = "plotly_white"


class InteractivePlotter:
    """Base class for interactive plots.

    Provides common functionality for creating
    interactive visualizations with Plotly.
    """

    def __init__(self, config: PlotConfig | None = None):
        """Initialize plotter.

        Args:
            config: Plot configuration.
        """
        self.config = config or PlotConfig()

    def _get_plotly(self):
        """Import and return plotly modules."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            return go, px
        except ImportError:
            raise ImportError("Plotly required: pip install plotly")


class TimeSeriesPlot(InteractivePlotter):
    """Interactive time series plot for EEG signals.

    Provides zoomable, scrollable multi-channel
    EEG visualization.

    Example:
        >>> plotter = TimeSeriesPlot()
        >>> fig = plotter.plot(data, fs=256, channels=['TP9', 'AF7'])
    """

    def plot(
        self,
        data: np.ndarray,
        fs: float = 256.0,
        channels: list[str] | None = None,
        title: str = "EEG Time Series",
        show_events: list[dict] | None = None,
    ) -> Any:
        """Create time series plot.

        Args:
            data: EEG data (channels, samples).
            fs: Sampling frequency.
            channels: Channel names.
            title: Plot title.
            show_events: Event markers to display.

        Returns:
            Plotly figure.
        """
        go, _ = self._get_plotly()

        n_channels, n_samples = data.shape
        time = np.arange(n_samples) / fs

        if channels is None:
            channels = [f"Ch {i+1}" for i in range(n_channels)]

        fig = go.Figure()

        # Add each channel with offset
        spacing = np.std(data) * 4
        for i, ch_name in enumerate(channels):
            offset = i * spacing
            fig.add_trace(go.Scatter(
                x=time,
                y=data[i] + offset,
                mode='lines',
                name=ch_name,
                line=dict(width=1),
                hovertemplate=(
                    f"<b>{ch_name}</b><br>"
                    "Time: %{x:.3f}s<br>"
                    "Value: %{customdata:.2f}μV"
                    "<extra></extra>"
                ),
                customdata=data[i],
            ))

        # Add event markers
        if show_events:
            for event in show_events:
                time_val = event.get('time', 0)
                label = event.get('label', 'Event')
                fig.add_vline(
                    x=time_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=label,
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (μV)",
            width=self.config.width,
            height=self.config.height * 1.5,
            template=self.config.template,
            hovermode='x unified',
            # Range slider for navigation
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear",
            ),
        )

        return fig


class SpectrogramPlot(InteractivePlotter):
    """Interactive spectrogram visualization.

    Creates time-frequency representations with
    interactive controls.

    Example:
        >>> plotter = SpectrogramPlot()
        >>> fig = plotter.plot(signal, fs=256)
    """

    def plot(
        self,
        data: np.ndarray,
        fs: float = 256.0,
        nperseg: int = 256,
        noverlap: int | None = None,
        title: str = "Spectrogram",
        freq_range: tuple[float, float] = (0, 50),
    ) -> Any:
        """Create spectrogram plot.

        Args:
            data: Signal data (1D array or single channel).
            fs: Sampling frequency.
            nperseg: Window size for FFT.
            noverlap: Overlap between windows.
            title: Plot title.
            freq_range: Frequency range to display.

        Returns:
            Plotly figure.
        """
        go, _ = self._get_plotly()
        from scipy.signal import spectrogram

        if data.ndim > 1:
            data = data[0]  # Use first channel

        if noverlap is None:
            noverlap = nperseg // 2

        # Compute spectrogram
        f, t, Sxx = spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)

        # Filter frequency range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=t,
            y=f,
            z=Sxx_db,
            colorscale=self.config.colormap,
            colorbar=dict(title="Power (dB)"),
            hovertemplate=(
                "Time: %{x:.2f}s<br>"
                "Freq: %{y:.1f}Hz<br>"
                "Power: %{z:.1f}dB"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
        )

        return fig


class TopoPlot(InteractivePlotter):
    """Interactive topographic map visualization.

    Creates scalp topography maps for EEG data.

    Example:
        >>> plotter = TopoPlot()
        >>> fig = plotter.plot(values, channels)
    """

    # Standard 10-20 electrode positions (normalized)
    ELECTRODE_POSITIONS = {
        'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
        'F7': (-0.7, 0.5), 'F3': (-0.35, 0.5), 'Fz': (0, 0.5),
        'F4': (0.35, 0.5), 'F8': (0.7, 0.5),
        'T3': (-1.0, 0), 'C3': (-0.5, 0), 'Cz': (0, 0),
        'C4': (0.5, 0), 'T4': (1.0, 0),
        'T5': (-0.7, -0.5), 'P3': (-0.35, -0.5), 'Pz': (0, -0.5),
        'P4': (0.35, -0.5), 'T6': (0.7, -0.5),
        'O1': (-0.3, -0.9), 'O2': (0.3, -0.9),
        # Muse positions (approximate)
        'TP9': (-0.85, -0.25), 'TP10': (0.85, -0.25),
        'AF7': (-0.5, 0.7), 'AF8': (0.5, 0.7),
    }

    def plot(
        self,
        values: np.ndarray | dict[str, float],
        channels: list[str] | None = None,
        title: str = "Topographic Map",
        show_labels: bool = True,
    ) -> Any:
        """Create topographic map.

        Args:
            values: Values for each channel.
            channels: Channel names (if values is array).
            title: Plot title.
            show_labels: Whether to show channel labels.

        Returns:
            Plotly figure.
        """
        go, _ = self._get_plotly()

        # Process input
        if isinstance(values, dict):
            channels = list(values.keys())
            values = np.array(list(values.values()))
        elif channels is None:
            channels = ['TP9', 'AF7', 'AF8', 'TP10'][:len(values)]

        # Get positions
        x_pos = []
        y_pos = []
        valid_channels = []
        valid_values = []

        for i, ch in enumerate(channels):
            if ch in self.ELECTRODE_POSITIONS:
                pos = self.ELECTRODE_POSITIONS[ch]
                x_pos.append(pos[0])
                y_pos.append(pos[1])
                valid_channels.append(ch)
                valid_values.append(values[i])

        # Create interpolated surface
        fig = go.Figure()

        # Add head outline
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
        ))

        # Add nose
        fig.add_trace(go.Scatter(
            x=[0, -0.1, 0, 0.1, 0],
            y=[1, 1.1, 1.2, 1.1, 1],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
        ))

        # Add electrode markers
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=20,
                color=valid_values,
                colorscale=self.config.colormap,
                colorbar=dict(title="Value"),
                showscale=True,
            ),
            text=valid_channels if show_labels else None,
            textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Value: %{marker.color:.3f}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=title,
            width=self.config.height,  # Square
            height=self.config.height,
            template=self.config.template,
            xaxis=dict(visible=False, range=[-1.3, 1.3]),
            yaxis=dict(visible=False, range=[-1.3, 1.5], scaleanchor='x'),
        )

        return fig


class BandPowerPlot(InteractivePlotter):
    """Interactive band power visualization.

    Creates bar charts and radar plots for
    frequency band power analysis.

    Example:
        >>> plotter = BandPowerPlot()
        >>> fig = plotter.plot_bars(band_powers)
    """

    BAND_NAMES = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    BAND_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_bars(
        self,
        band_powers: np.ndarray | dict[str, float],
        title: str = "Band Power Distribution",
        normalize: bool = True,
    ) -> Any:
        """Create bar chart of band powers.

        Args:
            band_powers: Power values for each band.
            title: Plot title.
            normalize: Whether to normalize to percentages.

        Returns:
            Plotly figure.
        """
        go, _ = self._get_plotly()

        if isinstance(band_powers, dict):
            bands = list(band_powers.keys())
            values = np.array(list(band_powers.values()))
        else:
            bands = self.BAND_NAMES[:len(band_powers)]
            values = band_powers

        if normalize:
            values = values / np.sum(values) * 100
            y_title = "Relative Power (%)"
        else:
            y_title = "Power (μV²)"

        fig = go.Figure(data=go.Bar(
            x=bands,
            y=values,
            marker_color=self.BAND_COLORS[:len(bands)],
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{y_title}: %{{y:.1f}}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Frequency Band",
            yaxis_title=y_title,
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
        )

        return fig

    def plot_radar(
        self,
        band_powers: np.ndarray | dict[str, float],
        title: str = "Band Power Profile",
    ) -> Any:
        """Create radar chart of band powers.

        Args:
            band_powers: Power values for each band.
            title: Plot title.

        Returns:
            Plotly figure.
        """
        go, _ = self._get_plotly()

        if isinstance(band_powers, dict):
            bands = list(band_powers.keys())
            values = np.array(list(band_powers.values()))
        else:
            bands = self.BAND_NAMES[:len(band_powers)]
            values = band_powers

        # Normalize for radar
        values = values / np.max(values)

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=bands,
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line_color='rgb(31, 119, 180)',
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Normalized: %{r:.2f}"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                ),
            ),
            width=self.config.height,  # Square
            height=self.config.height,
            template=self.config.template,
        )

        return fig

    def plot_comparison(
        self,
        band_powers_list: list[np.ndarray | dict],
        labels: list[str],
        title: str = "Band Power Comparison",
    ) -> Any:
        """Create grouped bar chart comparing multiple measurements.

        Args:
            band_powers_list: List of band power arrays/dicts.
            labels: Labels for each measurement.
            title: Plot title.

        Returns:
            Plotly figure.
        """
        go, _ = self._get_plotly()

        fig = go.Figure()

        for i, (powers, label) in enumerate(zip(band_powers_list, labels)):
            if isinstance(powers, dict):
                bands = list(powers.keys())
                values = np.array(list(powers.values()))
            else:
                bands = self.BAND_NAMES[:len(powers)]
                values = powers

            # Normalize
            values = values / np.sum(values) * 100

            fig.add_trace(go.Bar(
                name=label,
                x=bands,
                y=values,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "%{x}: %{y:.1f}%"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Frequency Band",
            yaxis_title="Relative Power (%)",
            barmode='group',
            width=self.config.width,
            height=self.config.height,
            template=self.config.template,
        )

        return fig


def create_dashboard_layout(
    eeg_data: np.ndarray,
    fs: float = 256.0,
    channels: list[str] | None = None,
    title: str = "EEG Dashboard",
) -> Any:
    """Create a comprehensive EEG dashboard.

    Combines multiple plot types into a single
    interactive dashboard.

    Args:
        eeg_data: EEG data (channels, samples).
        fs: Sampling frequency.
        channels: Channel names.
        title: Dashboard title.

    Returns:
        Plotly figure with subplots.
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required: pip install plotly")

    if channels is None:
        channels = [f"Ch {i+1}" for i in range(eeg_data.shape[0])]

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Time Series",
            "Band Power",
            "Spectrogram",
            "Topography",
            "Signal Quality",
            "Statistics",
        ),
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    n_channels, n_samples = eeg_data.shape
    time = np.arange(n_samples) / fs

    # 1. Time series (top, full width)
    spacing = np.std(eeg_data) * 4
    for i, ch in enumerate(channels):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=eeg_data[i] + i * spacing,
                name=ch,
                line=dict(width=1),
            ),
            row=1, col=1
        )

    # 2. Band power (compute from data)
    from scipy.signal import welch
    freqs, psd = welch(eeg_data.mean(axis=0), fs, nperseg=256)

    # Define bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50),
    }

    band_powers = []
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        power = np.trapezoid(psd[mask], freqs[mask])
        band_powers.append(power)

    total_power = sum(band_powers)
    band_powers = [p / total_power * 100 for p in band_powers]

    fig.add_trace(
        go.Bar(
            x=list(bands.keys()),
            y=band_powers,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        ),
        row=2, col=1
    )

    # 3. Spectrogram
    from scipy.signal import spectrogram
    f, t, Sxx = spectrogram(eeg_data[0], fs, nperseg=128, noverlap=64)
    freq_mask = f <= 50
    Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)

    fig.add_trace(
        go.Heatmap(
            x=t,
            y=f[freq_mask],
            z=Sxx_db,
            colorscale='viridis',
            showscale=False,
        ),
        row=2, col=2
    )

    # 4. Simple quality indicator (variance-based)
    quality_scores = []
    for i in range(n_channels):
        var = np.var(eeg_data[i])
        # Simple quality heuristic
        score = min(100, max(0, 100 - abs(var - 500) / 10))
        quality_scores.append(score)

    fig.add_trace(
        go.Bar(
            x=channels,
            y=quality_scores,
            marker_color=['green' if s > 70 else 'orange' if s > 40 else 'red'
                         for s in quality_scores],
        ),
        row=3, col=1
    )

    # 5. Statistics
    stats = {
        'Mean': np.mean(eeg_data),
        'Std': np.std(eeg_data),
        'Min': np.min(eeg_data),
        'Max': np.max(eeg_data),
    }

    fig.add_trace(
        go.Bar(
            x=list(stats.keys()),
            y=list(stats.values()),
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=900,
        showlegend=False,
        template='plotly_white',
    )

    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Band", row=2, col=1)
    fig.update_yaxes(title_text="Power (%)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Freq (Hz)", row=2, col=2)
    fig.update_xaxes(title_text="Channel", row=3, col=1)
    fig.update_yaxes(title_text="Quality (%)", row=3, col=1)

    return fig
