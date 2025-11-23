"""Live monitoring and streaming for real-time dashboard.

Handles real-time data streaming, quality monitoring, and alerts.
"""

from dataclasses import dataclass
from typing import Any, Callable
import threading
import queue
import time
import numpy as np


@dataclass
class SignalQualityAlert:
    """Alert for signal quality issues.

    Attributes:
        channel: Affected channel name.
        quality: Quality score (0-1).
        issue: Description of the issue.
        severity: Alert severity (info, warning, error).
        timestamp: Time of alert.
    """

    channel: str
    quality: float
    issue: str
    severity: str
    timestamp: float


class StreamingHandler:
    """Handle real-time data streaming.

    Manages data buffers and provides streaming interface
    for dashboard components.

    Example:
        >>> handler = StreamingHandler(buffer_size=5.0, fs=256)
        >>> handler.start()
        >>> handler.add_data(new_samples)
        >>> data = handler.get_buffer()
    """

    def __init__(
        self,
        buffer_size: float = 5.0,
        fs: float = 256.0,
        n_channels: int = 4,
    ):
        """Initialize streaming handler.

        Args:
            buffer_size: Buffer duration in seconds.
            fs: Sampling frequency.
            n_channels: Number of channels.
        """
        self.buffer_size = buffer_size
        self.fs = fs
        self.n_channels = n_channels

        # Calculate buffer samples
        self.buffer_samples = int(buffer_size * fs)

        # Initialize buffers
        self._data_buffer = np.zeros((n_channels, self.buffer_samples))
        self._timestamps = np.zeros(self.buffer_samples)

        # Thread safety
        self._lock = threading.Lock()
        self._running = False

        # Write position
        self._write_pos = 0

    def start(self) -> None:
        """Start streaming handler."""
        self._running = True

    def stop(self) -> None:
        """Stop streaming handler."""
        self._running = False

    def add_data(
        self,
        data: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> None:
        """Add new data to buffer.

        Args:
            data: New samples (channels, samples).
            timestamps: Optional timestamps.
        """
        if not self._running:
            return

        n_samples = data.shape[1]

        with self._lock:
            # Handle wrap-around
            if self._write_pos + n_samples <= self.buffer_samples:
                self._data_buffer[:, self._write_pos:self._write_pos + n_samples] = data
                if timestamps is not None:
                    self._timestamps[self._write_pos:self._write_pos + n_samples] = timestamps
                self._write_pos += n_samples
            else:
                # Wrap around
                first_part = self.buffer_samples - self._write_pos
                self._data_buffer[:, self._write_pos:] = data[:, :first_part]
                self._data_buffer[:, :n_samples - first_part] = data[:, first_part:]

                if timestamps is not None:
                    self._timestamps[self._write_pos:] = timestamps[:first_part]
                    self._timestamps[:n_samples - first_part] = timestamps[first_part:]

                self._write_pos = n_samples - first_part

    def get_buffer(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current buffer contents.

        Returns:
            Tuple of (data, timestamps).
        """
        with self._lock:
            # Return data in correct order
            if self._write_pos == 0:
                return self._data_buffer.copy(), self._timestamps.copy()

            # Reorder to put oldest data first
            data = np.concatenate([
                self._data_buffer[:, self._write_pos:],
                self._data_buffer[:, :self._write_pos]
            ], axis=1)

            timestamps = np.concatenate([
                self._timestamps[self._write_pos:],
                self._timestamps[:self._write_pos]
            ])

            return data, timestamps

    def get_latest(self, n_samples: int) -> np.ndarray:
        """Get latest n samples.

        Args:
            n_samples: Number of samples to get.

        Returns:
            Latest data (channels, samples).
        """
        with self._lock:
            if n_samples >= self.buffer_samples:
                return self._data_buffer.copy()

            start = (self._write_pos - n_samples) % self.buffer_samples
            if start + n_samples <= self.buffer_samples:
                return self._data_buffer[:, start:start + n_samples].copy()
            else:
                return np.concatenate([
                    self._data_buffer[:, start:],
                    self._data_buffer[:, :n_samples - (self.buffer_samples - start)]
                ], axis=1)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._data_buffer.fill(0)
            self._timestamps.fill(0)
            self._write_pos = 0


class LiveMonitor:
    """Real-time monitoring of EEG signals and quality.

    Monitors signal quality, detects artifacts, and generates
    alerts for poor signal conditions.

    Example:
        >>> monitor = LiveMonitor(fs=256, channel_names=["TP9", "AF7"])
        >>> monitor.start()
        >>> quality = monitor.get_quality()
        >>> alerts = monitor.get_alerts()
    """

    def __init__(
        self,
        fs: float = 256.0,
        channel_names: list[str] | None = None,
        quality_threshold: float = 0.6,
        alert_callback: Callable[[SignalQualityAlert], None] | None = None,
    ):
        """Initialize live monitor.

        Args:
            fs: Sampling frequency.
            channel_names: Names of channels.
            quality_threshold: Quality threshold for alerts.
            alert_callback: Callback for quality alerts.
        """
        self.fs = fs
        self.channel_names = channel_names or ["Ch0", "Ch1", "Ch2", "Ch3"]
        self.n_channels = len(self.channel_names)
        self.quality_threshold = quality_threshold
        self.alert_callback = alert_callback

        # Quality tracking
        self._quality_scores = np.ones(self.n_channels)
        self._artifact_counts = np.zeros(self.n_channels)

        # Alerts queue
        self._alerts: queue.Queue = queue.Queue()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None

        # Data queue for processing
        self._data_queue: queue.Queue = queue.Queue()

    def start(self) -> None:
        """Start the monitor."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def add_data(self, data: np.ndarray) -> None:
        """Add data for quality monitoring.

        Args:
            data: EEG data (channels, samples).
        """
        self._data_queue.put(data)

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Get data with timeout
                data = self._data_queue.get(timeout=0.1)
                self._analyze_quality(data)
            except queue.Empty:
                continue

    def _analyze_quality(self, data: np.ndarray) -> None:
        """Analyze signal quality.

        Args:
            data: EEG data to analyze.
        """
        for i in range(min(self.n_channels, data.shape[0])):
            channel_data = data[i]

            # Quality metrics
            quality = self._compute_channel_quality(channel_data)
            self._quality_scores[i] = quality

            # Check for alerts
            if quality < self.quality_threshold:
                issue = self._identify_issue(channel_data, quality)
                severity = "warning" if quality > 0.3 else "error"

                alert = SignalQualityAlert(
                    channel=self.channel_names[i],
                    quality=quality,
                    issue=issue,
                    severity=severity,
                    timestamp=time.time(),
                )

                self._alerts.put(alert)

                if self.alert_callback:
                    self.alert_callback(alert)

    def _compute_channel_quality(self, data: np.ndarray) -> float:
        """Compute quality score for a channel.

        Args:
            data: Single channel data.

        Returns:
            Quality score (0-1).
        """
        # Multiple quality indicators

        # 1. Check for flat line (disconnected)
        std = np.std(data)
        if std < 0.5:
            return 0.0

        # 2. Check for saturation
        max_amp = np.max(np.abs(data))
        saturation_score = 1.0 if max_amp < 100 else max(0, 1 - (max_amp - 100) / 100)

        # 3. Check for high-frequency noise
        # Simple: ratio of high-frequency to total power
        fft = np.fft.rfft(data)
        power = np.abs(fft) ** 2
        total_power = np.sum(power)
        if total_power == 0:
            return 0.0

        # High freq = above 50 Hz
        freq_res = self.fs / len(data)
        high_freq_idx = int(50 / freq_res)
        high_freq_power = np.sum(power[high_freq_idx:])
        noise_ratio = high_freq_power / total_power
        noise_score = max(0, 1 - noise_ratio * 2)

        # 4. Check for artifacts (simple threshold)
        artifact_ratio = np.sum(np.abs(data) > 80) / len(data)
        artifact_score = max(0, 1 - artifact_ratio * 5)

        # Combined quality
        quality = (saturation_score + noise_score + artifact_score) / 3

        return float(np.clip(quality, 0, 1))

    def _identify_issue(self, data: np.ndarray, quality: float) -> str:
        """Identify the quality issue.

        Args:
            data: Channel data.
            quality: Quality score.

        Returns:
            Issue description.
        """
        std = np.std(data)
        max_amp = np.max(np.abs(data))

        if std < 0.5:
            return "Signal disconnected or flat"
        elif max_amp > 100:
            return "Signal saturation detected"
        elif quality < 0.5:
            return "High noise or artifacts"
        else:
            return "Marginal signal quality"

    def get_quality(self) -> dict[str, float]:
        """Get current quality scores.

        Returns:
            Dict mapping channel names to quality scores.
        """
        return {
            name: float(score)
            for name, score in zip(self.channel_names, self._quality_scores)
        }

    def get_alerts(self, max_alerts: int = 10) -> list[SignalQualityAlert]:
        """Get recent alerts.

        Args:
            max_alerts: Maximum alerts to return.

        Returns:
            List of recent alerts.
        """
        alerts = []
        try:
            while len(alerts) < max_alerts:
                alert = self._alerts.get_nowait()
                alerts.append(alert)
        except queue.Empty:
            pass
        return alerts

    def clear_alerts(self) -> None:
        """Clear all pending alerts."""
        while not self._alerts.empty():
            try:
                self._alerts.get_nowait()
            except queue.Empty:
                break

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary.

        Returns:
            Summary statistics.
        """
        quality = self.get_quality()
        mean_quality = np.mean(list(quality.values()))

        return {
            "channel_quality": quality,
            "mean_quality": float(mean_quality),
            "status": "good" if mean_quality > 0.8 else (
                "fair" if mean_quality > 0.6 else "poor"
            ),
            "n_channels": self.n_channels,
        }


def create_quality_display(
    quality_scores: dict[str, float],
) -> str:
    """Create text display of quality scores.

    Args:
        quality_scores: Channel quality scores.

    Returns:
        Formatted string display.
    """
    lines = ["Signal Quality:"]

    for channel, quality in quality_scores.items():
        # Status indicator
        if quality >= 0.8:
            status = "✓"
            color = "green"
        elif quality >= 0.6:
            status = "!"
            color = "yellow"
        else:
            status = "✗"
            color = "red"

        lines.append(f"  {status} {channel}: {quality:.0%}")

    return "\n".join(lines)
