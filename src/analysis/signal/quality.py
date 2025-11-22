"""Signal quality assessment for EEG data.

Provides metrics and tools for evaluating EEG signal quality:
- Signal-to-noise ratio
- Flatline detection
- Impedance estimation
- Overall quality scores
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal, stats


@dataclass
class QualityReport:
    """Quality assessment report for a channel or session."""

    snr: float
    flatline_ratio: float
    variance: float
    kurtosis: float
    quality_score: float
    quality_label: str


def compute_snr(
    data: np.ndarray,
    fs: float,
    signal_band: tuple[float, float] = (1, 40),
    noise_band: tuple[float, float] = (50, 100),
) -> float:
    """Compute signal-to-noise ratio.

    SNR is computed as the ratio of power in the signal band
    to power in the noise band.

    Args:
        data: Signal array (samples,).
        fs: Sampling frequency in Hz.
        signal_band: Frequency range for signal (Hz).
        noise_band: Frequency range for noise (Hz).

    Returns:
        SNR in dB.
    """
    # Compute PSD
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))

    # Get power in each band
    signal_idx = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    noise_idx = (freqs >= noise_band[0]) & (freqs <= min(noise_band[1], fs / 2))

    signal_power = np.mean(psd[signal_idx]) if np.any(signal_idx) else 0
    noise_power = np.mean(psd[noise_idx]) if np.any(noise_idx) else 1e-10

    # Compute SNR in dB
    snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))

    return snr


def compute_flatline_ratio(
    data: np.ndarray,
    threshold: float = 0.1,
    window_size: int = 10,
) -> float:
    """Compute ratio of flatline (near-zero variance) segments.

    Args:
        data: Signal array (samples,).
        threshold: Variance threshold for flatline detection.
        window_size: Window size for variance computation.

    Returns:
        Ratio of flatline samples (0 to 1).
    """
    if len(data) < window_size:
        return 0.0

    # Compute rolling variance
    n_windows = len(data) - window_size + 1
    flatline_count = 0

    for i in range(0, n_windows, window_size):
        window = data[i:i + window_size]
        if np.var(window) < threshold:
            flatline_count += 1

    return flatline_count / (n_windows / window_size)


def compute_line_noise(
    data: np.ndarray,
    fs: float,
    line_freq: float = 60.0,
    bandwidth: float = 2.0,
) -> float:
    """Compute power at line noise frequency.

    Args:
        data: Signal array (samples,).
        fs: Sampling frequency in Hz.
        line_freq: Line noise frequency (50 or 60 Hz).
        bandwidth: Bandwidth around line frequency.

    Returns:
        Power at line frequency (relative to total).
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, len(data)))

    # Get power at line frequency
    line_idx = (freqs >= line_freq - bandwidth) & (freqs <= line_freq + bandwidth)
    total_power = np.sum(psd)
    line_power = np.sum(psd[line_idx])

    return line_power / max(total_power, 1e-10)


class SignalQualityAssessor:
    """Comprehensive signal quality assessment.

    Evaluates multiple quality metrics and provides
    an overall quality score and label.

    Example:
        >>> assessor = SignalQualityAssessor(fs=256)
        >>> report = assessor.assess_channel(eeg_channel)
        >>> print(report.quality_label)  # "Good"
    """

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "snr_good": 10,
        "snr_fair": 5,
        "flatline_good": 0.05,
        "flatline_fair": 0.15,
        "kurtosis_good": 5,
        "kurtosis_fair": 10,
    }

    def __init__(
        self,
        fs: float,
        line_freq: float = 60.0,
    ):
        """Initialize quality assessor.

        Args:
            fs: Sampling frequency in Hz.
            line_freq: Line noise frequency (50 or 60 Hz).
        """
        self.fs = fs
        self.line_freq = line_freq

    def assess_channel(self, data: np.ndarray) -> QualityReport:
        """Assess quality of a single channel.

        Args:
            data: Channel data (samples,).

        Returns:
            Quality report.
        """
        # Compute metrics
        snr = compute_snr(data, self.fs)
        flatline = compute_flatline_ratio(data)
        variance = np.var(data)
        kurt = stats.kurtosis(data)

        # Compute quality score (0-100)
        score = self._compute_score(snr, flatline, kurt)

        # Determine quality label
        label = self._get_label(score)

        return QualityReport(
            snr=snr,
            flatline_ratio=flatline,
            variance=variance,
            kurtosis=kurt,
            quality_score=score,
            quality_label=label,
        )

    def assess_session(
        self,
        data: np.ndarray,
        channel_names: list[str] | None = None,
    ) -> dict[str, QualityReport]:
        """Assess quality of all channels in a session.

        Args:
            data: EEG data (channels, samples).
            channel_names: Optional channel names.

        Returns:
            Dictionary mapping channel names to reports.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if channel_names is None:
            channel_names = [f"CH{i+1}" for i in range(data.shape[0])]

        reports = {}
        for i, name in enumerate(channel_names):
            reports[name] = self.assess_channel(data[i])

        return reports

    def get_overall_quality(
        self,
        data: np.ndarray,
    ) -> tuple[float, str]:
        """Get overall quality score for all channels.

        Args:
            data: EEG data (channels, samples).

        Returns:
            Tuple of (score, label).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        scores = []
        for i in range(data.shape[0]):
            report = self.assess_channel(data[i])
            scores.append(report.quality_score)

        avg_score = np.mean(scores)
        label = self._get_label(avg_score)

        return avg_score, label

    def _compute_score(
        self,
        snr: float,
        flatline: float,
        kurtosis: float,
    ) -> float:
        """Compute quality score from metrics.

        Args:
            snr: Signal-to-noise ratio.
            flatline: Flatline ratio.
            kurtosis: Kurtosis value.

        Returns:
            Quality score (0-100).
        """
        # SNR contribution (0-40 points)
        if snr >= self.QUALITY_THRESHOLDS["snr_good"]:
            snr_score = 40
        elif snr >= self.QUALITY_THRESHOLDS["snr_fair"]:
            snr_score = 25
        elif snr >= 0:
            snr_score = 10
        else:
            snr_score = 0

        # Flatline contribution (0-30 points)
        if flatline <= self.QUALITY_THRESHOLDS["flatline_good"]:
            flatline_score = 30
        elif flatline <= self.QUALITY_THRESHOLDS["flatline_fair"]:
            flatline_score = 15
        else:
            flatline_score = 0

        # Kurtosis contribution (0-30 points)
        # High kurtosis indicates artifacts
        abs_kurt = abs(kurtosis)
        if abs_kurt <= self.QUALITY_THRESHOLDS["kurtosis_good"]:
            kurt_score = 30
        elif abs_kurt <= self.QUALITY_THRESHOLDS["kurtosis_fair"]:
            kurt_score = 15
        else:
            kurt_score = 0

        return snr_score + flatline_score + kurt_score

    def _get_label(self, score: float) -> str:
        """Get quality label from score.

        Args:
            score: Quality score (0-100).

        Returns:
            Quality label.
        """
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Bad"

    def generate_report(
        self,
        data: np.ndarray,
        channel_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive quality report.

        Args:
            data: EEG data (channels, samples).
            channel_names: Optional channel names.

        Returns:
            Report dictionary.
        """
        channel_reports = self.assess_session(data, channel_names)
        overall_score, overall_label = self.get_overall_quality(data)

        return {
            "overall_score": overall_score,
            "overall_label": overall_label,
            "channels": {
                name: {
                    "snr": report.snr,
                    "flatline_ratio": report.flatline_ratio,
                    "variance": report.variance,
                    "kurtosis": report.kurtosis,
                    "quality_score": report.quality_score,
                    "quality_label": report.quality_label,
                }
                for name, report in channel_reports.items()
            },
            "recommendations": self._get_recommendations(channel_reports),
        }

    def _get_recommendations(
        self,
        reports: dict[str, QualityReport],
    ) -> list[str]:
        """Generate recommendations based on quality reports.

        Args:
            reports: Channel quality reports.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        for name, report in reports.items():
            if report.snr < self.QUALITY_THRESHOLDS["snr_fair"]:
                recommendations.append(
                    f"{name}: Low SNR ({report.snr:.1f} dB). "
                    "Check electrode contact."
                )

            if report.flatline_ratio > self.QUALITY_THRESHOLDS["flatline_fair"]:
                recommendations.append(
                    f"{name}: High flatline ratio ({report.flatline_ratio:.1%}). "
                    "Possible disconnection."
                )

            if abs(report.kurtosis) > self.QUALITY_THRESHOLDS["kurtosis_fair"]:
                recommendations.append(
                    f"{name}: High kurtosis ({report.kurtosis:.1f}). "
                    "Possible artifacts."
                )

        if not recommendations:
            recommendations.append("All channels show good quality.")

        return recommendations
