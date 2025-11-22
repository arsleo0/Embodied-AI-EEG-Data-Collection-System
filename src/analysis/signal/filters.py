"""Digital filters for EEG signal processing.

Provides bandpass, notch, highpass, and lowpass filters using
Butterworth filter design.
"""

from typing import Any

import numpy as np
from scipy import signal


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter to signal.

    Args:
        data: Input signal (samples,) or (channels, samples).
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal with same shape as input.
    """
    nyq = fs / 2
    low = low_freq / nyq
    high = high_freq / nyq

    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    b, a = signal.butter(order, [low, high], btype="band")

    # Handle 1D and 2D arrays
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in data])


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply notch filter to remove power line noise.

    Args:
        data: Input signal (samples,) or (channels, samples).
        notch_freq: Frequency to remove in Hz (e.g., 50 or 60).
        fs: Sampling frequency in Hz.
        quality_factor: Quality factor (higher = narrower notch).

    Returns:
        Filtered signal with same shape as input.
    """
    nyq = fs / 2
    freq = notch_freq / nyq

    b, a = signal.iirnotch(freq, quality_factor)

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in data])


def highpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply highpass filter to signal.

    Args:
        data: Input signal (samples,) or (channels, samples).
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal with same shape as input.
    """
    nyq = fs / 2
    freq = cutoff / nyq
    freq = max(0.001, min(freq, 0.999))

    b, a = signal.butter(order, freq, btype="high")

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in data])


def lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply lowpass filter to signal.

    Args:
        data: Input signal (samples,) or (channels, samples).
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal with same shape as input.
    """
    nyq = fs / 2
    freq = cutoff / nyq
    freq = max(0.001, min(freq, 0.999))

    b, a = signal.butter(order, freq, btype="low")

    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in data])


class FilterBank:
    """Configurable filter bank for EEG preprocessing.

    Applies a sequence of filters to EEG data.

    Example:
        >>> fb = FilterBank(fs=256)
        >>> fb.add_notch(60)
        >>> fb.add_bandpass(1, 50)
        >>> filtered = fb.apply(raw_eeg)
    """

    def __init__(self, fs: float):
        """Initialize filter bank.

        Args:
            fs: Sampling frequency in Hz.
        """
        self.fs = fs
        self._filters: list[dict[str, Any]] = []

    def add_bandpass(
        self,
        low_freq: float,
        high_freq: float,
        order: int = 4,
    ) -> "FilterBank":
        """Add bandpass filter to the bank.

        Args:
            low_freq: Low cutoff frequency.
            high_freq: High cutoff frequency.
            order: Filter order.

        Returns:
            Self for chaining.
        """
        self._filters.append({
            "type": "bandpass",
            "low_freq": low_freq,
            "high_freq": high_freq,
            "order": order,
        })
        return self

    def add_notch(
        self,
        freq: float,
        quality_factor: float = 30.0,
    ) -> "FilterBank":
        """Add notch filter to the bank.

        Args:
            freq: Frequency to notch out.
            quality_factor: Quality factor.

        Returns:
            Self for chaining.
        """
        self._filters.append({
            "type": "notch",
            "freq": freq,
            "quality_factor": quality_factor,
        })
        return self

    def add_highpass(self, cutoff: float, order: int = 4) -> "FilterBank":
        """Add highpass filter to the bank.

        Args:
            cutoff: Cutoff frequency.
            order: Filter order.

        Returns:
            Self for chaining.
        """
        self._filters.append({
            "type": "highpass",
            "cutoff": cutoff,
            "order": order,
        })
        return self

    def add_lowpass(self, cutoff: float, order: int = 4) -> "FilterBank":
        """Add lowpass filter to the bank.

        Args:
            cutoff: Cutoff frequency.
            order: Filter order.

        Returns:
            Self for chaining.
        """
        self._filters.append({
            "type": "lowpass",
            "cutoff": cutoff,
            "order": order,
        })
        return self

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply all filters in sequence.

        Args:
            data: Input signal.

        Returns:
            Filtered signal.
        """
        result = data.copy()

        for filt in self._filters:
            if filt["type"] == "bandpass":
                result = bandpass_filter(
                    result,
                    filt["low_freq"],
                    filt["high_freq"],
                    self.fs,
                    filt["order"],
                )
            elif filt["type"] == "notch":
                result = notch_filter(
                    result,
                    filt["freq"],
                    self.fs,
                    filt["quality_factor"],
                )
            elif filt["type"] == "highpass":
                result = highpass_filter(
                    result,
                    filt["cutoff"],
                    self.fs,
                    filt["order"],
                )
            elif filt["type"] == "lowpass":
                result = lowpass_filter(
                    result,
                    filt["cutoff"],
                    self.fs,
                    filt["order"],
                )

        return result

    def clear(self) -> "FilterBank":
        """Clear all filters from the bank.

        Returns:
            Self for chaining.
        """
        self._filters = []
        return self

    def __repr__(self) -> str:
        return f"FilterBank(fs={self.fs}, filters={len(self._filters)})"
