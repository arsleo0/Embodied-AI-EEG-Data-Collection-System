"""EEG device implementations."""

from .base import BaseEEGDevice
from .muse import MuseDevice

__all__ = ["BaseEEGDevice", "MuseDevice"]
