"""EEG device handling module."""

from .connection import EEGConnection, test_connection

__all__ = ["EEGConnection", "test_connection"]
