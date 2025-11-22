"""Audio recording for voice notes."""

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from ..data.models import AudioSegment


class AudioRecorder:
    """Audio recorder for voice notes.

    Phase 1: Basic recording functionality.
    Phase 2: Whisper transcription integration.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize audio recorder.

        Args:
            config: Audio configuration.
        """
        if not AUDIO_AVAILABLE:
            raise ImportError(
                "sounddevice and soundfile required. "
                "Install with: pip install sounddevice soundfile"
            )

        self.config = config or {}
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)
        self.device = self.config.get("device")

        self._is_recording = False
        self._frames: list[np.ndarray] = []
        self._stream = None

    def start_recording(self) -> None:
        """Start audio recording.

        Raises:
            RuntimeError: If already recording.
        """
        if self._is_recording:
            raise RuntimeError("Already recording")

        self._frames = []

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            device=self.device,
        )
        self._stream.start()
        self._is_recording = True

    def stop_recording(self) -> np.ndarray:
        """Stop audio recording.

        Returns:
            Recorded audio data.
        """
        if not self._is_recording:
            return np.array([])

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._is_recording = False

        if not self._frames:
            return np.array([])

        return np.concatenate(self._frames, axis=0)

    def save_recording(
        self,
        output_path: str | Path,
        audio_data: np.ndarray | None = None,
    ) -> AudioSegment:
        """Save recorded audio to file.

        Args:
            output_path: Output file path.
            audio_data: Audio data to save. Uses current recording if None.

        Returns:
            AudioSegment with file info.
        """
        if audio_data is None:
            audio_data = self.stop_recording()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save audio file
        sf.write(output_path, audio_data, self.sample_rate)

        # Calculate duration
        duration = len(audio_data) / self.sample_rate

        return AudioSegment(
            timestamp=time.time(),
            duration=duration,
            file_path=str(output_path),
            transcript=None,  # Will be populated by transcription
        )

    def record_note(
        self,
        duration: float,
        output_path: str | Path,
    ) -> AudioSegment:
        """Record a voice note of fixed duration.

        Args:
            duration: Recording duration in seconds.
            output_path: Output file path.

        Returns:
            AudioSegment with file info.
        """
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device,
        )
        sd.wait()

        # Save to file
        return self.save_recording(output_path, audio_data)

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def list_devices(self) -> list[dict[str, Any]]:
        """List available audio devices.

        Returns:
            List of device info dictionaries.
        """
        devices = sd.query_devices()
        return [
            {
                "id": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
            }
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._is_recording:
            self.stop_recording()
        return False


# Placeholder for Whisper transcription
class WhisperTranscriber:
    """Whisper-based audio transcription.

    To be implemented in Phase 2.
    """

    def __init__(self, model: str = "base"):
        """Initialize Whisper transcriber.

        Args:
            model: Whisper model size (tiny, base, small, medium, large).
        """
        self.model = model
        # TODO: Load Whisper model

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcription text.

        TODO: Implement in Phase 2.
        """
        return ""
