"""Qualia synthesis for consciousness research.

This module provides tools for cross-modal consciousness synthesis:
- Multimodal embedding fusion
- Synesthesia simulation
- EEG to sensory mapping
"""

from .synthesizer import (
    QualiaSynthesizer,
    CrossModalFusion,
    SynesthesiaSimulator,
    EmotionalStateGenerator,
)
from .modality_mapper import (
    ModalityMapper,
    EEGToAudioMapper,
    EEGToVisualMapper,
    EEGToTextMapper,
    CrossModalTranslator,
)

__all__ = [
    # Synthesizer
    "QualiaSynthesizer",
    "CrossModalFusion",
    "SynesthesiaSimulator",
    "EmotionalStateGenerator",
    # Modality Mapper
    "ModalityMapper",
    "EEGToAudioMapper",
    "EEGToVisualMapper",
    "EEGToTextMapper",
    "CrossModalTranslator",
]
