"""TTS backend abstraction — ready for voice cloning later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSBackend(ABC):
    """Synthesize speech for a text segment.

    ``reference_audio_path`` is reserved for future voice-cloning backends
    (e.g. XTTS) that clone the speaker from the original track.
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        out_path: Path,
        *,
        reference_audio_path: Path | None = None,
    ) -> Path:
        """Write synthesized audio to ``out_path`` and return it."""

    def close(self) -> None:
        """Optional cleanup."""
