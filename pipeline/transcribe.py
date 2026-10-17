"""Speech recognition with faster-whisper."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str
    translation: str = ""


_model_cache: dict[str, WhisperModel] = {}


def _get_model(model_size: str) -> WhisperModel:
    if model_size not in _model_cache:
        last_err: Exception | None = None
        for device, compute_type in (
            ("cuda", "float16"),
            ("cpu", "int8"),
        ):
            try:
                _model_cache[model_size] = WhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            raise RuntimeError(f"Не удалось загрузить Whisper ({model_size}): {last_err}")
    return _model_cache[model_size]


def load_audio_wav(path: Path) -> np.ndarray:
    """Load mono PCM WAV as float32 in [-1, 1]. Avoids PyAV (broken with some versions)."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # faster-whisper expects 16 kHz; resample if needed
    if rate != 16000:
        audio = _resample_linear(audio, rate, 16000)

    return audio


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    duration = len(audio) / src_rate
    n_dst = int(round(duration * dst_rate))
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    language: str | None = None,
) -> list[Segment]:
    """Transcribe audio into timed segments.

    Args:
        audio_path: WAV/media path (prefer 16 kHz mono PCM from download step).
        model_size: Whisper model name (tiny/base/small/medium/...).
        language: ISO code or None for auto-detect.
    """
    model = _get_model(model_size)
    audio = load_audio_wav(Path(audio_path))

    segments_iter, _info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )

    result: list[Segment] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        result.append(Segment(start=float(seg.start), end=float(seg.end), text=text))
    return result
