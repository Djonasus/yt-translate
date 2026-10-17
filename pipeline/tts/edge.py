"""Edge TTS backend (simple neural voices, no cloning)."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts

from pipeline.ffmpeg_util import ffmpeg_bin, ffprobe_bin, strip_ansi
from pipeline.tts.base import TTSBackend

logger = logging.getLogger(__name__)

RU_VOICES = [
    ("dmitry", "ru-RU-DmitryNeural"),
    ("svetlana", "ru-RU-SvetlanaNeural"),
]

DEFAULT_VOICE = "ru-RU-DmitryNeural"

# Edge refuses punctuation-only / emoji-only inputs with NoAudioReceived
_SPEAKABLE_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")


def list_ru_voices() -> list[tuple[str, str]]:
    """Return (label, voice_id) pairs for Gradio dropdown."""
    return RU_VOICES


def is_speakable(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and bool(_SPEAKABLE_RE.search(t))


class EdgeTTSBackend(TTSBackend):
    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        self.voice = voice or DEFAULT_VOICE

    def synthesize(
        self,
        text: str,
        out_path: Path,
        *,
        reference_audio_path: Path | None = None,
    ) -> Path:
        _ = reference_audio_path
        if not is_speakable(text):
            raise ValueError(f"Unspeakable text for edge-tts: {text!r}")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mp3_path = out_path.with_suffix(".mp3")
        _run_async(self._save_with_retry(text, mp3_path))

        if out_path.suffix.lower() == ".mp3":
            return mp3_path

        _ffmpeg_convert(mp3_path, out_path)
        mp3_path.unlink(missing_ok=True)
        return out_path

    def synthesize_many(
        self,
        items: list[tuple[str, Path]],
        *,
        concurrency: int = 6,
    ) -> list[Path]:
        """Synthesize many texts; skips failures instead of aborting the batch."""
        if not items:
            return []
        return _run_async(self._save_many(items, concurrency=concurrency))

    async def _save_with_retry(self, text: str, path: Path, attempts: int = 4) -> None:
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(str(path))
                if path.exists() and path.stat().st_size > 0:
                    return
                raise RuntimeError("edge-tts wrote empty file")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                # Empty/punct text never recovers — don't burn retries
                if "no audio was received" in msg and not is_speakable(text):
                    break
                await asyncio.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"edge-tts failed for {text[:60]!r}: {last_err}") from last_err

    async def _save_many(
        self,
        items: list[tuple[str, Path]],
        *,
        concurrency: int,
    ) -> list[Path]:
        sem = asyncio.Semaphore(max(1, concurrency))
        results: list[Path | None] = [None] * len(items)

        async def one(idx: int, text: str, out_path: Path) -> None:
            if not is_speakable(text):
                logger.warning("Skip unspeakable TTS segment %s: %r", idx, text[:40])
                return
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mp3_path = out_path.with_suffix(".mp3")
            try:
                async with sem:
                    await self._save_with_retry(text, mp3_path)
                if out_path.suffix.lower() == ".mp3":
                    results[idx] = mp3_path
                    return
                await asyncio.to_thread(_ffmpeg_convert, mp3_path, out_path)
                mp3_path.unlink(missing_ok=True)
                results[idx] = out_path
            except Exception as exc:  # noqa: BLE001
                logger.warning("TTS segment %s failed (%s): %r", idx, exc, text[:40])
                # Soft pause after failures — often rate/network related
                await asyncio.sleep(0.4)

        await asyncio.gather(*(one(i, t, p) for i, (t, p) in enumerate(items)))
        ok = [p for p in results if p is not None]
        if not ok and items:
            raise RuntimeError(
                "Озвучка: ни один сегмент не синтезировался (edge-tts). "
                "Проверьте сеть или попробуйте снова."
            )
        return ok


def _run_async(coro):
    """Run coroutine whether or not an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _ffmpeg_convert(src: Path, dst: Path) -> None:
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i",
        str(src),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "24000",
        "-ac",
        "1",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(strip_ansi(f"ffmpeg convert failed:\n{result.stderr}"))


def fit_duration(wav_path: Path, target_sec: float, out_path: Path | None = None) -> Path:
    """Stretch/pad audio to approximately ``target_sec`` seconds via atempo + pad."""
    out_path = out_path or wav_path
    if target_sec <= 0.05:
        return wav_path

    probe = subprocess.run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        current = float(probe.stdout.strip())
    except ValueError:
        current = 0.0

    if current <= 0:
        return wav_path

    ratio = current / target_sec
    filters: list[str] = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.02:
        filters.append(f"atempo={remaining:.4f}")

    filters.append(f"apad=whole_dur={target_sec}")
    filter_str = ",".join(filters)

    tmp = wav_path.with_name(wav_path.stem + "_fit.wav")
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i",
        str(wav_path),
        "-af",
        filter_str,
        "-t",
        f"{target_sec:.3f}",
        "-acodec",
        "pcm_s16le",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(strip_ansi(f"ffmpeg fit_duration failed:\n{result.stderr}"))

    if out_path.resolve() == wav_path.resolve():
        tmp.replace(wav_path)
        return wav_path
    tmp.replace(out_path)
    return out_path
