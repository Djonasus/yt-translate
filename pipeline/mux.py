"""Assemble dubbed audio timeline and mux with video."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.ffmpeg_util import ffmpeg_bin, ffprobe_bin, strip_ansi
from pipeline.transcribe import Segment
from pipeline.tts.base import TTSBackend
from pipeline.tts.edge import EdgeTTSBackend, fit_duration, is_speakable


# Concurrent edge-tts requests — enough for speed, low enough to avoid rate limits
TTS_CONCURRENCY = 6

ProgressCb = Callable[[str, float | None], None]


def build_dubbed_audio(
    segments: list[Segment],
    tts: TTSBackend,
    workdir: Path,
    total_duration: float,
    *,
    concurrency: int = TTS_CONCURRENCY,
    progress: ProgressCb | None = None,
) -> Path:
    """Synthesize each segment (in parallel when possible), place on a timeline."""
    segs_dir = workdir / "tts_segments"
    segs_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[int, float, float, str, Path, Path]] = []
    skipped = 0
    for i, seg in enumerate(segments):
        text = (seg.translation or seg.text or "").strip()
        if not text:
            continue
        if not is_speakable(text):
            skipped += 1
            continue
        dur = max(0.1, seg.end - seg.start)
        raw = segs_dir / f"seg_{i:04d}.wav"
        fitted = segs_dir / f"seg_{i:04d}_fit.wav"
        jobs.append((i, seg.start, dur, text, raw, fitted))

    if progress and skipped:
        progress(f"Озвучка: пропуск {skipped} пустых/без букв сегм.", None)

    if isinstance(tts, EdgeTTSBackend):
        fitted_clips = _synthesize_parallel_edge(tts, jobs, concurrency, progress)
    else:
        fitted_clips = _synthesize_sequential(tts, jobs, progress)


    out_wav = workdir / "dubbed_audio.wav"
    silence = workdir / "silence.wav"
    ff = ffmpeg_bin()
    _run(
        [
            ff,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            f"{total_duration:.3f}",
            "-acodec",
            "pcm_s16le",
            str(silence),
        ]
    )

    if not fitted_clips:
        silence.replace(out_wav)
        return out_wav

    cmd: list[str] = [ff, "-y", "-i", str(silence)]

    filter_parts: list[str] = []
    labels = ["[0:a]"]

    for start, path in fitted_clips:
        cmd.extend(["-i", str(path)])
        inp = len(labels)
        delay_ms = int(max(0.0, start) * 1000)
        lab = f"a{inp}"
        filter_parts.append(
            f"[{inp}:a]adelay={delay_ms}|{delay_ms},asetpts=PTS-STARTPTS[{lab}]"
        )
        labels.append(f"[{lab}]")

    mix_inputs = "".join(labels)
    n_inputs = len(labels)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={n_inputs}:duration=first:dropout_transition=0:normalize=0[out]"
    )

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[out]",
            "-acodec",
            "pcm_s16le",
            "-t",
            f"{total_duration:.3f}",
            str(out_wav),
        ]
    )
    _run(cmd)
    return out_wav


def _synthesize_sequential(
    tts: TTSBackend,
    jobs: list[tuple[int, float, float, str, Path, Path]],
    progress: ProgressCb | None,
) -> list[tuple[float, Path]]:
    fitted_clips: list[tuple[float, Path]] = []
    total = len(jobs)
    for n, (_i, start, dur, text, raw, fitted) in enumerate(jobs, start=1):
        if progress:
            progress(f"Озвучка {n}/{total}…", None)
        try:
            if not (fitted.exists() and fitted.stat().st_size > 0):
                tts.synthesize(text, raw)
                if not (raw.exists() and raw.stat().st_size > 0):
                    continue
                fit_duration(raw, dur, fitted)
            if fitted.exists() and fitted.stat().st_size > 0:
                fitted_clips.append((start, fitted))
        except Exception:  # noqa: BLE001 — one bad segment must not kill the job
            continue
    return fitted_clips


def _synthesize_parallel_edge(
    tts: EdgeTTSBackend,
    jobs: list[tuple[int, float, float, str, Path, Path]],
    concurrency: int,
    progress: ProgressCb | None,
) -> list[tuple[float, Path]]:
    """Synthesize many segments concurrently via one asyncio event loop."""
    to_synth: list[tuple[str, Path]] = []
    for _i, _start, _dur, text, raw, fitted in jobs:
        if fitted.exists() and fitted.stat().st_size > 0:
            continue
        to_synth.append((text, raw))

    total = len(jobs)
    pending = len(to_synth)
    if progress:
        progress(f"Озвучка (параллельно ×{concurrency}): {pending} из {total}…", None)

    if to_synth:
        tts.synthesize_many(to_synth, concurrency=concurrency)

    # Fit only segments that actually got raw audio
    fit_jobs = [
        (dur, raw, fitted)
        for _i, _start, dur, _text, raw, fitted in jobs
        if raw.exists()
        and raw.stat().st_size > 0
        and not (fitted.exists() and fitted.stat().st_size > 0)
    ]

    def _fit(item: tuple[float, Path, Path]) -> None:
        dur, raw, fitted = item
        try:
            fit_duration(raw, dur, fitted)
        except Exception:  # noqa: BLE001
            pass

    if fit_jobs:
        workers = min(4, len(fit_jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_fit, fit_jobs))

    return [
        (start, fitted)
        for _i, start, _dur, _text, _raw, fitted in jobs
        if fitted.exists() and fitted.stat().st_size > 0
    ]


def _has_audio_stream(path: Path) -> bool:
    result = subprocess.run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def mux_video(
    video_path: Path,
    dubbed_audio: Path,
    out_path: Path,
    *,
    keep_original_volume: float = 0.08,
) -> Path:
    """Replace/mix audio onto video. Original track kept at low volume for ambience."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    replace_only = keep_original_volume <= 0 or not _has_audio_stream(video_path)

    ff = ffmpeg_bin()
    if replace_only:
        cmd = [
            ff,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(dubbed_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
        _run(cmd)
        return out_path

    vol = keep_original_volume
    filter_complex = (
        f"[0:a]volume={vol}[orig];"
        f"[1:a]volume=1.0[dub];"
        f"[orig][dub]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    cmd = [
        ff,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(dubbed_audio),
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    _run(cmd)
    return out_path


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not probe duration of {path}") from exc


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(strip_ansi(f"Command failed: {' '.join(cmd)}\n{result.stderr}"))
