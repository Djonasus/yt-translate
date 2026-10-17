"""Download YouTube video/audio via yt-dlp."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import yt_dlp

from pipeline.ffmpeg_util import ffmpeg_bin, strip_ansi

ProgressCb = Callable[[str, float | None], None]


def download_video(
    url: str,
    workdir: Path,
    *,
    progress: ProgressCb | None = None,
) -> tuple[Path, Path]:
    """Download best video+audio mux and extract a WAV for ASR.

    Returns:
        (video_path, audio_wav_path)
    """
    workdir.mkdir(parents=True, exist_ok=True)
    video_path = workdir / "video.mp4"
    ffmpeg = ffmpeg_bin()

    def report(msg: str, fraction: float | None = None) -> None:
        if progress:
            progress(msg, fraction)

    report("Скачивание: получение информации о видео…", 0.02)
    last_ui = {"t": 0.0}

    def hook(d: dict) -> None:
        now = time.monotonic()
        # Throttle UI updates (~4/sec) except on finish
        if d.get("status") != "finished" and now - last_ui["t"] < 0.25:
            return
        last_ui["t"] = now

        status = d.get("status")
        if status == "downloading":
            msg, frac = _format_download_progress(d)
            report(msg, frac)
        elif status == "finished":
            name = Path(d.get("filename") or "").name or "файл"
            report(f"Скачивание: загружен «{name}», обработка…", 0.22)
        elif status == "error":
            report("Скачивание: ошибка yt-dlp", 0.05)

    ydl_opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "outtmpl": str(workdir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": str(Path(ffmpeg).parent),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
        "progress_hooks": [hook],
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = (info or {}).get("title") or ""
            if title:
                report(f"Скачивание: «{title[:60]}» — готово", 0.24)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(strip_ansi(str(exc))) from exc

    report("Скачивание: извлечение аудиодорожки…", 0.26)
    downloaded = _resolve_downloaded(workdir, info)
    if downloaded.resolve() != video_path.resolve():
        if video_path.exists():
            video_path.unlink()
        downloaded.replace(video_path)

    audio_path = workdir / "audio.wav"
    _extract_wav(video_path, audio_path)
    report("Скачивание: готово", 0.28)
    return video_path, audio_path


def _format_download_progress(d: dict) -> tuple[str, float]:
    total = d.get("total_bytes") or d.get("total_bytes_estimate")
    done = d.get("downloaded_bytes") or 0
    speed = d.get("speed")
    eta = d.get("eta")
    frag_i = d.get("fragment_index")
    frag_n = d.get("fragment_count")

    info_dict = d.get("info_dict") or {}
    fmt = info_dict.get("format_note") or info_dict.get("format_id") or ""
    vcodec = info_dict.get("vcodec") or "none"
    kind = "видео" if vcodec not in ("none", None) else "аудио"

    parts = [f"Скачивание ({kind})"]
    pct = None
    if total and total > 0:
        pct = min(100.0, done * 100.0 / total)
        parts.append(f"{pct:.0f}%")
        parts.append(f"{_fmt_bytes(done)} / {_fmt_bytes(total)}")
    else:
        parts.append(_fmt_bytes(done))

    if frag_i is not None and frag_n:
        parts.append(f"фрагмент {frag_i}/{frag_n}")
    if speed:
        parts.append(f"{_fmt_bytes(speed)}/s")
    if eta is not None:
        parts.append(f"ETA {_fmt_eta(eta)}")
    if fmt:
        parts.append(str(fmt))

    # Map download percent into overall pipeline 5%–22%
    frac = 0.05 + (0.17 * (pct / 100.0)) if pct is not None else 0.08
    return " · ".join(parts), frac


def _fmt_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_eta(seconds: float | int) -> str:
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "?"
    if s < 0:
        return "?"
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _resolve_downloaded(workdir: Path, info: dict) -> Path:
    req = info.get("requested_downloads") or []
    for item in req:
        fp = item.get("filepath")
        if fp and Path(fp).exists():
            return Path(fp)

    filename = info.get("_filename") or info.get("filename")
    if filename and Path(filename).exists():
        return Path(filename)

    candidates = sorted(workdir.glob("video.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return path

    raise FileNotFoundError("yt-dlp did not produce a video file")


def _extract_wav(video_path: Path, audio_path: Path) -> None:
    cmd = [
        ffmpeg_bin(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(strip_ansi(f"ffmpeg audio extract failed:\n{result.stderr}"))
