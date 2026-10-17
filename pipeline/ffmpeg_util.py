"""Locate ffmpeg/ffprobe and ensure they are usable by yt-dlp and subprocess."""

from __future__ import annotations

import os
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _candidate_bin_dirs() -> list[Path]:
    dirs: list[Path] = []
    local = Path.home() / "AppData" / "Local"
    winget_links = local / "Microsoft" / "WinGet" / "Links"
    if winget_links.is_dir():
        dirs.append(winget_links)

    winget_pkgs = local / "Microsoft" / "WinGet" / "Packages"
    if winget_pkgs.is_dir():
        for bin_dir in winget_pkgs.glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
            if bin_dir.is_dir():
                dirs.append(bin_dir)

    for base in (
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
        Path(r"C:\Program Files\Gyan\ffmpeg\bin"),
    ):
        if base.is_dir():
            dirs.append(base)

    # Env override
    for key in ("FFMPEG_PATH", "FFMPEG_BINARY"):
        val = os.environ.get(key)
        if val:
            p = Path(val)
            dirs.insert(0, p.parent if p.is_file() else p)

    return dirs


@lru_cache(maxsize=1)
def find_ffmpeg() -> Path:
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)

    for d in _candidate_bin_dirs():
        exe = d / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
        if exe.is_file():
            return exe

    raise FileNotFoundError(
        "ffmpeg не найден. Установите его и добавьте в PATH "
        "(winget install Gyan.FFmpeg) или задайте FFMPEG_PATH."
    )


@lru_cache(maxsize=1)
def find_ffprobe() -> Path:
    which = shutil.which("ffprobe")
    if which:
        return Path(which)

    ffmpeg = find_ffmpeg()
    probe = ffmpeg.with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
    if probe.is_file():
        return probe

    for d in _candidate_bin_dirs():
        exe = d / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if exe.is_file():
            return exe

    raise FileNotFoundError("ffprobe не найден рядом с ffmpeg")


def ensure_ffmpeg_on_path() -> Path:
    """Put ffmpeg's directory on PATH for this process (helps yt-dlp)."""
    ffmpeg = find_ffmpeg()
    bin_dir = str(ffmpeg.parent)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep)
    if bin_dir not in parts:
        os.environ["PATH"] = bin_dir + os.pathsep + path
    return ffmpeg


def ffmpeg_bin() -> str:
    return str(ensure_ffmpeg_on_path())


def ffprobe_bin() -> str:
    ensure_ffmpeg_on_path()
    return str(find_ffprobe())
