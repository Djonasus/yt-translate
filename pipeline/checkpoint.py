"""Job checkpoints: persist each pipeline step for resume."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pipeline.transcribe import Segment

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output"

STEPS = ("download", "transcribe", "translate", "tts", "mux")


@dataclass
class JobMeta:
    job_id: str
    url: str
    video_id: str = ""
    whisper_model: str = "base"
    voice: str = ""
    keep_original_volume: float = 0.08
    steps: dict[str, bool] = field(default_factory=dict)
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            self.steps = {s: False for s in STEPS}
        if not self.video_id:
            self.video_id = extract_video_id(self.url)


def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]
    if "youtube" in host or "youtube-nocookie" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0] in ("embed", "shorts", "live", "v") and len(parts) > 1:
            return parts[1]
    m = re.search(r"(?:v=|/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url


def workdir_for(job_id: str) -> Path:
    path = OUTPUT_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def meta_path(workdir: Path) -> Path:
    return workdir / "meta.json"


def segments_path(workdir: Path) -> Path:
    return workdir / "segments.json"


def load_meta(workdir: Path) -> JobMeta | None:
    path = meta_path(workdir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return JobMeta(**data)


def save_meta(workdir: Path, meta: JobMeta) -> None:
    meta.updated_at = datetime.now(timezone.utc).isoformat()
    meta_path(workdir).write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mark_step(workdir: Path, meta: JobMeta, step: str, done: bool = True) -> None:
    meta.steps[step] = done
    save_meta(workdir, meta)


def save_segments(workdir: Path, segments: list[Segment]) -> None:
    payload = [asdict(s) for s in segments]
    segments_path(workdir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_segments(workdir: Path) -> list[Segment]:
    path = segments_path(workdir)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Segment(**item) for item in data]


def find_resumable_job(url: str, job_id: str | None = None) -> Path | None:
    """Find an incomplete job to resume (explicit id or latest matching URL)."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if job_id:
        wd = OUTPUT_ROOT / job_id
        if wd.is_dir() and meta_path(wd).exists():
            return wd
        return None

    vid = extract_video_id(url)
    candidates: list[tuple[float, Path]] = []
    for wd in OUTPUT_ROOT.iterdir():
        if not wd.is_dir() or wd.name.startswith("_"):
            continue
        meta = load_meta(wd)
        if not meta:
            continue
        if meta.video_id != vid and meta.url.strip() != url.strip():
            continue
        if meta.steps.get("mux"):
            continue  # already finished
        mtime = meta_path(wd).stat().st_mtime
        candidates.append((mtime, wd))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def has_download(workdir: Path) -> bool:
    video = workdir / "video.mp4"
    audio = workdir / "audio.wav"
    return video.exists() and audio.exists() and video.stat().st_size > 0
