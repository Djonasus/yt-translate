"""End-to-end pipeline orchestration with resume checkpoints."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.checkpoint import (
    OUTPUT_ROOT,
    JobMeta,
    find_resumable_job,
    has_download,
    load_meta,
    load_segments,
    mark_step,
    save_meta,
    save_segments,
    workdir_for,
)
from pipeline.download import download_video
from pipeline.mux import build_dubbed_audio, mux_video, probe_duration
from pipeline.transcribe import Segment, transcribe
from pipeline.translate import translate_segments, translations_done
from pipeline.tts.edge import DEFAULT_VOICE, EdgeTTSBackend

ProgressCb = Callable[[str, float | None], None]


@dataclass
class PipelineResult:
    video_path: Path
    segments: list[Segment] = field(default_factory=list)
    job_id: str = ""


def run_pipeline(
    url: str,
    *,
    whisper_model: str = "base",
    voice: str = DEFAULT_VOICE,
    language: str | None = None,
    keep_original_volume: float = 0.08,
    progress: ProgressCb | None = None,
    resume: bool = True,
    job_id: str | None = None,
) -> PipelineResult:
    """Download → ASR → translate → TTS → mux. Resumes from checkpoints when possible."""

    def report(msg: str, fraction: float | None = None) -> None:
        if progress:
            progress(msg, fraction)

    workdir, meta = _resolve_job(
        url,
        whisper_model=whisper_model,
        voice=voice,
        keep_original_volume=keep_original_volume,
        resume=resume,
        job_id=job_id,
    )
    report(f"Job {meta.job_id} · шаги: {_steps_summary(meta)}", 0.01)

    # --- download ---
    video_path = workdir / "video.mp4"
    audio_path = workdir / "audio.wav"
    if meta.steps.get("download") and has_download(workdir):
        report("Скачивание: пропуск (есть checkpoint)", 0.28)
    else:
        video_path, audio_path = download_video(url, workdir, progress=report)
        mark_step(workdir, meta, "download")

    # --- transcribe ---
    segments: list[Segment] = []
    if meta.steps.get("transcribe"):
        segments = load_segments(workdir)
        if segments:
            report(f"Распознавание: пропуск ({len(segments)} сегм. из checkpoint)", 0.50)
        else:
            meta.steps["transcribe"] = False

    if not meta.steps.get("transcribe"):
        report(f"Распознавание речи (Whisper {whisper_model})…", 0.32)
        try:
            segments = transcribe(audio_path, model_size=whisper_model, language=language)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Whisper упал: {exc}") from exc
        if not segments:
            raise RuntimeError("Не удалось распознать речь — сегменты пустые")
        save_segments(workdir, segments)
        mark_step(workdir, meta, "transcribe")
        report(f"Распознавание готово: {len(segments)} сегм. (сохранено)", 0.50)

    # --- translate ---
    if meta.steps.get("translate") and translations_done(segments):
        report(f"Перевод: пропуск ({len(segments)} сегм. из checkpoint)", 0.68)
    else:
        # Reload in case resume mid-translate
        disk = load_segments(workdir)
        if disk and len(disk) == len(segments):
            for local, saved in zip(segments, disk):
                if saved.translation:
                    local.translation = saved.translation
        elif disk and not segments:
            segments = disk

        pending = sum(1 for s in segments if s.text.strip() and not (s.translation or "").strip())
        report(f"Перевод на русский ({pending} без перевода из {len(segments)})…", 0.55)

        def _persist() -> None:
            save_segments(workdir, segments)

        try:
            translate_segments(segments, progress=report, on_save=_persist)
        except Exception as exc:  # noqa: BLE001
            save_segments(workdir, segments)  # keep partial progress
            raise RuntimeError(
                f"Перевод упал (частичный результат сохранён в {workdir}): {exc}"
            ) from exc

        save_segments(workdir, segments)
        if translations_done(segments):
            mark_step(workdir, meta, "translate")
        report(f"Перевод готов: {len(segments)} сегм. (сохранено)", 0.68)

    # --- tts ---
    dubbed_audio = workdir / "dubbed_audio.wav"
    if meta.steps.get("tts") and dubbed_audio.exists() and dubbed_audio.stat().st_size > 0:
        report("Озвучка: пропуск (есть checkpoint)", 0.88)
    else:
        report(f"Озвучка (edge-tts), {len(segments)} сегм.…", 0.70)
        duration = probe_duration(video_path)
        tts = EdgeTTSBackend(voice=voice or DEFAULT_VOICE)
        dubbed_audio = build_dubbed_audio(
            segments,
            tts,
            workdir,
            duration,
            progress=report,
        )
        mark_step(workdir, meta, "tts")

    # --- mux ---
    out_path = workdir / "dubbed.mp4"
    if meta.steps.get("mux") and out_path.exists() and out_path.stat().st_size > 0:
        report("Склейка: пропуск (есть checkpoint)", 0.98)
    else:
        report("Склейка видео…", 0.90)
        mux_video(
            video_path,
            dubbed_audio,
            out_path,
            keep_original_volume=keep_original_volume,
        )
        mark_step(workdir, meta, "mux")

    report("Готово", 1.0)
    return PipelineResult(video_path=out_path, segments=segments, job_id=meta.job_id)


def _resolve_job(
    url: str,
    *,
    whisper_model: str,
    voice: str,
    keep_original_volume: float,
    resume: bool,
    job_id: str | None,
) -> tuple[Path, JobMeta]:
    existing: Path | None = None
    if resume:
        existing = find_resumable_job(url, job_id=job_id or None)

    if existing is not None:
        meta = load_meta(existing)
        if meta is None:
            meta = JobMeta(
                job_id=existing.name,
                url=url,
                whisper_model=whisper_model,
                voice=voice,
                keep_original_volume=keep_original_volume,
            )
            save_meta(existing, meta)
        else:
            # Keep original url/model; update voice / volume if provided
            if voice:
                meta.voice = voice
            if abs(meta.keep_original_volume - keep_original_volume) > 1e-6:
                meta.keep_original_volume = keep_original_volume
                # Volume only affects mux — force remux on resume
                meta.steps["mux"] = False
            save_meta(existing, meta)
        return existing, meta

    new_id = uuid.uuid4().hex[:12]
    workdir = workdir_for(new_id)
    meta = JobMeta(
        job_id=new_id,
        url=url,
        whisper_model=whisper_model,
        voice=voice,
        keep_original_volume=keep_original_volume,
    )
    save_meta(workdir, meta)
    return workdir, meta


def _steps_summary(meta: JobMeta) -> str:
    parts = []
    for step, done in meta.steps.items():
        parts.append(f"{step}{'✓' if done else '·'}")
    return " ".join(parts)


def list_jobs(limit: int = 20) -> list[JobMeta]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[float, JobMeta]] = []
    for wd in OUTPUT_ROOT.iterdir():
        if not wd.is_dir() or wd.name.startswith("_"):
            continue
        meta = load_meta(wd)
        if not meta:
            continue
        mtime = (wd / "meta.json").stat().st_mtime
        jobs.append((mtime, meta))
    jobs.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in jobs[:limit]]
