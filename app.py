"""Gradio UI for YouTube → Russian dubbing."""

from __future__ import annotations

import threading
from collections.abc import Generator
from queue import Empty, Queue

import gradio as gr

from pipeline.checkpoint import load_segments
from pipeline.ffmpeg_util import ensure_ffmpeg_on_path, strip_ansi
from pipeline.run import PipelineResult, list_jobs, run_pipeline
from pipeline.tts.edge import DEFAULT_VOICE, list_ru_voices

ensure_ffmpeg_on_path()

WHISPER_MODELS = ["tiny", "base", "small", "medium"]
VOICE_CHOICES = {label: vid for label, vid in list_ru_voices()}


def _format_transcript(segments) -> str:
    lines: list[str] = []
    for seg in segments:
        ts = f"[{seg.start:06.1f}–{seg.end:06.1f}]"
        ru = seg.translation or "—"
        lines.append(f"{ts}\nEN: {seg.text}\nRU: {ru}\n")
    return "\n".join(lines)


def _jobs_table() -> str:
    jobs = list_jobs(15)
    if not jobs:
        return "Нет сохранённых job."
    lines = ["job_id | video | шаги | обновлён"]
    for j in jobs:
        steps = "".join("✓" if j.steps.get(s) else "·" for s in ("download", "transcribe", "translate", "tts", "mux"))
        lines.append(f"{j.job_id} | {j.video_id or '?'} | {steps} | {j.updated_at[:19]}")
    return "\n".join(lines)


def translate_video(
    url: str,
    whisper_model: str,
    voice_label: str,
    original_volume: float,
    resume: bool,
    job_id: str,
    progress=gr.Progress(),
) -> Generator[tuple[str | None, str, str], None, None]:
    if not url or not url.strip():
        raise gr.Error("Вставьте ссылку на YouTube")

    voice = VOICE_CHOICES.get(voice_label, DEFAULT_VOICE)
    vol = max(0.0, min(1.0, float(original_volume)))
    events: Queue = Queue()

    def on_progress(msg: str, fraction: float | None = None) -> None:
        events.put(("status", msg, fraction))

    def worker() -> None:
        try:
            result = run_pipeline(
                url.strip(),
                whisper_model=whisper_model or "base",
                voice=voice,
                keep_original_volume=vol,
                progress=on_progress,
                resume=bool(resume),
                job_id=(job_id or "").strip() or None,
            )
            events.put(("done", result, None))
        except Exception as exc:  # noqa: BLE001
            events.put(("error", exc, None))

    yield None, "", "Запуск…"
    progress(0.01, desc="Запуск…")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            kind, payload, fraction = events.get(timeout=0.5)
        except Empty:
            if not thread.is_alive():
                raise gr.Error("Пайплайн завершился без результата")
            continue

        if kind == "status":
            msg = str(payload)
            frac = fraction if fraction is not None else 0.05
            progress(min(0.99, max(0.0, frac)), desc=msg)
            yield None, "", msg
        elif kind == "done":
            result: PipelineResult = payload  # type: ignore[assignment]
            transcript = _format_transcript(result.segments)
            status = f"Готово · job {result.job_id} → {result.video_path}"
            progress(1.0, desc="Готово")
            yield str(result.video_path), transcript, status
            break
        elif kind == "error":
            err = strip_ansi(str(payload))
            # Try show partial transcript from latest matching job
            partial = ""
            try:
                from pipeline.checkpoint import find_resumable_job

                wd = find_resumable_job(url.strip(), job_id=(job_id or "").strip() or None)
                if wd:
                    segs = load_segments(wd)
                    if segs:
                        partial = _format_transcript(segs)
            except Exception:  # noqa: BLE001
                pass
            progress(0.0, desc="Ошибка")
            yield None, partial, f"Ошибка: {err}"
            raise gr.Error(err)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="yt-translate") as demo:
        gr.Markdown(
            "# yt-translate\n"
            "Скачать YouTube-видео и озвучить на русском "
            "(Whisper → перевод → edge-tts).\n\n"
            "Каждый шаг сохраняется в `output/<job>/` — можно продолжить после ошибки."
        )
        with gr.Row():
            with gr.Column(scale=2):
                url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=…",
                )
                whisper_model = gr.Dropdown(
                    choices=WHISPER_MODELS,
                    value="base",
                    label="Модель Whisper",
                )
                voice = gr.Dropdown(
                    choices=list(VOICE_CHOICES.keys()),
                    value=list(VOICE_CHOICES.keys())[0],
                    label="Голос TTS (edge-tts)",
                )
                original_volume = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.08,
                    step=0.01,
                    label="Громкость оригинальной дорожки",
                    info="0 = только дубляж · 0.08 ≈ тихий фон · 1 = оригинал как есть",
                )
                resume = gr.Checkbox(
                    label="Продолжить с checkpoint (тот же URL / job id)",
                    value=True,
                )
                job_id = gr.Textbox(
                    label="Job id (опционально)",
                    placeholder="оставьте пустым — найдёт по URL",
                )
                gr.Checkbox(
                    label="Клонировать голос из оригинала (скоро)",
                    value=False,
                    interactive=False,
                )
                run_btn = gr.Button("Перевести", variant="primary")
                jobs_box = gr.Textbox(
                    label="Сохранённые job (d/t/tr/tts/mux)",
                    value=_jobs_table(),
                    lines=8,
                    interactive=False,
                )
                refresh_btn = gr.Button("Обновить список job")
            with gr.Column(scale=3):
                video_out = gr.Video(label="Результат")
                status = gr.Textbox(label="Статус", interactive=False, lines=3)
                with gr.Accordion("Текст: оригинал и перевод", open=False):
                    transcript = gr.Textbox(lines=16, interactive=False)

        run_btn.click(
            fn=translate_video,
            inputs=[url, whisper_model, voice, original_volume, resume, job_id],
            outputs=[video_out, transcript, status],
        ).then(fn=_jobs_table, outputs=[jobs_box])
        refresh_btn.click(fn=_jobs_table, outputs=[jobs_box])
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch(server_name="127.0.0.1", server_port=7860)
