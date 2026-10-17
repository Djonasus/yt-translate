# yt-translate

Локальный дубляж YouTube-видео на русский: скачивание → распознавание речи → перевод → озвучка → готовое видео.

## Требования

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) (обязательно)

Windows (winget):

```bash
winget install Gyan.FFmpeg
```

После установки **перезапустите** терминал / Gradio. Приложение также само ищет ffmpeg в типичных путях WinGet.

Проверка:

```bash
ffmpeg -version
```


## Установка

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

На Linux/macOS: `source .venv/bin/activate`.

## Запуск

```bash
python app.py
```

Откройте http://127.0.0.1:7860

## Пайплайн

1. **yt-dlp** — скачивание видео и аудио
2. **faster-whisper** — транскрипция с таймкодами
3. **перевод** — bing / yandex / alibaba / sogou / google (+ fallback deep-translator)
4. **edge-tts** — озвучка (простой TTS; клонирование голоса — позже)
5. **ffmpeg** — склейка озвучки с видео

### Checkpoint / resume

Каждый шаг пишется в `output/<job_id>/`:

| Файл | Шаг |
|------|-----|
| `video.mp4`, `audio.wav` | download |
| `segments.json` | transcribe (+ перевод по мере готовности) |
| `dubbed_audio.wav` | tts |
| `dubbed.mp4` | mux |
| `meta.json` | состояние шагов |

В UI включите «Продолжить с checkpoint» — при том же URL подхватится незавершённый job. Можно указать `job id` вручную.

Результаты: `output/<job_id>/dubbed.mp4`.
