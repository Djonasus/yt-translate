"""Translate segment texts to Russian via multiple free providers."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from pipeline.transcribe import Segment

ProgressCb = Callable[[str, float | None], None]
SaveCb = Callable[[], None]

# Prefer engines that worked without API keys in practice
TRANSLATOR_ENGINES = (
    "bing",
    "yandex",
    "alibaba",
    "sogou",
    "google",
)


def translate_segments(
    segments: list[Segment],
    source: str = "en",
    target: str = "ru",
    *,
    progress: ProgressCb | None = None,
    on_save: SaveCb | None = None,
    engines: tuple[str, ...] = TRANSLATOR_ENGINES,
) -> list[Segment]:
    """Translate missing segments; skip those that already have ``translation``."""
    pending = [
        (i, seg)
        for i, seg in enumerate(segments)
        if seg.text.strip() and not (seg.translation or "").strip()
    ]
    for seg in segments:
        if not seg.text.strip():
            seg.translation = ""

    if not pending:
        return segments

    src = "en" if source in ("auto", "", None) else source
    total = len(pending)
    dead_engines: set[str] = set()

    for n, (i, seg) in enumerate(pending):
        if progress and (n % 3 == 0 or n + 1 == total):
            progress(
                f"Перевод {n + 1}/{total} (сегмент {i})…",
                0.55 + 0.13 * (n / max(total, 1)),
            )

        seg.translation = _translate_text(
            seg.text.strip(),
            src,
            target,
            engines=engines,
            dead_engines=dead_engines,
        )

        if on_save and (n % 5 == 0 or n + 1 == total):
            on_save()

        time.sleep(0.35)

    if on_save:
        on_save()
    return segments


def _translate_text(
    text: str,
    source: str,
    target: str,
    *,
    engines: tuple[str, ...],
    dead_engines: set[str],
) -> str:
    errors: list[str] = []

    # 1) translators package (bing/yandex/…)
    try:
        import translators as ts
    except ImportError:
        ts = None  # type: ignore[assignment]

    if ts is not None:
        for eng in engines:
            if eng in dead_engines:
                continue
            for attempt in range(2):
                try:
                    result = ts.translate_text(
                        text,
                        translator=eng,
                        from_language=source,
                        to_language=target,
                    )
                    if result and result.strip():
                        return result.strip()
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc).lower()
                    errors.append(f"{eng}: {exc}")
                    if any(k in msg for k in ("too many", "429", "rate", "forbidden", "401")):
                        dead_engines.add(eng)
                        break
                    time.sleep(0.8 * (attempt + 1))

    # 2) deep_translator fallbacks
    for name, fn in _deep_backends(source, target):
        if name in dead_engines:
            continue
        try:
            result = fn(text)
            if result and result.strip():
                return result.strip()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            dead_engines.add(name)
            time.sleep(0.4)

    raise RuntimeError(
        "Translation failed on all backends: " + " | ".join(errors[:6])
    )


def _deep_backends(source: str, target: str) -> list[tuple[str, Callable[[str], str]]]:
    backends: list[tuple[str, Callable[[str], str]]] = []
    try:
        from deep_translator import GoogleTranslator, LibreTranslator, MyMemoryTranslator

        gt = GoogleTranslator(source="auto" if source == "en" else source, target=target)
        backends.append(("deep_google", gt.translate))

        try:
            mm = MyMemoryTranslator(source="en-GB", target="ru-RU")
            backends.append(("deep_mymemory", mm.translate))
        except Exception:  # noqa: BLE001
            pass

        try:
            lt = LibreTranslator(source=source, target=target)
            backends.append(("libre", lt.translate))
        except Exception:  # noqa: BLE001
            pass
    except ImportError:
        pass
    return backends


def translations_done(segments: list[Segment]) -> bool:
    texts = [s for s in segments if s.text.strip()]
    if not texts:
        return False
    return all((s.translation or "").strip() for s in texts)
