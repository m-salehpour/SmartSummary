# src/translate_helpers.py
from __future__ import annotations

import asyncio
import json
from pathlib import Path
# --- add these imports at top if missing ---
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Tuple

import docx  # pip install python-docx


def read_asr_json_text_and_lang(json_path: str | Path) -> Tuple[str, Optional[str]]:
    """
    Read a Whisper/WhisperX-style JSON and return:
      (full_text, language_if_present)

    The JSON can be:
      - {"segments": [{"text": "..."}...], "language": "fa"/"he"/...}
      - {"text": "...", "language": "..."}  (fallback)
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    lang = data.get("language")

    if isinstance(data.get("segments"), Iterable):
        text = " ".join((seg.get("text", "") or "").strip() for seg in data["segments"])
    else:
        text = (data.get("text") or "").strip()

    if not text:
        raise ValueError(f"No text found in JSON: {p}")

    return text.strip(), lang


async def translate_asr_json_to_docx(
    json_path: str | Path,
    out_docx: str | Path | None = None,
    *,
    # Expect your async streaming translator to be available in scope:
    # async def translate_he_or_fa_to_en_streaming(text: str, source_lang: Optional[str] = None) -> AsyncIterator[str]
    translator = None,
) -> Path:
    """
    Load ASR JSON → stream-translate (HE/FA→EN) → save a single-paragraph .docx.

    Usage:
        await translate_asr_json_to_docx("path/to/asr.json", translator=translate_he_or_fa_to_en_streaming)
    """
    if translator is None:
        raise ValueError(
            "translator callable is required (e.g., translate_he_or_fa_to_en_streaming)"
        )

    text, lang = read_asr_json_text_and_lang(json_path)

    # Collect streaming chunks
    chunks = []
    async for piece in _translate_to_stream(translator, text):
        if piece:
            chunks.append(piece)
    translated = "".join(chunks).strip()

    # Save to .docx
    out = Path(out_docx) if out_docx else Path(json_path).with_suffix(".en.docx")
    doc = docx.Document()
    doc.add_paragraph(translated)
    doc.save(out)
    return out


def translate_asr_json_to_docx_sync(
    json_path: str | Path,
    out_docx: str | Path | None = None,
    *,
    translator=None,
) -> Path:
    """
    Synchronous convenience wrapper around the async function.
    """
    return asyncio.run(
        translate_asr_json_to_docx(json_path, out_docx, translator=translator)
    )


# --- it normalizes any translator to an async-iterator of chunks ---
async def _translate_to_stream(
    translator: Callable[[str], Any],
    text: str,
) -> AsyncIterator[str]:
    """
    Accepts a translator that may be:
      - an async generator yielding str chunks
      - an async coroutine returning a full str
      - a sync function returning a full str
    and exposes a unified async iterator of str chunks.
    """
    ret = translator(text)

    # Case 1: async generator (has __aiter__)
    if hasattr(ret, "__aiter__"):
        async for chunk in ret:  # type: ignore[attr-defined]
            yield chunk
        return

    # Case 2: coroutine returning str
    if asyncio.iscoroutine(ret):
        whole = await ret  # type: ignore[func-returns-value]
        if whole:
            yield str(whole)
        return

    # Case 3: plain sync return (str or None)
    if ret:
        yield str(ret)


# --- tiny helper to pick a language label for the DOCX header if you want one ---
def _display_lang(lang: Optional[str]) -> str:
    return lang if lang else "unknown"
