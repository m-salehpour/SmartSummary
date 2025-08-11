#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# summarise.py   ·  one‑shot document summary   (cached)
# ----------------------------------------------------------------------
# • Reads a WhisperX JSON (list or {"segments":[…]}).
# • Concats all segment texts (cropped to 12 000 chars for context).
# • Asks an LLM for a < 200‑word Persian summary + key terms list.
# • Caches the result by MD5 of the whole file: .cache_summary/<hash>.json
# ──────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from ollama import AsyncClient
import config

# ── coloured logging symbols (same set used elsewhere) ────────────────
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN  = "\033[96m"; GREY   = "\033[90m"; RESET = "\033[0m"
CHECK = "✔"; CROSS = "✗"; WARN = "⚠️"; INFO = "ℹ️"

# ── prompt template ───────────────────────────────────────────────────
SUMMARY_PROMPT = (
    "📝 لطفاً در حداکثر ۲۰۰ کلمه، خلاصه‌ای رسمی از گفت‌وگو بنویس و تمام نام‌ها، "
    "مفاهیم تخصصی یا اصطلاحاتِ مهم را فهرست کن.\n"
)


SUMMARY_PROMPT_ENG = (
    "📝 ENGLISH ONLY. In ≤200 words, write a formal summary of the text. "
    "Then list all proper names, technical concepts, and key terms. "
    "Be neutral and concise. Do not quote long passages. "
    "Output must be in English even if the source text is Persian.\n\n"
    "## Summary\n"
    "{summary}\n\n"
    "## Key Terms\n"
    "- <Term 1>\n"
    "- <Term 2>\n"
    "- <Term 3>\n"
)

# ── cache directory (created on first run) ────────────────────────────
CACHE_DIR = Path(".cache_summary")
CACHE_DIR.mkdir(exist_ok=True)

# ======================================================================
# async helper → calls Ollama once and returns plain‑text summary
# ======================================================================
def run_coro(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

async def _summarise(text: str, model: str = config.OLLAMA_MODEL_TAG) -> str:
    client = AsyncClient()
    rsp = await client.chat(
        model    = model,
        messages = [
            {"role": "user", "content": SUMMARY_PROMPT + "\n" + text}
        ],
        options  = {"temperature": 0.2},
    )
    return rsp["message"]["content"].strip()

# ======================================================================
# public sync helper (used by pipeline_clean.py)
# returns {"summary": "..."}  ← same shape as before
# ======================================================================
def get_summary(transcript: Path, *, model: str = config.OLLAMA_MODEL_TAG) -> Dict[str, Any]:
    """Return cached or freshly‑generated summary for a WhisperX JSON file."""
    raw_text = transcript.read_text(encoding="utf-8")
    file_hash = hashlib.md5(raw_text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{file_hash}.json"

    # ---------- cache hit ------------------------------------------------
    if cache_file.exists():
        print(f"{GREEN}{CHECK}{RESET} summary cache hit → {cache_file}")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    # ---------- need fresh summary --------------------------------------
    print(f"{YELLOW}{WARN}{RESET} no cache — generating summary with {model} …")

    # safe JSON parse
    try:
        data = json.loads(raw_text)
        segments = data["segments"] if isinstance(data, dict) else data
        assert isinstance(segments, list), "unexpected JSON shape"
    except Exception as e:
        print(f"{RED}{CROSS}{RESET} failed to parse JSON: {e}")
        sys.exit(1)

    # concat & trim to keep within context length
    joined = " ".join(s.get("text", "") for s in segments)[:12_000]

    # run LLM (synchronously from caller PoV)
    # summary = asyncio.run(_summarise(joined, model=model))
    summary = run_coro(_summarise(joined, model=model))

    # store & report
    out: Dict[str, Any] = {"summary": summary}
    cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{GREEN}{CHECK}{RESET} summary written → {cache_file}")
    print(f"{CYAN}{INFO}{RESET} preview:\n{GREY}{summary[:300]}…{RESET}")

    return out
