import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional

import httpx

import config

# -------------------------
# Config & prompt templates
# -------------------------

# TRANSLATION_SYSTEM_PROMPT = """
# You are a professional translator. Translate **Persian (Farsi)** or **Hebrew** into clear, natural **English**.
#
# Guidelines:
# - The source text may contain **typos, ASR errors, repetitions, conversational fragments, or slang/idioms**. Translate for **intended meaning and tone**, not literal word-for-word.
# - Use surrounding **context** to resolve ambiguous phrasing; merge obviously broken fragments when needed. If still ambiguous, choose the **most plausible** reading.
# - **Correct obvious spelling/grammar errors** in the source during translation if they hinder meaning.
# - Render **slang/idioms** as their closest natural English equivalents (add a brief neutral gloss only if essential for meaning).
# - Keep **proper nouns** in Latin script; transliterate only when a widely used English form doesn’t exist.
# - Normalize numbers, dates, and units into standard English (e.g., “۳۵” → “35”, “۱۳۹۹” → “2020”, “کیلومتر” → “kilometers”).
# - Preserve speaker labels or structural markers **if present** (e.g., “Speaker A:”); otherwise do not invent them.
# - Do **not** add content, summaries, or commentary. Do **not** include the source text. Output **only** the English translation.
#
# If a segment is non-linguistic noise (e.g., music, laughter) or uninterpretable, omit it rather than guessing.
# """

TRANSLATION_SYSTEM_PROMPT = """
You are a professional Persian→Hebrew translator. 
Translate the user’s Persian text into formal modern Hebrew (no niqqud) with strict fidelity. 
Do not add, omit, interpret, summarize, or rearrange. 
Preserve sentence/verse boundaries, line breaks, punctuation, parallelism, and quotation marks from the source. 
Use established Hebrew forms for proper names and religious terms; when uncertain, transliterate into standard Hebrew. 
Convert Persian numerals to standard numerals consistently. 
If any word or segment is unclear, write [בלתי ברור] at that spot rather than guessing. 
Output Hebrew text only, with no explanations, notes, or back-translation."""

# Deterministic generation options for Ollama /api/chat
# DETERMINISTIC_OPTS = {
#     "temperature": 0.0,
#     "top_p": 0.0,
#     "seed": 0,
#     "repeat_penalty": 1.0,
#     "num_predict": 512,  # cap tokens to reduce drift; bump if you need longer outputs
#     # You can also set "stop": [] here if you use any custom stops.
# }

DETERMINISTIC_OPTS = {
    "temperature": 0.2,  # small, not zero
    "top_p": 0.9,  # allow some diversity
    "seed": 0,
    "repeat_penalty": 1.2,  # or 1.15–1.3
    "num_predict": 1536,  # give room for full verse; tune as needed
}

# -------------------------
# Utility: safe chunking
# -------------------------


def _rough_token_len(s: str) -> int:
    """Very rough token estimate: ~4 chars per token fallback."""
    # If you already use a tokenizer, plug it here. This heuristic avoids extra deps.
    return max(1, math.ceil(len(s) / 4))


def chunk_text(
    text: str,
    max_out_tokens: int = 512,
    max_in_tokens: int = 1024,
) -> List[str]:
    """
    Split a long input into chunks that respect a crude token budget.
    This helps avoid model/context limits deterministically.
    """
    if _rough_token_len(text) <= max_in_tokens:
        return [text]

    # Split on paragraph boundaries first, then sentences as needed.
    parts: List[str] = []
    for para in text.split("\n\n"):
        if not para.strip():
            continue
        if _rough_token_len(para) <= max_in_tokens:
            parts.append(para)
        else:
            # fallback: sentence-ish split
            buf = []
            buf_len = 0
            for seg in re.split(r"(?<=[\.!?])\s+", para.strip()):
                seg_tokens = _rough_token_len(seg)
                if buf_len + seg_tokens > max_in_tokens and buf:
                    parts.append(" ".join(buf))
                    buf, buf_len = [seg], seg_tokens
                else:
                    buf.append(seg)
                    buf_len += seg_tokens
            if buf:
                parts.append(" ".join(buf))
    return parts


# ----------------------------------------
# Async streaming client for Ollama /api/chat
# ----------------------------------------


@dataclass
class OllamaClientConfig:
    base_url: str = config.OLLAMA_URL
    model: str = config.OLLAMA_MODEL_TAG
    options: Optional[dict] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_backoff: float = 1.5  # seconds multiplier


class OllamaTranslator:
    """
    Minimal async streaming translator client against Ollama's /api/chat.
    Deterministic options are the default; you can override via config.options.
    """

    def __init__(self, config: Optional[OllamaClientConfig] = None):
        self.cfg = config or OllamaClientConfig()
        self._options = dict(DETERMINISTIC_OPTS)
        if self.cfg.options:
            self._options.update(self.cfg.options)

    def _build_messages(self, text: str, lang_hint: Optional[str] = None) -> List[dict]:
        sys = TRANSLATION_SYSTEM_PROMPT
        if lang_hint:
            sys += f"\nLanguage hint: {lang_hint}"
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": text},
        ]

    async def _chat_stream_once(
        self,
        client: httpx.AsyncClient,
        text: str,
        lang_hint: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Single attempt streaming call. Yields content chunks as they arrive.
        """
        payload = {
            "model": self.cfg.model,
            "messages": self._build_messages(text, lang_hint),
            "stream": True,
            "options": self._options,
        }
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        async with client.stream(
            "POST", url, json=payload, timeout=self.cfg.timeout
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # Ollama streams JSON lines like: {"message":{"role":"assistant","content":"..."},"done":false}
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "message" in event and "content" in event["message"]:
                    yield event["message"]["content"]
                # You can also inspect "done","total_duration" etc if needed.

    async def chat_stream(
        self,
        text: str,
        lang_hint: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Streaming with minimal retry on network / HTTP errors.
        """
        attempt = 0
        backoff = self.cfg.retry_backoff
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    async for chunk in self._chat_stream_once(client, text, lang_hint):
                        yield chunk
                    return
                except (httpx.HTTPError,):
                    attempt += 1
                    if attempt > self.cfg.max_retries:
                        raise
                    await asyncio.sleep(backoff)
                    backoff *= self.cfg.retry_backoff

    async def translate_text(
        self,
        text: str,
        lang_hint: Optional[str] = None,
        max_in_tokens: int = 1024,
        max_out_tokens: int = 512,
        join_with: str = "\n\n",
    ) -> str:
        """
        Deterministic translation of (possibly long) text by chunking + streaming.
        Returns the concatenated translation.
        """
        chunks = chunk_text(
            text, max_out_tokens=max_out_tokens, max_in_tokens=max_in_tokens
        )

        results: List[str] = []
        for ch in chunks:
            buf = []
            async for part in self.chat_stream(ch, lang_hint=lang_hint):
                buf.append(part)
            results.append("".join(buf).strip())

        # Join chunks with paragraph separators to preserve structure
        return join_with.join(results).strip()


# -------------------------
# Convenience wrappers
# -------------------------


async def translate_he_or_fa_to_en_streaming(
    text: str,
    model: str = config.OLLAMA_MODEL_TAG,
    base_url: str = config.OLLAMA_URL,
    lang_hint: Optional[str] = None,  # "fa" | "he" | None
    max_in_tokens: int = 1024,
    max_out_tokens: int = 512,
    options: Optional[dict] = None,
) -> str:
    """
    High-level call: translate Persian/Hebrew → English.
    Deterministic options; async streaming under the hood; token-budget aware.
    """
    client = OllamaTranslator(
        OllamaClientConfig(base_url=base_url, model=model, options=options)
    )
    return await client.translate_text(
        text=text,
        lang_hint=lang_hint,
        max_in_tokens=max_in_tokens,
        max_out_tokens=max_out_tokens,
    )


async def translate_many_streaming(
    texts: Iterable[str],
    model: str = config.OLLAMA_MODEL_TAG,
    base_url: str = config.OLLAMA_URL,
    lang_hint: Optional[str] = None,
    concurrency: int = 4,
    max_in_tokens: int = 1024,
    max_out_tokens: int = 512,
    options: Optional[dict] = None,
) -> List[str]:
    """
    Parallel (bounded) translation of multiple texts. Deterministic settings carried through.
    """

    sem = asyncio.Semaphore(concurrency)
    client = OllamaTranslator(
        OllamaClientConfig(base_url=base_url, model=model, options=options)
    )

    async def _one(t: str) -> str:
        async with sem:
            return await client.translate_text(
                text=t,
                lang_hint=lang_hint,
                max_in_tokens=max_in_tokens,
                max_out_tokens=max_out_tokens,
            )

    return await asyncio.gather(*[_one(t) for t in texts])


# -------------------------
# Example usage (keep or move to your runner)
# -------------------------
if __name__ == "__main__":
    # input_file_name = "response_1730994879150_output-farsi.mp3_16k_no_llm_cleaned_llm_from_no_llm.json"
    input_file_name = "farsi_firstsource.mp3_16k.json"
    input_file_path = "/Users/pouya/PycharmProjects/SmartSummary/src/transcripts_json/"
    input = input_file_path + "/" + input_file_name

    from src.translate_helpers import (read_asr_json_text_and_lang,
                                       translate_asr_json_to_docx_sync)

    # 1) Just read text + language
    text, lang = read_asr_json_text_and_lang(input)
    print(lang, text[:200])

    # 2) Translate and save as DOCX (sync helper)
    out_docx = translate_asr_json_to_docx_sync(
        input,
        translator=translate_he_or_fa_to_en_streaming,
    )
    print("Saved:", out_docx)
