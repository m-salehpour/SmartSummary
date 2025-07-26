#!/usr/bin/env python3
# normalise_segments.py
#
# Clean each WhisperX segment with a *context‑aware* prompt:
#   EXAMPLE_BLOCK      ← static, high‑precision rules
#   DYNAMIC_EXAMPLES   ← 2 best matches from previous segments
#   SUMMARY            ← one‑paragraph topic overview
#
#   « RULES »          ← always‑on bullet list
#
# The module is fully self‑contained — only requires `ollama`.

import asyncio, re, difflib, textwrap, time
from typing import List, Dict
from ollama import AsyncClient
import config

# ─────────────────────────────────────────────────────────────────────
PARA_MAX   = 2       # max concurrent requests – keep memory footprint low
N_RETRIEVE = 2       # how many “similar” past sentences we reuse

MODEL_TAG  = config.OLLAMA_MODEL_TAG      # tweak freely
OLLAMA_URL = config.OLLAMA_URL

START_PROMPT = """
    You are a Persian text normalizer.

    ● وظیفهٔ ما: اصلاحِ املاء، فاصله‌گذاری، نشانه‌گذاری و تبدیلِ گفتارِ محاوره‌ای به نثرِ رسمی.
    ● ترجمه ممنوع است؛ فقط متنِ فارسی را تمیز کن.
    ● خلاصه‌سازی یا حذف جمله‌ها ممنوع؛ تا می‌شود به متنِ اصلی وفادار بمان.
    ● خروجی باید دقیقاً یک جملهٔ تمیزِ فارسی باشد، بدون هیچ کد، حاشیه یا فرمت اضافی.
    """
# Markdown‑style example block (same as before, truncated for brevity)
EXAMPLE_BLOCK = textwrap.dedent("""\
<examples>
RAW: تاریخ اجرا یکشنبه سیزدهم آوریل 2017 …
CLEAN: تاریخ اجرا یکشنبه سیزدهم آوریل ۲۰۲۵ …
⋮   (keep all your good pairs here)
</examples>
""")

RULES = (
    "🔹 غلط املایی، فاصله‌گذاری، نشانه‌گذاری و تبدیل گفتار محاوره‌ای ↦ نثر رسمی.\n"
    "🔹 ترجمه یا خلاصه‌سازی نکن؛ به متن اصلی وفادار بمان.\n"
    "🔹 خروجی = فقط متن تمیزِ فارسی.\n"
)

# ─────────────────────────────────────────────────────────────────────
class SegmentCleaner:
    """
    Clean a list of WhisperX segments *in‑place* (adds ``fa_clean``).
    """

    def __init__(self, summary: str):
        self.summary    = summary.strip()
        self.client     = AsyncClient(host=OLLAMA_URL)
        self.sem        = None
        self.done: List[str] = []          # already‑cleaned texts (for retrieval)

    # ---------- helpers ------------------------------------------------
    @staticmethod
    def _strip(llm_out: str) -> str:
        llm_out = llm_out.strip()
        if llm_out.startswith("```"):
            llm_out = re.sub(r"^```.*?\n|\n```$", "", llm_out, flags=re.DOTALL).strip()
        llm_out = re.sub(r'^["“]|["”]$', "", llm_out).strip()
        llm_out = re.sub(r"\s+", " ", llm_out)
        return llm_out

    def _nearest_examples(self, current: str) -> str:
        if not self.done:
            return ""                      # first few segments → nothing yet
        # crude similarity: longest common subsequence length
        scored = [
            (difflib.SequenceMatcher(None, current, prev).ratio(), prev)
            for prev in self.done
        ]
        best = [txt for _, txt in sorted(scored, reverse=True)[:N_RETRIEVE]]
        if not best:
            return ""
        out = ""
        for b in best:
            out += f"RAW: {b}\nCLEAN: {b}\n"
        return "<dynamic>\n" + out + "</dynamic>\n"

    # ---------- single‑segment cleaning --------------------------------
    async def _clean_one(self, seg: Dict) -> None:
        if self.sem is None:  # first coroutine → now loop exists
            self.sem = asyncio.Semaphore(PARA_MAX)

        idx  = seg.get("_idx", "?")
        user = seg["text"].strip()

        # compose prompt
        dyn_examples = self._nearest_examples(user)
        sys_prompt = (
            START_PROMPT +
            EXAMPLE_BLOCK +
            dyn_examples +
            f"\nخلاصهٔ بافت:\n{self.summary}\n" +
            "────────────\n" + RULES
        )

        async with self.sem:
            for attempt in range(1, 4):
                t0 = time.time()
                print(f"🔸 [{idx:03}] attempt {attempt} — sending to {MODEL_TAG} …")
                try:
                    # stream tokens for nicer UX (but we collect to a string)
                    stream = await self.client.chat(
                        model    = MODEL_TAG,
                        stream   = True,
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user",   "content": user}
                        ],
                        options  = {"temperature": 0.0}
                    )
                    tokens = []
                    async for part in stream:
                        tokens.append(part["message"]["content"])
                    raw = "".join(tokens)
                    clean = self._strip(raw)
                    dt = time.time() - t0
                    if clean:
                        print(f"✅ [{idx:03}] ok in {dt:.2f}s → {clean!r}")
                        seg["fa_clean"] = clean
                        self.done.append(clean)
                        return
                    print(f"⚠️  [{idx:03}] empty reply")
                except Exception as e:
                    print(f"❌ [{idx:03}] {type(e).__name__}: {e}")

                await asyncio.sleep(attempt)   # linear back‑off

        # after three failures
        print(f"🟥 [{idx:03}] giving up, keep original")
        seg["fa_clean"] = user
        self.done.append(user)

    # ---------- public: clean list -------------------------------------
    async def clean_all(self, segments: List[Dict]) -> None:
        print(f"🚀 cleaning {len(segments)} segments "
              f"(parallel={PARA_MAX}, model={MODEL_TAG})")
        await asyncio.gather(*(self._clean_one(s) for s in segments))
