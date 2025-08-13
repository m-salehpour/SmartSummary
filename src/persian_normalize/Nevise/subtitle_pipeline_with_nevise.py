"""subtitle_pipeline_with_nevise.py
A single-file pipeline that
1. loads WhisperX JSON segments
2. cleans Persian text with **Hazm** + **Nevise** spell-checker
3. sends every segment to a local **Mixtral-Instruct** model (Ollama) for English translation
4. writes perfectly-timed **.srt** subtitles.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from helpers import (batch_iter, bert_tokenize_for_valid_examples,
                     get_model_nparams, labelize, load_vocab_dict,
                     untokenize_without_unks)
from ollama import AsyncClient
from tqdm.asyncio import tqdm_asyncio

import utils
from hazm import Normalizer
from models import SubwordBert
from utils import get_sentences_splitters

# ──────────────────────────────────────────────────────────────────────
# 1 « Nevise » spell-checker glue (chatty)
# ──────────────────────────────────────────────────────────────────────


def _load_nevise(vocab_path: Path, ckpt_path: Path, device: str):
    vocab = load_vocab_dict(str(vocab_path))
    model = SubwordBert(
        3 * len(vocab["chartoken2idx"]),
        vocab["token2idx"][vocab["pad_token"]],
        len(vocab["token_freq"]),
    )
    print(
        f"[Nevise] Loaded SubwordBert model with {get_model_nparams(model):,} parameters"
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"[Nevise] Checkpoint '{ckpt_path.name}' loaded onto {device}")
    return model, vocab


@torch.inference_mode()
def _nevise_correct(
    model: SubwordBert, vocab: dict, device: str, hazm_norm: Normalizer, text: str
) -> str:
    print(f"\n[Nevise] Original sentence: {text!r}")
    sub_sents, _ = get_sentences_splitters(text)
    sub_sents = [
        hazm_norm.normalize(utils.space_special_chars(s))
        for s in sub_sents
        if s.strip()
    ]
    if not sub_sents:
        print("[Nevise] No sub-sentences found, returning original")
        return text.strip()

    fixed = []
    for batch_labels, batch_sentences in batch_iter(
        [(s, s) for s in sub_sents], batch_size=8, shuffle=False
    ):
        print(f"[Nevise] Processing batch: {batch_sentences}")
        b_lbl, b_sent, bert_inp, bert_split = bert_tokenize_for_valid_examples(
            batch_labels, batch_sentences
        )
        if not b_lbl:
            print("[Nevise]  → Skipped (tokenization mismatch)")
            fixed.extend(batch_sentences)
            continue
        bert_inp = {k: v.to(device) for k, v in bert_inp.items()}
        lbl_ids, lens = labelize(b_lbl, vocab)
        lbl_ids, lens = lbl_ids.to(device), lens.to(device)
        loss, preds = model(bert_inp, bert_split, targets=lbl_ids, topk=1)
        preds_txt = untokenize_without_unks(preds, lens.cpu().numpy(), vocab, b_sent)
        for orig, corr in zip(b_sent, preds_txt):
            print(f"[Nevise]   → '{orig}'  →  '{corr}'")
        fixed.extend(preds_txt)

    joined = utils.de_space_special_chars(" ".join(fixed))
    cleaned = re.sub(r"\s+", " ", joined).strip() or text.strip()
    print(f"[Nevise] Final cleaned: {cleaned!r}")
    return cleaned


class NeviseCorrector:
    """Facade for Nevise spell-checking."""

    def __init__(self, vocab_path: Path, ckpt_path: Path, device: str = "auto"):
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        print(f"[NeviseCorrector] Using device: {self.device}")
        self.hazm_norm = Normalizer(
            correct_spacing=True, persian_numbers=True, remove_diacritics=True
        )
        self.model, self.vocab = _load_nevise(vocab_path, ckpt_path, self.device)

    def clean(self, sentence: str) -> str:
        return _nevise_correct(
            self.model, self.vocab, self.device, self.hazm_norm, sentence
        )


# ──────────────────────────────────────────────────────────────────────
# 2 Mixtral translator (Ollama) — chatty
# ──────────────────────────────────────────────────────────────────────

import json
import re
from typing import Dict


class MixtralTranslator:
    SYSTEM_PROMPT = (
        "You are an expert Persian linguist and translator. "
        "**Important Note:** You do NOT append any explanations, notes, or extra text."
        "1) Output exactly one JSON object, nothing else. "
        "2) Use keys `fa_clean` and `en` ONLY. "
        "3) Do NOT append any explanations, notes, or extra text."
        "**Important Note:** Do NOT append any explanations, notes, or extra text.\n\n"
        "Example:\n"
        "```json\n"
        '{"fa_clean":"سلام","en":"Hello"}\n'
        "```"
    )

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(
        self, model_tag: str = "mixtral:instruct", host: str = "http://localhost:11434"
    ):
        self.client = AsyncClient(host=host)
        self.model_tag = model_tag
        print(
            f"[Translator] Initialized Ollama client for model '{model_tag}' at {host}"
        )

    async def translate(self, segment: Dict) -> Dict:
        idx = segment.get("_idx", "?")
        payload = json.dumps(
            {
                "speaker": segment.get("speaker", "SP"),
                "start": segment["start"],
                "end": segment["end"],
                "fa": segment["fa_clean"],
            },
            ensure_ascii=False,
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            print(
                f"\n[Translator] Segment {idx}: attempt {attempt}/{self.MAX_RETRIES}…"
            )
            rsp = await self.client.chat(
                model=self.model_tag,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                options={"temperature": 0.2},
            )

            raw = rsp["message"]["content"]
            print(f"[Translator] ▶ raw reply (len={len(raw)}): {raw!r}")

            # auto-close truncated JSON
            txt = raw.strip()
            if txt.startswith("{") and not txt.endswith("}"):
                raw = raw.rstrip() + "}"
                print("[Translator] ⚠️ Appended closing '}'")

            # extract first {...} block
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not m:
                print(f"[Translator] ❌ No JSON found on attempt {attempt}")
            else:
                json_block = m.group(0).replace(r"\_", "_")
                print(f"[Translator] ▶ JSON block candidate:\n{json_block!r}")

                try:
                    data = json.loads(json_block)
                except json.JSONDecodeError as e:
                    print(f"[Translator] ❌ JSON parse error on attempt {attempt}: {e}")
                else:
                    # success!
                    fa_clean = data.get("fa_clean", segment["fa_clean"])
                    en = data.get("en", "")
                    print(f"[Translator] ✔ fa_clean: {fa_clean!r}")
                    print(f"[Translator] ✔ en      : {en!r}")
                    segment["fa_final"] = fa_clean
                    segment["en"] = en
                    return segment

            # if we get here, this attempt failed
            if attempt < self.MAX_RETRIES:
                print(f"[Translator] ⏳ Waiting {self.RETRY_DELAY}s before retry…")
                await asyncio.sleep(self.RETRY_DELAY)
                return None
            else:
                print(
                    f"[Translator] ❌ All {self.MAX_RETRIES} attempts failed; skipping segment {idx}"
                )
                segment["fa_final"] = segment["fa_clean"]
                segment["en"] = ""
                return segment
        return None


# ──────────────────────────────────────────────────────────────────────
# 3 helpers
# ──────────────────────────────────────────────────────────────────────


def ts(sec: float) -> str:
    return str(timedelta(seconds=float(sec)))[:12].replace(".", ",").zfill(12)


# ──────────────────────────────────────────────────────────────────────
# 4 pipeline (with index tagging for chatty logs)
# ──────────────────────────────────────────────────────────────────────


def run_pipeline(
    transcript_path: Path, output_srt: Path, nevise_ckpt: Path, vocab_path: Path
):
    print("🗂  Loading WhisperX segments…")
    raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        segments = raw["segments"]
    elif isinstance(raw, list):
        segments = raw
    else:
        raise ValueError("Transcript JSON must be list or contain 'segments'.")

    # tag each with an index for logging
    for i, seg in enumerate(segments):
        seg["_idx"] = i + 1

    print("🔎  Spell-checking Persian with Nevise…")
    corrector = NeviseCorrector(vocab_path, nevise_ckpt)
    for seg in segments:
        idx = seg["_idx"]
        orig = seg.get("text", "")
        print(f"\n[Pipeline] Segment {idx}: original fa: {orig!r}")
        cleaned = corrector.clean(orig)
        seg["fa_clean"] = cleaned
        print(f"[Pipeline] Segment {idx}: stored fa_clean: {cleaned!r}")

    print("\n🌐  Translating with Mixtral…")
    translator = MixtralTranslator()

    async def _go():
        tasks = [translator.translate(seg) for seg in segments]
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            await fut

    asyncio.run(_go())

    print("\n💾  Writing SRT to", output_srt)
    with output_srt.open("w", encoding="utf-8") as f:
        for seg in segments:
            idx = seg["_idx"]
            # ensure we always have an "en" key
            en = seg.get("en", "")
            f.write(f"{idx}\n")
            f.write(f"{ts(seg['start'])} --> {ts(seg['end'])}\n")
            f.write(f"» {seg.get('speaker', 'SP')}: {en}\n\n")
            print(f"[Pipeline] Wrote subtitle {idx}: {en!r}")

    print("\n✅  All done! Subtitles at", output_srt)


# ──────────────────────────────────────────────────────────────────────
# 5 CLI + example
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage:\n  python subtitle_pipeline_with_nevise.py transcript.json english_subs.srt",
            file=sys.stderr,
        )
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    output_srt = Path(sys.argv[2])
    nevise_ckpt = Path("model/model.pth.tar")
    vocab_path = Path("model/vocab.pkl")

    start = time.time()
    run_pipeline(transcript_path, output_srt, nevise_ckpt, vocab_path)
    print(f"\n⏱  total elapsed: {time.time() - start:.1f}s")
