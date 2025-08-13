# eng_to_persian_ollama.py
# -----------------------------------------------------------
# English → Persian (Farsi) translation with:
#  - Primary engine: local Ollama (e.g., aya:8b, qwen, gemma)
#  - Per-chunk fallback: NLLB-200 distilled (facebook/nllb-200-distilled-600M)
#  - Sentence/paragraph chunking
#  - Light Persian normalization & repetition cleanup
#
# Usage examples:
#   python eng_to_persian_ollama.py --text "Do your worst..." \
#       --engine hybrid --ollama-model aya:8b --host http://localhost:11434
#
#   python eng_to_persian_ollama.py --file path/to/english.txt --engine nllb
#
# Notes:
#  - First NLLB run will download the model (1-2 GB).
#  - Requires: pip install transformers torch ollama
#    (ollama python client: pip install ollama; server must be running)
# -----------------------------------------------------------

import argparse
import re
from typing import List

import config

# ----- Ollama (optional if you use --engine nllb only)
try:
    from ollama import Client as OllamaClient  # pip install ollama

    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

# ----- Transformers / Torch for NLLB
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _HAS_HF = True
except Exception:
    _HAS_HF = False

# -------------- Config --------------
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "aya:8b"  # adjust to your local model tag
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "pes_Arab"  # Iranian Persian

# -------------- System Prompt for Ollama --------------
SYSTEM_PROMPT = (
    "You are a professional English→Persian translator. "
    "Translate the user's English text into fluent, formal Persian (Farsi) in Arabic script. "
    "Do NOT summarize, add, omit, or rearrange. "
    "Preserve sentence boundaries and punctuation. "
    "Output PERSIAN ONLY (no English, no transliteration)."
)

# -------------- Regex helpers --------------
_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0670\u0657]"
)
_TATWEEL = re.compile(r"\u0640")
_BIDI_MARKS = re.compile(r"[\u200e\u200f]")
_MULTI_WS = re.compile(r"[ \t]+")
# collapse 3+ repeated identical Persian words/tokens
_REPEAT_TOKENS_FA = re.compile(r"(\b[\u0600-\u06FF]{2,}\b)(?:\s+\1){2,}")

_ASCII_LETTERS = re.compile(r"[A-Za-z]")
_PERSIAN_LETTERS = re.compile(r"[\u0600-\u06FF]")  # rough

# Simple sentence splitter for English
_SENT_SPLIT_EN = re.compile(r"(?<=[\.\!\?])\s+")
_PAR_SPLIT = re.compile(r"\n{2,}")


# -------------- Post-processing (Persian) --------------
def normalize_persian(text: str) -> str:
    # Arabic → Persian forms
    text = text.replace("ي", "ی").replace("ك", "ک")
    # punctuation: ASCII comma/question → Persian comma/question
    text = text.replace("?", "؟")
    # Replace commas when surrounded by Persian context
    text = re.sub(r",", "،", text)
    # remove diacritics & tatweel & bidi marks
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _BIDI_MARKS.sub("", text)
    # trim spaces
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


def light_cleanup_persian(text: str) -> str:
    s = normalize_persian(text)
    s = _REPEAT_TOKENS_FA.sub(r"\1", s)
    # remove duplicated Arabic comma spaces
    s = re.sub(r"\s*،\s*", "، ", s)
    s = re.sub(r"\s+([\.!؟])", r"\1", s)  # no space before punctuation
    s = re.sub(r"([\.!؟])([^\s])", r"\1 \2", s)  # ensure space after punctuation
    return s.strip()


def fa_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    fa = len(_PERSIAN_LETTERS.findall(text))
    total_letters = fa + len(_ASCII_LETTERS.findall(text))
    return fa / max(1, total_letters)


def looks_persian_enough(text: str, min_ratio: float = 0.65) -> bool:
    # Heuristic: mostly Persian letters and not too much ASCII
    if not text:
        return False
    if _ASCII_LETTERS.search(text):
        # allow tiny English snippets, but prefer mostly Persian
        return fa_char_ratio(text) >= min_ratio
    return True


def looks_complete_sentence(text: str) -> bool:
    return bool(text) and text.strip()[-1:] in ("۔", ".", "؟", "!", "…")


# -------------- Chunking --------------
def chunk_english(text: str, max_chars: int = 900) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    paragraphs = [p.strip() for p in _PAR_SPLIT.split(text) if p.strip()]
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sents = [s.strip() for s in _SENT_SPLIT_EN.split(para) if s.strip()]
        buf = ""
        for s in sents:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf += " " + s
            else:
                chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
    return chunks


# -------------- NLLB (HF) --------------
_nllb_tok = None
_nllb_mdl = None


def _ensure_nllb(device: str = "cpu"):
    global _nllb_tok, _nllb_mdl
    if not _HAS_HF:
        raise RuntimeError(
            "Transformers/Torch not installed. `pip install transformers torch`"
        )
    if _nllb_tok is None or _nllb_mdl is None:
        _nllb_tok = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        _nllb_mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
        _nllb_mdl.to(device)


def en_to_fa_nllb(text: str, device: str = "cpu") -> str:
    _ensure_nllb(device)
    tok = _nllb_tok
    mdl = _nllb_mdl

    # Set source language (important for NLLB)
    if hasattr(tok, "src_lang"):
        tok.src_lang = SRC_LANG

    # Target language BOS id
    try:
        bos_id = tok.lang_code_to_id[TGT_LANG]  # some versions expose this
    except Exception:
        bos_id = tok.convert_tokens_to_ids(TGT_LANG)

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    for k in inputs:
        inputs[k] = inputs[k].to(mdl.device)

    gen = mdl.generate(
        **inputs,
        forced_bos_token_id=bos_id,
        max_new_tokens=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
    )
    out = tok.batch_decode(gen, skip_special_tokens=True)[0]
    return light_cleanup_persian(out)


# -------------- Ollama (chat) --------------
def en_to_fa_ollama(client, model: str, text: str, temperature: float = 0.2) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    resp = client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.18,
            "num_predict": 2048,
            "seed": 0,
        },
    )
    out = resp["message"]["content"]
    return light_cleanup_persian(out)


def continue_if_cut_ollama(client, model: str, partial_fa: str) -> str:
    if looks_complete_sentence(partial_fa):
        return partial_fa
    cont_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": partial_fa},
        {
            "role": "user",
            "content": "ادامهٔ ترجمه از همان‌جا. متن تکراری ننویس. فقط فارسی.",
        },
    ]
    more = client.chat(model=model, messages=cont_msgs, options={"num_predict": 512})[
        "message"
    ]["content"]
    return light_cleanup_persian(partial_fa + " " + more)


# -------------- Orchestrator --------------
def translate_chunk(client, engine: str, model: str, text: str, device: str) -> str:
    """
    engine: 'ollama' | 'nllb' | 'hybrid'
    """
    if engine == "nllb":
        return en_to_fa_nllb(text, device=device)

    if engine == "ollama":
        if not _HAS_OLLAMA:
            raise RuntimeError("Ollama client not installed. `pip install ollama`")
        fa = en_to_fa_ollama(client, model, text)
        fa = continue_if_cut_ollama(client, model, fa)
        return fa

    # hybrid: try ollama per-chunk, backstop with NLLB if looks wrong
    if not _HAS_OLLAMA:
        return en_to_fa_nllb(text, device=device)

    fa = en_to_fa_ollama(client, model, text)
    if not looks_persian_enough(fa):
        # fallback for this chunk
        fa = en_to_fa_nllb(text, device=device)
    else:
        fa = continue_if_cut_ollama(client, model, fa)
    return fa


def translate_en_to_fa_chunked(
    text_en: str,
    engine: str = "hybrid",
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    max_chars: int = 900,
    device: str = "cpu",
) -> str:
    chunks = chunk_english(text_en, max_chars=max_chars)
    if not chunks:
        return ""

    client = None
    if engine in ("ollama", "hybrid") and _HAS_OLLAMA:
        client = OllamaClient(host=host)

    out: List[str] = []
    print(f"[debug] chunk_count={len(chunks)} | max_chars={max_chars}")
    if chunks:
        print(f"[debug] first_chunk_len={len(chunks[0])}")

    for i, ch in enumerate(chunks, 1):
        print(f"[debug] translating chunk {i}/{len(chunks)} (len={len(ch)})...")
        fa = translate_chunk(client, engine, model, ch, device)
        out.append(fa)

    # Join with blank lines to preserve paragraph breaks
    return "".join(out).strip()


# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser(
        description="English → Persian translator (Ollama + NLLB fallback)."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Inline English text to translate.")
    src.add_argument("--file", type=str, help="Path to UTF-8 English text file.")
    ap.add_argument(
        "--engine",
        choices=["ollama", "nllb", "hybrid"],
        default="hybrid",
        help="Which engine to use (default: hybrid).",
    )
    ap.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model tag (e.g., aya:8b, qwen2:7b, gemma2:9b).",
    )
    ap.add_argument(
        "--host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama server URL."
    )
    ap.add_argument("--max-chars", type=int, default=900, help="Max chars per chunk.")
    ap.add_argument("--device", type=str, default="cpu", help="HF device: cpu or cuda")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text_en = f.read()
    else:
        text_en = args.text

    print(f"[debug] input chars={len(text_en)}")
    print(f"[debug] first 400 of INPUT =\n\n{text_en[:400]}\n")

    fa = translate_en_to_fa_chunked(
        text_en=text_en,
        engine=args.engine,
        model=args.ollama_model,
        host=args.host,
        max_chars=args.max_chars,
        device=args.device,
    )

    print("\n" + fa)


if __name__ == "__main__":
    # main()
    text_en = """
    Do your worst. This is a Libravox recording. All Libravox recordings are in the public domain. For more information or to volunteer, please visit Libravox.org. Speech given by Winston Churchill to the House of Commons, 14 July 1941. The impressive and inspiring spectacle we have witnessed displays the vigor and efficiency of the Civil Defense Forces. They have grown up in the stress of emergency. They have been shaped and tempered by the fire of the enemy, and we saw them all, in their many grades and classes, the wardens, the rescue and first aid parties, the casualty services, the decontamination squads, the fire services, the report and control center staffs, the highway and public utility services, the messengers, the police. No one could but feel how great a people, how great a nation we have the honor to belong to. how complex, sensitive, and resilient is the society we have evolved over the centuries, and how capable of withstanding the most unexpected strain. I must, however, admit that when the storm broke in September I was for several weeks very anxious about the result. Sometimes the gas failed, sometimes the electricity. There were grievous complaints about the shelters and about conditions in them. water was cut off railways were cut or broken large districts were destroyed thousands were killed and many more thousands were wounded but there was only one thing about which there was never any doubt the courage the unconquerable grit and stamina of our people showed itself from the very outset without that all would have failed upon that rock all stood unshakable All the public services were carried on, and all the intricate arrangements, far-reaching details, involving the daily lives of so many millions, were carried out, improvised, elaborated, and perfected, in the very teeth of the cruel and devastating storm. We have to ask ourselves this question. Will the bombing attacks come back again? We have proceeded on the assumption that they will. Many new arrangements are being contrived as a result of the hard experience through which we have passed, and the many mistakes which no doubt we have made. For success is the result of making many mistakes, and learning from experience. If the lull is to end, if the storm is to renew itself, we will be ready. We will not flinch. We can take it again. We ask no favors of the enemy. We seek from them no compunction. on the contrary if to-night our people were asked to cast their vote whether a convention should be entered into to stop the bombing of cities the overwhelming majority would cry no we will mete out to them the measure and more than the measure than they have meted out to us The people, with one voice, would say, You have committed every crime under the sun. Where you have been the least resisted, there you have been the most brutal. It was you who began the indiscriminate bombing. We will have no truce or parley with you, or the grisly gang who work your wicked will. You do your worst, and we will do our best. Perhaps it may be our turn soon. Perhaps it may be our turn now. We live in a terrible epoch of the human story, but we believe there is a broad and sure justice running through its theme. It is time that the enemy should be made to suffer in their own homelands something of the torment they have let loose upon their neighbors and upon the world. We believe it to be in our power to keep this process going, on a steadily rising tide, month after month, year after year, until they are either extirpated by us, or, better still, torn to pieces by their own people. it is for this reason that i must ask you to be prepared for vehement counteraction by the enemy our methods of dealing with them have steadily improved they no longer relish their trips to our shores i do not know why they do not come but it is certainly not because they have begun to love us more It may be because they are saving up, but even if that be so, the very fact that they have to save up should give us confidence by revealing the truth of our steady advance from an almost unarmed position to superiority. But all engaged in our defense forces must prepare themselves for further heavy assaults. Your organization, your vigilance, your devotion to duty, your zeal for the cause, must be raised to the highest intensity. We do not expect to hit without being hit back, and we intend with every week that passes to hit harder. Prepare yourselves, then, my friends and comrades, for this renewal of your exertions. We shall never turn from our purpose, however somber the road, however grievous the cost, because we know that out of this time of trial and tribulation will be born a new freedom and glory for all mankind. End of speech.
    """

    print(f"[debug] input chars={len(text_en)}")
    print(f"[debug] first 500 chars of INPUT =\n{text_en[:500]}")

    hebrew = translate_en_to_fa_chunked(
        text_en,
        max_chars=300,
        model=config.OLLAMA_MODEL_TAG,  # e.g., "aya:8b"
        host=config.OLLAMA_URL,
        engine="nllb",
    )
    print(hebrew)
