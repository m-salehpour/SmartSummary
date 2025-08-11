#!/usr/bin/env python3
# eng_to_he_ollama.py
# Single-file EN->HE translator with Ollama-first + NLLB fallback, chunking, and light postprocessing.

import argparse, re, sys, os
from typing import List, Optional

# ---- optional local config (graceful fallback if missing) --------------------
try:
    import config  # expected fields: OLLAMA_URL, OLLAMA_MODEL_TAG
    OLLAMA_URL = getattr(config, "OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL_TAG = getattr(config, "OLLAMA_MODEL_TAG", "gemma3:4b-it-q4_K_M")
except Exception:
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL_TAG = os.environ.get("OLLAMA_MODEL_TAG", "gemma3:4b-it-q4_K_M")

# ---- optional Ollama client (only needed if --method auto/ollama) ------------
try:
    from ollama import Client as OllamaClient
except Exception:  # pragma: no cover
    OllamaClient = None

# ---- optional HF transformers (only needed if --method auto/nllb) ------------
_OPUS_AVAIL = False
_NLLB_AVAIL = True
_nllb_tok = None
_nllb_mdl = None

# Model: smallest reasonable NLLB for 8GB setups
_NLLB_MDL = "facebook/nllb-200-distilled-600M"
_NLLB_SRC = "eng_Latn"
_NLLB_TGT = "heb_Hebr"

# ------------------------------------------------------------------------------
#                           Post-processing / Guards
# ------------------------------------------------------------------------------

_HEB_DIACRITICS = re.compile(r'[\u0591-\u05C7]')   # niqqud/ta'amim
_BIDI_MARKS     = re.compile(r'[\u200e\u200f]')     # LRM/RLM
_REPEAT_TOKENS  = re.compile(r'(\b[\u05D0-\u05EA]{2,}\b)(?:\s+\1){2,}')
_LATIN_CHARS    = re.compile(r'[A-Za-z]+')

def postprocess_hebrew(s: str) -> str:
    # remove diacritics & bidi marks, collapse repeats, normalize spaces
    s = _HEB_DIACRITICS.sub('', s)
    s = _BIDI_MARKS.sub('', s)
    s = _REPEAT_TOKENS.sub(r'\1', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = s.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return s.strip()

def hebrew_ratio(s: str) -> float:
    if not s: return 0.0
    heb = len(re.findall(r'[\u0590-\u05FF]', s))
    return heb / max(1, len(s))

def looks_complete(s: str) -> bool:
    return bool(s) and s[-1] in '.!?״”’"'

def looks_like_refusal_or_chatter(s: str) -> bool:
    # crude blockers for “I can’t help” / small-talk slip
    bad_fragments = [
        "אני לא יכול", "אני לא יכולה", "לא יכול/ה", "לא אוכל לעזור",
        "מה שלומך", "מזג האוויר", "שנה טובה", "תודה שפנית", "אני מקווה"
    ]
    t = s.strip()
    return any(b in t for b in bad_fragments)

# ------------------------------------------------------------------------------
#                                   Chunking
# ------------------------------------------------------------------------------

# sentence-ish split (English), fallback to paragraphs, then hard-wrap
_SENT_SPLIT_EN = re.compile(r'(?<=[\.\!\?])\s+')

def chunk_english(text: str, max_chars: int = 450) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    for para in re.split(r'\n{2,}', text):
        p = para.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            chunks.append(p); continue
        sents = [s.strip() for s in _SENT_SPLIT_EN.split(p) if s.strip()]
        buf = ""
        for s in sents:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf += " " + s
            else:
                chunks.append(buf); buf = s
        if buf:
            chunks.append(buf)

    # final safety: if any chunk still too long, hard-wrap
    out: List[str] = []
    for ch in chunks:
        if len(ch) <= max_chars:
            out.append(ch); continue
        start = 0
        while start < len(ch):
            out.append(ch[start:start+max_chars])
            start += max_chars
    return out

# ------------------------------------------------------------------------------
#                             NLLB (HF) Fallback
# ------------------------------------------------------------------------------

def _ensure_nllb(device: str = "cpu"):
    global _nllb_tok, _nllb_mdl
    if _nllb_tok is not None and _nllb_mdl is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _nllb_tok = AutoTokenizer.from_pretrained(_NLLB_MDL)
    _nllb_mdl = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MDL).to(device)

def en_to_he_nllb(text_en: str, device: str = "cpu", max_new_tokens: int = 512) -> str:
    from transformers import GenerationConfig
    _ensure_nllb(device)
    _nllb_tok.src_lang = _NLLB_SRC
    enc = _nllb_tok(
        text_en, return_tensors="pt", padding=False, truncation=True
    ).to(device)

    # robust way to get target BOS id across transformers versions
    try:
        bos_id = _nllb_tok.lang_code_to_id[_NLLB_TGT]
    except Exception:
        bos_id = _nllb_tok.convert_tokens_to_ids(_NLLB_TGT)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    out_tokens = _nllb_mdl.generate(
        **enc,
        forced_bos_token_id=bos_id,
        generation_config=gen_cfg
    )
    heb = _nllb_tok.batch_decode(out_tokens, skip_special_tokens=True)[0]
    return postprocess_hebrew(heb)

# ------------------------------------------------------------------------------
#                                Ollama (LLM) path
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a professional English➔Hebrew translator. "
    "Translate the user's English text into formal modern Hebrew WITHOUT niqqud. "
    "Do not add, omit, summarize, interpret, or comment. "
    "Use neutral register; keep punctuation and line breaks. "
    "Output HEBREW ONLY."
)

def translate_chunk_ollama(client, model: str, chunk_en: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": chunk_en},
    ]
    resp = client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.18,
            "seed": 0,
            "num_predict": 1500,
        }
    )
    he = resp["message"]["content"]
    he = postprocess_hebrew(he)

    # if cut mid-sentence, nudge a short continuation (2 tries max)
    tries = 0
    while not looks_complete(he) and tries < 2:
        tries += 1
        cont = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": he},
                {"role": "user", "content": "המשך את התרגום מן הנקודה המדויקת. "
                                             "אל תחזור על טקסט קודם. עברית בלבד, בלי ניקוד."}
            ],
            options={"temperature": 0.2, "repeat_penalty": 1.2, "num_predict": 400, "seed": 0}
        )["message"]["content"]
        he = postprocess_hebrew(he + " " + cont)

    return he

# nllb_hebrew_lightfix.py
# Minimal post-editor for Hebrew MT output (NLLB etc.)
import re
from typing import List

# --- basic normalizations ---
_BIDI = re.compile(r'[\u200e\u200f]')
_SPACES_MULTI = re.compile(r'[ \t]{2,}')
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([,.;:?!])')
_SMART_QUOTES = str.maketrans({'“':'"', '”':'"', '„':'"', '‟':'"', '’':"'", '‚':"'"})

# Common mistranslations / clunky phrases seen in NLLB outputs for this domain
# Keep it conservative: exact phrases or safe contexts only.
_REPLACEMENTS: List[tuple[re.Pattern, str]] = [
    # LibriVox + domain
    (re.compile(r'\bLibravox\b', re.IGNORECASE), 'LibriVox'),
    (re.compile(r'\bLibravax\b', re.IGNORECASE), 'LibriVox'),
    (re.compile(r'LibriVox[.\s]*org', re.IGNORECASE), 'LibriVox.org'),
    # Services / roles
    (re.compile(r'\bהממלטים\b'), 'המקלטים'),
    (re.compile(r'\bשירותים?\s*אש\b'), 'שירותי כבאות'),
    (re.compile(r'\bמשרדי\s+ניקוי\s+זיהום\b'), 'צוותי טיהור מזיהום'),
    (re.compile(r'\bצוות\s+מרכז\s+האבטחה\b'), 'מרכזי דיווח ובקרה'),
    (re.compile(r'\bשירותי\s+פצועים\b'), 'שירותי טיפול בנפגעים'),
    # Word choices / inflections
    (re.compile(r'\bתקפות\b'), 'התקפות'),
    (re.compile(r'\bהמים\s+נחתקו\b'), 'המים נותקו'),
    (re.compile(r'\bמחוזים\b'), 'מחוזות'),
    (re.compile(r'\bהחוצפה\s+וה?יעילות\b'), 'המרץ והיעילות'),
    (re.compile(r'\bנבנו\s+ונדרמו\b'), 'עוצבו והוקשחו'),
    (re.compile(r'\bהופעה\b'), 'המופע'),  # soft: “the show/spectacle”
    # Minor style tweaks found often
    (re.compile(r'\bאנו\s+לא\b'), 'איננו'),
    (re.compile(r'\bלא\s+נתרעש\b'), 'לא נירתע'),
]

# Split on sentence ends (., !, ?, hebrew sof pasuq if present) or hard newlines.
_SENT_SPLIT = re.compile(r'(?<=[\.!?])\s+|\n+')

def split_sentences(text: str) -> List[str]:
    s = text.strip()
    if not s:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(s)]
    return [p for p in parts if p]

def collapse_exact_repeats(text: str, min_len: int = 30) -> str:
    """Remove immediate duplicate sentences (exact string match), keeping rhetoric intact."""
    out = []
    last = None
    for sent in split_sentences(text):
        if last is not None and len(sent) >= min_len and sent == last:
            # skip exact immediate duplicate
            continue
        out.append(sent)
        last = sent
    # Re-stitch with spacing that respects Hebrew punctuation.
    return ' '.join(out)

def light_replace(text: str) -> str:
    s = text.translate(_SMART_QUOTES)
    s = _BIDI.sub('', s)
    for pat, rep in _REPLACEMENTS:
        s = pat.sub(rep, s)
    # spacing & punctuation
    s = _SPACE_BEFORE_PUNCT.sub(r'\1', s)   # remove space before ,.;:?!
    s = _SPACES_MULTI.sub(' ', s)
    # Normalize weird doubled punctuation like ".." -> "."
    s = re.sub(r'([.!?])\1+', r'\1', s)
    return s.strip()

def lightfix_hebrew(text: str, dedupe: bool = True) -> str:
    s = light_replace(text)
    if dedupe:
        s = collapse_exact_repeats(s)
    # Tiny cleanups after rejoin
    s = _SPACES_MULTI.sub(' ', s)
    s = s.strip()
    return s


# ------------------------------------------------------------------------------
#                           Orchestrator (auto / choice)
# ------------------------------------------------------------------------------

def translate_en_to_he_chunked(
    text_en: str,
    method: str = "auto",          # "auto" | "ollama" | "nllb"
    model: Optional[str] = None,   # Ollama model tag
    host: Optional[str] = None,    # Ollama URL
    device: str = "cpu",
    max_chars: int = 450
) -> str:
    model = model or OLLAMA_MODEL_TAG
    host = host or OLLAMA_URL

    chunks = chunk_english(text_en, max_chars=max_chars)
    out: List[str] = []

    # set up Ollama client only if needed
    client = None
    if method in ("auto", "ollama"):
        if OllamaClient is None:
            if method == "ollama":
                raise RuntimeError("ollama client not available; pip install ollama")
        else:
            client = OllamaClient(host=host)

    for i, ch in enumerate(chunks, 1):
        # prefer Ollama small model if requested/available
        if method in ("auto", "ollama") and client is not None:
            try:
                print("NLLB is in used")
                # he = translate_chunk_ollama(client, model, ch)
                nllb_output = en_to_he_nllb(ch, device=device, max_new_tokens=512)
                print("light fixing is in use")
                he = lightfix_hebrew(nllb_output)

                # sanity: must be mostly Hebrew and not a refusal/chatter
                if hebrew_ratio(he) >= 0.60 and not looks_like_refusal_or_chatter(he):
                    out.append(he); continue
            except Exception as e:
                # fall through to NLLB
                pass

        # fallback: NLLB distilled 600M
        print("NLLB is in used")
        he = en_to_he_nllb(ch, device=device, max_new_tokens=512)
        out.append(he)

    return "\n\n".join(out).strip()


# ------------------------------------------------------------------------------
#                                         CLI
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Translate English ➔ Hebrew (Ollama small model + NLLB fallback).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Inline English text to translate.")
    src.add_argument("--file", type=str, help="Path to UTF-8 text file to translate.")

    ap.add_argument("--method", choices=["auto","ollama","nllb"], default="auto",
                    help="Translation path. 'auto' tries Ollama then falls back to NLLB.")
    ap.add_argument("--ollama-model", default=OLLAMA_MODEL_TAG, help="Ollama model tag (for ollama/auto).")
    ap.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama base URL.")
    ap.add_argument("--device", default="cpu", help="Device for NLLB (cpu/cuda).")
    ap.add_argument("--max-chars", type=int, default=450, help="Max chars per chunk.")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text_en = f.read()
    else:
        text_en = args.text

    print(f"[debug] input chars={len(text_en)}")
    print(f"[debug] first 300 chars of INPUT =\n\n{text_en[:300]}\n")

    he = translate_en_to_he_chunked(
        text_en,
        method=args.method,
        model=args.ollama_model,
        host=args.ollama_url,
        device=args.device,
        max_chars=args.max_chars
    )
    print(he)

if __name__ == "__main__":
    # main()
    text_en = """
    Do your worst. This is a Libravox recording. All Libravox recordings are in the public domain. For more information or to volunteer, please visit Libravox.org. Speech given by Winston Churchill to the House of Commons, 14 July 1941. The impressive and inspiring spectacle we have witnessed displays the vigor and efficiency of the Civil Defense Forces. They have grown up in the stress of emergency. They have been shaped and tempered by the fire of the enemy, and we saw them all, in their many grades and classes, the wardens, the rescue and first aid parties, the casualty services, the decontamination squads, the fire services, the report and control center staffs, the highway and public utility services, the messengers, the police. No one could but feel how great a people, how great a nation we have the honor to belong to. how complex, sensitive, and resilient is the society we have evolved over the centuries, and how capable of withstanding the most unexpected strain. I must, however, admit that when the storm broke in September I was for several weeks very anxious about the result. Sometimes the gas failed, sometimes the electricity. There were grievous complaints about the shelters and about conditions in them. water was cut off railways were cut or broken large districts were destroyed thousands were killed and many more thousands were wounded but there was only one thing about which there was never any doubt the courage the unconquerable grit and stamina of our people showed itself from the very outset without that all would have failed upon that rock all stood unshakable All the public services were carried on, and all the intricate arrangements, far-reaching details, involving the daily lives of so many millions, were carried out, improvised, elaborated, and perfected, in the very teeth of the cruel and devastating storm. We have to ask ourselves this question. Will the bombing attacks come back again? We have proceeded on the assumption that they will. Many new arrangements are being contrived as a result of the hard experience through which we have passed, and the many mistakes which no doubt we have made. For success is the result of making many mistakes, and learning from experience. If the lull is to end, if the storm is to renew itself, we will be ready. We will not flinch. We can take it again. We ask no favors of the enemy. We seek from them no compunction. on the contrary if to-night our people were asked to cast their vote whether a convention should be entered into to stop the bombing of cities the overwhelming majority would cry no we will mete out to them the measure and more than the measure than they have meted out to us The people, with one voice, would say, You have committed every crime under the sun. Where you have been the least resisted, there you have been the most brutal. It was you who began the indiscriminate bombing. We will have no truce or parley with you, or the grisly gang who work your wicked will. You do your worst, and we will do our best. Perhaps it may be our turn soon. Perhaps it may be our turn now. We live in a terrible epoch of the human story, but we believe there is a broad and sure justice running through its theme. It is time that the enemy should be made to suffer in their own homelands something of the torment they have let loose upon their neighbors and upon the world. We believe it to be in our power to keep this process going, on a steadily rising tide, month after month, year after year, until they are either extirpated by us, or, better still, torn to pieces by their own people. it is for this reason that i must ask you to be prepared for vehement counteraction by the enemy our methods of dealing with them have steadily improved they no longer relish their trips to our shores i do not know why they do not come but it is certainly not because they have begun to love us more It may be because they are saving up, but even if that be so, the very fact that they have to save up should give us confidence by revealing the truth of our steady advance from an almost unarmed position to superiority. But all engaged in our defense forces must prepare themselves for further heavy assaults. Your organization, your vigilance, your devotion to duty, your zeal for the cause, must be raised to the highest intensity. We do not expect to hit without being hit back, and we intend with every week that passes to hit harder. Prepare yourselves, then, my friends and comrades, for this renewal of your exertions. We shall never turn from our purpose, however somber the road, however grievous the cost, because we know that out of this time of trial and tribulation will be born a new freedom and glory for all mankind. End of speech.
    """

    print(f"[debug] input chars={len(text_en)}")
    print(f"[debug] first 500 chars of INPUT =\n{text_en[:500]}")

    hebrew = translate_en_to_he_chunked(
        text_en,
        max_chars=300,
        model=config.OLLAMA_MODEL_TAG,  # e.g., "aya:8b"
        host=config.OLLAMA_URL
    )
    print(hebrew)
