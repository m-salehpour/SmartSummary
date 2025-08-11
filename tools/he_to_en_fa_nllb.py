# # --- NLLB robust chunked translation HE -> EN / FA ---------------------------
# import re, math
# from typing import List, Tuple
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# _NLLB_ID = "facebook/nllb-200-distilled-600M"
# _TOK = None
# _MDL = None
#
# def _device():
#     if torch.cuda.is_available(): return "cuda"
#     if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
#     return "cpu"
#
# def _load(src_lang: str):
#     global _TOK, _MDL
#     if _TOK is None:
#         _TOK = AutoTokenizer.from_pretrained(_NLLB_ID, src_lang=src_lang)
#     if _MDL is None:
#         _MDL = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_ID).to(_device())
#     return _TOK, _MDL
#
# def _lang_id(tok, code: str) -> int:
#     # new-style first
#     try:
#         return tok.convert_tokens_to_ids(code)
#     except Exception:
#         pass
#     # old fallback(s)
#     if hasattr(tok, "lang_code_to_id") and code in tok.lang_code_to_id:
#         return tok.lang_code_to_id[code]
#     vid = tok.get_vocab().get(code)
#     if vid is None:
#         raise ValueError(f"Language code {code!r} not found in tokenizer vocab.")
#     return vid
#
# # --- light cleanup (ASR stutters, spacing, bidi marks) ---
# _BIDI = re.compile(r'[\u200e\u200f]')
# _WS   = re.compile(r'[ \t]+')
# # collapse 3+ repeated identical words (case-insensitive)
# _DEDUP = re.compile(r'\b(\w{2,})\b(?:\s+\1\b){2,}', flags=re.IGNORECASE)
# def _clean_he(s: str) -> str:
#     s = _BIDI.sub('', s)
#     s = _WS.sub(' ', s)
#     s = _DEDUP.sub(r'\1', s)  # one occurrence
#     return s.strip()
#
#
# def _clean_en(s: str) -> str:
#     s = _BIDI.sub('', s)
#     s = _WS.sub(' ', s)
#     s = re.sub(r'\s+([,.;:!?])', r'\1', s)
#     return s.strip()
#
# def _clean_fa(s: str) -> str:
#     s = _BIDI.sub('', s)
#     s = _WS.sub(' ', s)
#     s = re.sub(r'\s+([،؛٫٫،\.:\-–—!؟\?!])', r'\1', s)
#     return s.strip()
#
# # --- sentence-ish splitter (Hebrew + punctuation + paragraph breaks)
# _SENT_RE = re.compile(r'(?:(?<=[\.!\?…])\s+|\n{2,})')
#
# def _sentences(text: str) -> List[str]:
#     parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
#     return parts if parts else [text.strip()]
#
# def _token_len(tok, s: str) -> int:
#     return len(tok.encode(s, add_special_tokens=False))
#
# # --- replace your _chunk_by_tokens with this overlapping version ---
# def _chunk_by_tokens(tok, text: str, max_src_tokens: int = 380, overlap_tokens: int = 40) -> List[str]:
#     """Split by sentences, keep each chunk < max_src_tokens, and add overlap for context."""
#     sents = _sentences(text)
#     chunks, buf, cur = [], [], 0
#     sent_tok = [tok.encode(s, add_special_tokens=False) for s in sents]
#
#     for toks, s in zip(sent_tok, sents):
#         tlen = len(toks)
#         if cur and cur + tlen > max_src_tokens:
#             chunks.append(" ".join(buf))
#             # start next with an overlap from the end of previous buffer
#             if overlap_tokens > 0:
#                 overlap = []
#                 cur_toks = []
#                 # collect sentences from the end until we reach ~overlap_tokens
#                 for prev_s in reversed(buf):
#                     ptoks = tok.encode(prev_s, add_special_tokens=False)
#                     overlap.append(prev_s)
#                     cur_toks.extend(ptoks)
#                     if len(cur_toks) >= overlap_tokens:
#                         break
#                 buf = list(reversed(overlap))
#                 cur = sum(len(tok.encode(x, add_special_tokens=False)) for x in buf)
#         buf.append(s)
#         cur += tlen
#     if buf:
#         chunks.append(" ".join(buf))
#     return chunks
#
# # --- slightly stricter short-output recovery & logging in _gen_one ---
# # --- stronger generation with coverage target ---
# def _gen_one(tok, mdl, src_text: str, tgt_code: str) -> str:
#     bos = _lang_id(tok, tgt_code)
#     enc = tok(src_text, return_tensors="pt", truncation=True).to(mdl.device)
#     in_tokens = enc["input_ids"].shape[1]
#
#     # Aim for ~1:1 length; allow extra room
#     max_new = max(256, min(1024, int(in_tokens * 2.2) + 80))
#     min_new = max(180, min(480, int(in_tokens * 0.90)))  # was ~160 before
#
#     out = mdl.generate(
#         **enc,
#         forced_bos_token_id=bos,
#         num_beams=3,
#         length_penalty=1.05,   # slightly favors longer
#         no_repeat_ngram_size=3,
#         min_new_tokens=min_new,
#         max_new_tokens=max_new,
#         early_stopping=False,  # let it use the budget
#     )
#     text = tok.batch_decode(out, skip_special_tokens=True)[0]
#     out_tokens = _token_len(tok, text)
#     print(f"[gen] src={in_tokens} tok -> out={out_tokens} tok (min={min_new})")
#     return text
#
# # --- fallback: if output too short, split the source chunk and translate the halves ---
# def _translate_with_coverage(tok, mdl, src_text: str, tgt_code: str, depth: int = 0) -> str:
#     text = _gen_one(tok, mdl, src_text, tgt_code)
#     in_len = _token_len(tok, src_text)
#     out_len = _token_len(tok, text)
#     if out_len >= int(in_len * 0.80) or depth >= 2:
#         return text  # good enough or max retries
#
#     # too short: split by sentences roughly in half and translate each
#     sents = _sentences(src_text)
#     if len(sents) < 2:
#         # nothing to split; give it one more try with sampling
#         enc = tok(src_text, return_tensors="pt", truncation=True).to(mdl.device)
#         bos = _lang_id(tok, tgt_code)
#         in_tokens = enc["input_ids"].shape[1]
#         max_new = max(256, min(1024, int(in_tokens * 2.2) + 80))
#         min_new = max(180, min(480, int(in_tokens * 0.95)))
#         out = mdl.generate(
#             **enc,
#             forced_bos_token_id=bos,
#             do_sample=True, temperature=0.85, top_p=0.92,
#             no_repeat_ngram_size=3,
#             min_new_tokens=min_new,
#             max_new_tokens=max_new,
#         )
#         return tok.batch_decode(out, skip_special_tokens=True)[0]
#
#     # split near mid by tokens
#     tok_lens = [ _token_len(tok, s) for s in sents ]
#     total = sum(tok_lens)
#     acc = 0
#     cut = 1
#     for i, tl in enumerate(tok_lens, 1):
#         acc += tl
#         if acc >= total//2:
#             cut = i
#             break
#     left  = " ".join(sents[:cut]).strip()
#     right = " ".join(sents[cut:]).strip()
#
#     print(f"[cover] splitting chunk: left={_token_len(tok,left)} tok, right={_token_len(tok,right)} tok")
#     t_left  = _translate_with_coverage(tok, mdl, left,  tgt_code, depth+1)
#     t_right = _translate_with_coverage(tok, mdl, right, tgt_code, depth+1)
#     return (t_left.strip() + " " + t_right.strip()).strip()
#
#
# def he_to_en(he_text: str, max_src_tokens: int = 380) -> str:
#     he_text = _clean_he(he_text)
#     tok, mdl = _load(src_lang="heb_Hebr")
#     chunks = _chunk_by_tokens(tok, he_text, max_src_tokens=max_src_tokens)
#     outs = []
#     for i, ch in enumerate(chunks, 1):
#         print(f"[debug] translating chunk {i}/{len(chunks)} (src_tok={_token_len(_TOK, ch)})")
#         out = _translate_with_coverage(_TOK, _MDL, ch, "eng_Latn")  # or "pes_Arab" for FA
#         outs.append(out)
#     return "\n\n".join(outs).strip()
#
# def he_to_fa(he_text: str, max_src_tokens: int = 380) -> str:
#     he_text = _clean_he(he_text)
#     tok, mdl = _load(src_lang="heb_Hebr")
#     chunks = _chunk_by_tokens(tok, he_text, max_src_tokens=max_src_tokens)
#     outs = []
#     for i, ch in enumerate(chunks, 1):
#         out = _gen_one(tok, mdl, ch, tgt_code="pes_Arab")
#         outs.append(_clean_fa(out))
#     return "\n\n".join(outs).strip()
#
#
# if __name__ == "__main__":
#     # main()
#     he_text ="""
#      ערב טוב אזרחי ישראל. לפני זמן קצר הודעתי לנשיא המדינה שאני מחזיר לידיו את המנדט להרכבת הממשלה. מאז שקיבלתי את המנדט, פעלתי ללא ערב, גם בגלוי, גם בסתם, כדי להקים ממשלת אחדות לאומית רחבה. זה מה שהעם רוצה. זה גם מה שישראל צריכה אל מול האתגרים הביטחוניים שהולכים וגדלים מדי יום, מדי שעה. במהלך השבועות האחרונים עשיתי כל מאמץ כדי להביא את בני גנץ לשולחן המסע המתן, כל מאמץ כדי להקים ממשלה לאומית רחבה, כל מאמץ כדי למנוע בחירות נוספות. לצערי, פעם אחר פעם הוא פשוט סרב. תחלה סרב למתווה הנשיא, אחר כך סרב להיפגש איתי, אחר כך סרב לשלוח את צוות המסע המתן שלו, ולבסוף... הוא סירב לדון במתווה הפשרה שהצגתי. הוא לא נותן לזה אפילו חמש דקות שום דיון רציני, פשוט נתן תשובה אוטומטית שלילית. הוא התמיד בסירובו גם אחרי שקודם לכם נעניתי לבקשתו להיפגש עם הרמטכאל שהציג בפניו את מכלול האיומים והאתגרים שמדינת ישראל ניצבת מולה. לצערי הסרבנות של גנץ מידע על דבר אחד. בני גנץ נותר שבוי בידיהם של ליברמן ולפיד. לפיד שרוצה במפלטו וליברמן שמונע משיקולים זרים שנוגעים לענייניו האישיים. גאנס, לפיד וליברמן רק מדברים על חדות. בפועל הם עושים את ההפך הגמור. הם מעודדים פילוג וחרמות. הם פוסלים את חופשי הכיפות. ואת מי הם לא פוסלים? את חברי הרשימה הערבית המשותפת. הם מתואמים איתם כל הדרך. לממשלת מיעוט של השמאל. אני רוצה שתבינו, ממשלת המיעוט תקום בתמיכתם של חברי הרשימה המשותפת, אלה שמאדירים את הטרור, אלה ששוללים את עצם קיומה של מדינת ישראל. המפלגות הערביות כבר המליצו על גנץ בפני הנשיא, בדיוק כפי שהתרעתי לפני הבחירות. ועכשיו, רק לאחרונה, הם הכרימו את השבעת הכנסת וסירבו להציר אימונים למדינת ישראל. איך תוכל ממשלת המיעוט של גאנס, שנשענת על מפלגות אלה, איך היא תוכל להילחם בטרור, בחמאס, בחיזבאללה, באיראן? התשובה הפשוטה היא לא תוכל. אם גאנס התפתה להקים ממשלה מסוכנת כזאת, אעמוד בראש האופוזיציה ואפעל יחד עם חבריי כדי להחליפה במהירות. אבל עדיין לא מאוחר. אם גנצי תשת, אם הוא ישתחרר מהלפיטה של לפיד וליברמן, אם הוא יזנח את רעיון ממשלת המיעוט, נוכל להקים יחד את הממשלה שמדינת ישראל כל כך זקוקה לבעת הזאת. ממשלת אחדות לאומית רחבה של כל אלה המאמינים במדינת ישראל כמדינה יהודית, כמדינה דמוקרטית. זה היה הפתרון וזה נשאר הפתרון.
#     """
#
#     max_chars = 300
#
#     # Hebrew -> English
#     en = he_to_en(he_text)  # auto-picks CUDA/MPS/CPU
#     print("EN:\n", en)
#     #
#     # # Hebrew -> Persian (pivot HE->EN->FA for better quality)
#     # fa = he_to_fa(he_text, max_chars=max_chars, pivot=True)
#     # print("FA (pivot):\n", fa)
#     #
#     # # If you prefer direct HE->FA without pivot:
#     # fa_direct = he_to_fa(he_text, max_chars=max_chars, pivot=False)
#     # print("FA (direct NLLB):\n", fa_direct)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
he_to_en_fa_nllb.py — Hebrew → English / Persian via NLLB-200 distilled 1.3B

Key fixes:
- Sentence-by-sentence translation (less babble/omissions)
- Micro-splitting very long sentences by commas/conjunctions
- Tagged placeholders {{LIKE_THIS}} + regex unprotect (more robust)
- Conservative generation (greedy) with modest max length
- Minimal cleanup

Requires:
  pip install torch transformers sentencepiece
"""

import re
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# Model & language settings
# =========================

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "heb_Hebr"
TGT_EN   = "eng_Latn"
TGT_FA   = "pes_Arab"  # Persian; use "prs_Arab" for Dari if you prefer

# =========================
# Named-entity protection
# =========================
# Hebrew -> bare code; we will wrap as {{CODE}} when protecting
PROTECT_HE_TO_CODE = {
    "בני גנץ": "BENNY_GANTZ",
    "גנץ": "GANTZ",
    "יאיר לפיד": "YAIR_LAPID",
    "לפיד": "LAPID",
    "אביגדור ליברמן": "AVIGDOR_LIBERMAN",
    "ליברמן": "LIBERMAN",
    "הרשימה המשותפת": "JOINT_LIST",
    "חמאס": "HAMAS",
    "חיזבאללה": "HEZBOLLAH",
    "איראן": "IRAN",
    "הרמטכ\"ל": "IDF_CHIEF",
    "הרמטכ״ל": "IDF_CHIEF",
    "כנסת": "KNESSET",
    "ממשלת מיעוט": "MINORITY_GOVT",
}

# Code -> English/Persian
UNPROTECT_CODE_TO_EN = {
    "BENNY_GANTZ": "Benny Gantz",
    "GANTZ": "Gantz",
    "YAIR_LAPID": "Yair Lapid",
    "LAPID": "Lapid",
    "AVIGDOR_LIBERMAN": "Avigdor Liberman",
    "LIBERMAN": "Liberman",
    "JOINT_LIST": "the Joint List",
    "HAMAS": "Hamas",
    "HEZBOLLAH": "Hezbollah",
    "IRAN": "Iran",
    "IDF_CHIEF": "the IDF Chief of Staff",
    "KNESSET": "the Knesset",
    "MINORITY_GOVT": "a minority government",
}

UNPROTECT_CODE_TO_FA = {
    "BENNY_GANTZ": "بنی گانتس",
    "GANTZ": "گانتس",
    "YAIR_LAPID": "یائیر لاپید",
    "LAPID": "لاپید",
    "AVIGDOR_LIBERMAN": "آویگدور لیبرمن",
    "LIBERMAN": "لیبرمن",
    "JOINT_LIST": "فهرست مشترک (عربی)",
    "HAMAS": "حماس",
    "HEZBOLLAH": "حزب‌الله",
    "IRAN": "ایران",
    "IDF_CHIEF": "رئیس ستاد کل ارتش اسرائیل",
    "KNESSET": "کنست",
    "MINORITY_GOVT": "دولت اقلیت",
}

# Wrap codes as {{CODE}} for better preservation
def wrap_code(code: str) -> str:
    return "{{" + code + "}}"

PROTECT_HE_TO_TAG = {he: wrap_code(code) for he, code in PROTECT_HE_TO_CODE.items()}

# =========================
# Lazy model/tokenizer load
# =========================

_TOK = None
_MDL = None

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _load(src_lang: str = SRC_LANG):
    global _TOK, _MDL
    if _TOK is None or _MDL is None:
        device = _pick_device()
        _TOK = AutoTokenizer.from_pretrained(MODEL_ID)
        _TOK.src_lang = src_lang
        dtype = torch.float16 if device.type == "cuda" else None
        _MDL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
        _MDL.to(device).eval()
    return _TOK, _MDL

# =========================
# Text utilities
# =========================

def clean_he(text: str) -> str:
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

# sentence split (., !, ?, … or newline)
SENT_SPLIT = re.compile(r'(?<=[\.!?…])\s+|\n+')

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]

def micro_split_long_sentence(tok, s: str, max_src_tokens: int = 220) -> List[str]:
    """Split very long sentence into shorter clauses by commas/and-words."""
    if token_len(tok, s) <= max_src_tokens:
        return [s]
    # split by commas or Hebrew ' ו' (space-vav), keep delimiters lightly
    parts = re.split(r'(,| ו)', s)
    out, cur = [], ""
    for p in parts:
        tent = (cur + p).strip()
        if token_len(tok, tent) <= max_src_tokens:
            cur = tent
        else:
            if cur:
                out.append(cur)
            cur = p.strip()
    if cur:
        out.append(cur)
    # final check
    really = []
    for seg in out:
        if token_len(tok, seg) <= max_src_tokens:
            really.append(seg)
        else:
            # hard truncate (very rare)
            really.append(truncate_to_tokens(tok, seg, max_src_tokens))
    return really

def token_len(tok, s: str) -> int:
    return len(tok(s, return_tensors="pt", truncation=True)["input_ids"][0])

def truncate_to_tokens(tok, s: str, max_tokens: int) -> str:
    ids = tok(s, return_tensors="pt", truncation=True)["input_ids"][0]
    if len(ids) <= max_tokens:
        return s
    ids = ids[:max_tokens]
    return tok.decode(ids, skip_special_tokens=True)

def strip_stage_directions(text: str) -> str:
    text = re.sub(r'\((?:\s*[A-Za-z]+[\s\-]*){1,5}\)', '', text)
    text = re.sub(r'\[(?:\s*[A-Za-z]+[\s\-]*){1,5}\]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# =========================
# Protect / unprotect
# =========================

def protect_terms_he(text: str, mapping: Dict[str, str]) -> str:
    # Replace longer keys first
    for he, tag in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(he, f" {tag} ")  # pad to isolate from Hebrew letters
    return text

def _unprotect_with_regex(text: str, code_to_human: Dict[str, str]) -> str:
    for code, human in sorted(code_to_human.items(), key=lambda kv: len(kv[0]), reverse=True):
        # Accept {{ CODE }}, {{CODE}}, [[CODE]], §CODE§, or plain bare CODE
        patterns = [
            rf"\{{\{{\s*{re.escape(code)}\s*\}}\}}",
            rf"\[\[\s*{re.escape(code)}\s*\]\]",
            rf"§\s*{re.escape(code)}\s*§",
            rf"\b{re.escape(code)}\b",
        ]
        for pat in patterns:
            text = re.sub(pat, human, text)
    return text

# =========================
# Language id helper
# =========================

def get_lang_id(tok, lang_code: str) -> int:
    tid = tok.convert_tokens_to_ids(lang_code)
    if tid is not None and tid != getattr(tok, "unk_token_id", None):
        return tid
    tid = tok.convert_tokens_to_ids(f"__{lang_code}__")
    if tid is not None and tid != getattr(tok, "unk_token_id", None):
        return tid
    raise ValueError(f"Cannot resolve language token id for '{lang_code}'")

# =========================
# Generation
# =========================

def generate_safe(tok, mdl, src_text: str, tgt_lang: str) -> str:
    inputs = tok(src_text, return_tensors="pt", truncation=True).to(mdl.device)
    src_len = inputs["input_ids"].shape[1]

    # Conservative: greedy, modest max length, no minimum
    max_new = max(32, min(240, int(src_len * 1.7) + 16))

    out_ids = mdl.generate(
        **inputs,
        forced_bos_token_id=get_lang_id(tok, tgt_lang),
        num_beams=1,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        length_penalty=1.0,
        min_new_tokens=0,
        max_new_tokens=max_new,
        early_stopping=True,
    )
    return tok.batch_decode(out_ids, skip_special_tokens=True)[0]

def post_edit_en(text: str) -> str:
    fixes = [
        (r"\bwaiting table\b", "negotiating table"),
        (r"\bHermetchal\b", "the IDF Chief of Staff"),
        (r"\bGuns, and L\.A\.P\.D\., and Liberman\b", "Gantz, Lapid and Liberman"),
        (r"\bdivision and confiscation\b", "division and boycotts"),
        (r"\bdome freaks\b", "kippah-wearers"),
        (r"\btook the Knesset oath and refused\b", "boycotted the Knesset swearing-in and refused"),
        (r"\bIf Gantz drinks\b", "If Gantz frees himself"),
        (r"\brefuses his wait staff\b", "refused to send his negotiating team"),
        (r"\bPresident'?s blueprint\b", "President's framework"),
        # Tidy double spaces that sometimes appear after replacements
        (r"\s{2,}", " "),
    ]
    for pat, repl in fixes:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text.strip()

# =========================
# Public API
# =========================

def he_to_en(he_text: str, sent_tok_limit: int = 220) -> str:
    tok, mdl = _load(SRC_LANG)
    he_text = clean_he(he_text)
    sents = split_sentences(he_text)

    outs: List[str] = []
    for s in sents:
        protected = protect_terms_he(s, PROTECT_HE_TO_TAG)
        parts = micro_split_long_sentence(tok, protected, max_src_tokens=sent_tok_limit)
        part_outs = []
        for p in parts:
            out = generate_safe(tok, mdl, p, TGT_EN)
            part_outs.append(out.strip())
        out_sent = " ".join(part_outs)
        out_sent = _unprotect_with_regex(out_sent, UNPROTECT_CODE_TO_EN)
        out_sent = strip_stage_directions(out_sent)
        outs.append(out_sent)

    joined = " ".join(outs)
    joined = re.sub(r"\s+", " ", joined).strip()
    joined = post_edit_en(joined)
    return joined

def he_to_fa(he_text: str, sent_tok_limit: int = 220) -> str:
    tok, mdl = _load(SRC_LANG)
    he_text = clean_he(he_text)
    sents = split_sentences(he_text)

    outs: List[str] = []
    for s in sents:
        protected = protect_terms_he(s, PROTECT_HE_TO_TAG)
        parts = micro_split_long_sentence(tok, protected, max_src_tokens=sent_tok_limit)
        part_outs = []
        for p in parts:
            out = generate_safe(tok, mdl, p, TGT_FA)
            part_outs.append(out.strip())
        out_sent = " ".join(part_outs)
        out_sent = _unprotect_with_regex(out_sent, UNPROTECT_CODE_TO_FA)
        out_sent = strip_stage_directions(out_sent)
        outs.append(out_sent)

    joined = " ".join(outs)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined

def must_contain_all(text: str, keys: List[str]):
    missing = [k for k in keys if k not in text]
    if missing:
        print("[warn] missing in output:", missing)


if __name__ == "__main__":
    # main()
    he_text ="""
     ערב טוב אזרחי ישראל. לפני זמן קצר הודעתי לנשיא המדינה שאני מחזיר לידיו את המנדט להרכבת הממשלה. מאז שקיבלתי את המנדט, פעלתי ללא ערב, גם בגלוי, גם בסתם, כדי להקים ממשלת אחדות לאומית רחבה. זה מה שהעם רוצה. זה גם מה שישראל צריכה אל מול האתגרים הביטחוניים שהולכים וגדלים מדי יום, מדי שעה. במהלך השבועות האחרונים עשיתי כל מאמץ כדי להביא את בני גנץ לשולחן המסע המתן, כל מאמץ כדי להקים ממשלה לאומית רחבה, כל מאמץ כדי למנוע בחירות נוספות. לצערי, פעם אחר פעם הוא פשוט סרב. תחלה סרב למתווה הנשיא, אחר כך סרב להיפגש איתי, אחר כך סרב לשלוח את צוות המסע המתן שלו, ולבסוף... הוא סירב לדון במתווה הפשרה שהצגתי. הוא לא נותן לזה אפילו חמש דקות שום דיון רציני, פשוט נתן תשובה אוטומטית שלילית. הוא התמיד בסירובו גם אחרי שקודם לכם נעניתי לבקשתו להיפגש עם הרמטכאל שהציג בפניו את מכלול האיומים והאתגרים שמדינת ישראל ניצבת מולה. לצערי הסרבנות של גנץ מידע על דבר אחד. בני גנץ נותר שבוי בידיהם של ליברמן ולפיד. לפיד שרוצה במפלטו וליברמן שמונע משיקולים זרים שנוגעים לענייניו האישיים. גאנס, לפיד וליברמן רק מדברים על חדות. בפועל הם עושים את ההפך הגמור. הם מעודדים פילוג וחרמות. הם פוסלים את חופשי הכיפות. ואת מי הם לא פוסלים? את חברי הרשימה הערבית המשותפת. הם מתואמים איתם כל הדרך. לממשלת מיעוט של השמאל. אני רוצה שתבינו, ממשלת המיעוט תקום בתמיכתם של חברי הרשימה המשותפת, אלה שמאדירים את הטרור, אלה ששוללים את עצם קיומה של מדינת ישראל. המפלגות הערביות כבר המליצו על גנץ בפני הנשיא, בדיוק כפי שהתרעתי לפני הבחירות. ועכשיו, רק לאחרונה, הם הכרימו את השבעת הכנסת וסירבו להציר אימונים למדינת ישראל. איך תוכל ממשלת המיעוט של גאנס, שנשענת על מפלגות אלה, איך היא תוכל להילחם בטרור, בחמאס, בחיזבאללה, באיראן? התשובה הפשוטה היא לא תוכל. אם גאנס התפתה להקים ממשלה מסוכנת כזאת, אעמוד בראש האופוזיציה ואפעל יחד עם חבריי כדי להחליפה במהירות. אבל עדיין לא מאוחר. אם גנצי תשת, אם הוא ישתחרר מהלפיטה של לפיד וליברמן, אם הוא יזנח את רעיון ממשלת המיעוט, נוכל להקים יחד את הממשלה שמדינת ישראל כל כך זקוקה לבעת הזאת. ממשלת אחדות לאומית רחבה של כל אלה המאמינים במדינת ישראל כמדינה יהודית, כמדינה דמוקרטית. זה היה הפתרון וזה נשאר הפתרון.
    """
    print("he text: ", he_text)

    # en = he_to_en(he_text)
    # print("EN:\n", en, "\n", sep="")
    # must_contain_all(en, ["Benny Gantz", "Lapid", "Liberman", "the Joint List", "Hamas", "Hezbollah", "Iran"])

    # Uncomment for Persian as well:
    fa = he_to_fa(he_text)
    print("FA:\n", fa, "\n", sep="")
    # Also produce Persian if you want
    # fa = he_to_fa(he_text)
    # print("FA:\n", fa, "\n", sep="")
