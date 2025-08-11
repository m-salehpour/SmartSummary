import argparse, re, sys
from ollama import Client

import config

gemma_opts = {
    "temperature": 0.30,
    "top_p": 0.90,
    "top_k": 40,
    "repeat_penalty": 1.42,   # <- strong anti-repeat
    "repeat_last_n": 256,
    "num_predict": 640,       # <- short leash; prevents rambling
    "seed": 0,
    "stop": [
        "CONTINUE","Please continue","HE:","FA:","User:","Assistant:",
        "מה שלומך","בסדר גמור","אנא ספק","אני מקווה"  # kill small-talk/boilerplate
    ],
}

import re
from collections import Counter

RE_WORD    = re.compile(r"[^\s]+")
RE_NIQQUD  = re.compile(r'[\u0591-\u05C7]')
RE_ARABIC  = re.compile(r'[\u0600-\u06FF]')
SMALLTALK  = re.compile(r"(מה שלומך|אני מקווה|בסדר גמור|אנא ספק|לא יכול.?ה לעזור|ננסה שוב)", re.I)
META       = re.compile(r"(הנה תרגום לעברית|תרגום|עִברית|עברית:|HE:)", re.I)

def is_degenerate(he: str, max_ratio=0.22) -> bool:
    s = RE_NIQQUD.sub('', he).strip()
    words = RE_WORD.findall(s)
    if len(words) < 12: return False
    w, c = Counter(words).most_common(1)[0]
    return (c / len(words)) >= max_ratio

def looks_bad(he: str) -> bool:
    if len(he) < 24: return True
    if RE_ARABIC.search(he): return True
    if SMALLTALK.search(he) or META.search(he): return True
    return is_degenerate(he)

def sanitize_meta(he: str) -> str:
    # remove obvious meta/chat lines entirely
    lines = [ln for ln in he.splitlines()
             if not SMALLTALK.search(ln) and not META.search(ln)]
    he = " ".join(lines)
    # collapse spaces once more
    he = re.sub(r'\s+', ' ', he).strip()
    return he

def translate_chunk_gemma(client, model, fa_chunk, base_opts):
    messages = [
        {"role":"system","content":
         "You are a Persian→Hebrew translator. Output HEBREW ONLY (U+05D0–U+05EA), no niqqud. Do not chat."},
        {"role":"user","content": "Translate to Hebrew only:\n\n" + fa_chunk}
    ]
    opts = {**base_opts}
    out = client.chat(model=model, messages=messages, options=opts)["message"]["content"].strip()
    he = sanitize_meta(postprocess_hebrew(out))

    if looks_bad(he):
        # One retry: different seed + slightly different sampling
        retry_opts = {**opts, "temperature": 0.35, "top_p": 0.70,
                      "repeat_penalty": 1.55, "seed": opts.get("seed",0) + 1}
        out2 = client.chat(model=model, messages=messages, options=retry_opts)["message"]["content"].strip()
        he2 = sanitize_meta(postprocess_hebrew(out2))
        if not looks_bad(he2):
            he = he2
    return he

# requires: pip install transformers sentencepiece torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_MODEL_ID = "facebook/nllb-200-distilled-600M"
_SRC = "pes_Arab"   # Persian (Farsi)
_TGT = "heb_Hebr"   # Hebrew

_tok   = AutoTokenizer.from_pretrained(_MODEL_ID, src_lang=_SRC)  # keep use_fast default (True)
_model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID)

def fa_to_he_nllb(text: str) -> str:
    inputs = _tok(text, return_tensors="pt")
    bos_id = _tok.convert_tokens_to_ids(_TGT)  # works for Fast & Slow tokenizers
    out = _model.generate(
        **inputs,
        forced_bos_token_id=bos_id,
        max_new_tokens=512,
        no_repeat_ngram_size=3,
    )
    return _tok.batch_decode(out, skip_special_tokens=True)[0]

# --- add near your imports ---
# pip install transformers sentencepiece torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_OPUS_MDL = "Helsinki-NLP/opus-mt-fa-he"
_opus_tok = _opus = None

def _ensure_opus(device="cpu"):
    global _opus_tok, _opus
    if _opus is None:
        _opus_tok = AutoTokenizer.from_pretrained(_OPUS_MDL)
        _opus = AutoModelForSeq2SeqLM.from_pretrained(_OPUS_MDL).to(device).eval()

def fa_to_he_opus(text: str, device: str = "cpu") -> str:
    _ensure_opus(device)
    enc = _opus_tok(text, return_tensors="pt", truncation=True).to(device)
    gen = _opus.generate(
        **enc,
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        max_new_tokens=512,
        early_stopping=True,
    )
    he = _opus_tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return postprocess_hebrew(he)

SYSTEM_PROMPT = (
    "You are a professional **Persian** to **Hebrew** translator. "
    "Translate the user's **Persian** text into formal modern **Hebrew** WITHOUT niqqud."
    "Do not add, omit, interpret, summarize, or rearrange."
    "Do NOT repeat the **Persian** text."
    "***Output HEBREW ONLY.**"
)

# --- ADD: Aya-specific system + few-shot (anchors FA→HE) ---
AYA_SYSTEM_PROMPT = (
    "You are a professional Persian→Hebrew translator.\n"
    "Output HEBREW ONLY (Unicode U+05D0–U+05EA), NO niqqud. Translate ONLY; do not greet or chat.\n"
    "Do NOT add/omit/interpret/reorder. Do NOT quote or repeat the Persian.\n"
    "If unclear, write [בלתי ברור]."
)
AYA_FEWSHOT = [
    {"role": "user", "content": "FA: سلام"},
    {"role": "assistant", "content": "HE: שלום"},
]

# ---------- tiny post-processing (yours, unchanged) ----------
_HEB_DIACRITICS = re.compile(r'[\u0591-\u05C7]')
_BIDI_MARKS     = re.compile(r'[\u200e\u200f]')
_REPEAT_TOKENS  = re.compile(r'(\b[\u05D0-\u05EA]{2,}\b)(?:\s+\1){2,}')

def postprocess_hebrew(s: str) -> str:
    s = _HEB_DIACRITICS.sub('', s)
    s = _BIDI_MARKS.sub('', s)
    s = _REPEAT_TOKENS.sub(r'\1', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()

def looks_complete(s: str) -> bool:
    return bool(s) and s[-1] in '.!?״”’"״'

# --- ADD: Hebrew-only validator (blocks echos/mixed script) ---
_HEB_ONLY = re.compile(r'^[\u05D0-\u05EA0-9\s\.,;:!\?()\-\u05F3\u05F4"״׳]+$')
_HAS_ARABIC_PERSIAN = re.compile(r'[\u0600-\u06FF]')

def is_hebrew_only(s: str) -> bool:
    if not s.strip():
        return False
    if _HAS_ARABIC_PERSIAN.search(s):
        return False
    return bool(_HEB_ONLY.match(s))

# ---------- chunking (yours, unchanged) ----------
# replace your _SENT_SPLIT with this (broader end-of-sentence set)
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?؟؛…:])\s+')

def _hard_wrap_by_words(s: str, max_chars: int) -> list[str]:
    """Greedy wrap at whitespace so no chunk exceeds max_chars."""
    chunks, i, n = [], 0, len(s)
    while i < n:
        j = min(n, i + max_chars)
        if j < n:
            # try last whitespace within window
            k = s.rfind(' ', i, j)
            if k == -1 or k <= i + int(max_chars * 0.6):
                k = j  # no good space; hard cut
        else:
            k = j
        chunk = s[i:k].strip()
        if chunk:
            chunks.append(chunk)
        i = k
    return chunks

def chunk_persian(text: str, max_chars: int = 1100) -> list[str]:
    """Paragraph → sentences (wide delimiters) → word-wrapped fallback."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    paragraphs = re.split(r'\n{2,}', text)  # keep your paragraph logic
    for para in paragraphs:
        p = re.sub(r'\s+', ' ', para.strip())  # normalize spaces
        if not p:
            continue
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        # try sentence grouping first
        sents = [s.strip() for s in _SENT_SPLIT.split(p) if s.strip()]
        if len(sents) == 1:
            # no sentence breaks found — hard wrap
            chunks.extend(_hard_wrap_by_words(p, max_chars))
            continue
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
        # safety: if any chunk still too long (rare), hard-wrap it
        fixed = []
        for c in chunks:
            if len(c) > max_chars:
                fixed.extend(_hard_wrap_by_words(c, max_chars))
            else:
                fixed.append(c)
        chunks = fixed
    return chunks
# ---------- translate one chunk (generic path; unchanged) ----------
def translate_chunk(client: Client, model: str, fa_chunk: str, options: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": fa_chunk},
    ]
    resp = client.chat(model=model, messages=messages, options=options)
    he = postprocess_hebrew(resp["message"]["content"])
    tries = 0
    while not looks_complete(he) and tries < 2:
        tries += 1
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": he},
            {"role": "user", "content": "המשך את התרגום מהנקודה שבה הפסקת. "
                                        "אל תחזור על טקסט קודם. עברית בלבד, בלי ניקוד."}
        ]
        more = client.chat(model=model, messages=messages, options=options)["message"]["content"]
        he += " " + postprocess_hebrew(more)
    return he.strip()

# --- ADD: Aya translation (anti-echo, 'HE:' anchor, retry if Persian appears) ---
def translate_aya(client: Client, model: str, fa_text: str, options: dict) -> str:
    messages = [{"role": "system", "content": AYA_SYSTEM_PROMPT}, *AYA_FEWSHOT, {
        "role": "user",
        "content": (
            "Translate Persian → Hebrew. Hebrew ONLY (no niqqud). "
            "Begin with 'HE:' and do not include any Persian.\n\n"
            "FA:\n" + fa_text.strip()
        ),
    }]
    aya_opts = {
        "temperature": options.get("temperature", 0.25),
        "top_p": options.get("top_p", 0.9),
        "top_k": options.get("top_k", 40),
        "repeat_penalty": options.get("repeat_penalty", 1.25),
        "repeat_last_n": options.get("repeat_last_n", 192),
        "num_predict": options.get("num_predict", 2048),
        "seed": options.get("seed", 0),
        "stop": options.get("stop", ["CONTINUE", "Please continue", "请继续", "继续", "继续翻译"]),
    }
    out = client.chat(model=model, messages=messages, options=aya_opts)["message"]["content"].strip()
    if out.startswith("HE:"):
        out = out[3:].lstrip()
    out = postprocess_hebrew(out)
    if not is_hebrew_only(out) or out.startswith("שלום"):
        # Retry once with slightly stronger anti-repeat, slightly higher temp
        messages.append({"role": "user", "content":
            "Output Hebrew only (U+05D0–U+05EA). Remove all Persian/Arabic letters and redo the translation. "
            "Do not repeat prior text; continue the translation only. Begin with 'HE:'."})
        stricter = dict(aya_opts)
        stricter["temperature"] = 0.30
        stricter["repeat_penalty"] = 1.30
        out2 = client.chat(model=model, messages=messages, options=stricter)["message"]["content"].strip()
        if out2.startswith("HE:"):
            out2 = out2[3:].lstrip()
        out2 = postprocess_hebrew(out2)
        if is_hebrew_only(out2):
            return out2
    return out

def translate_chunk_aya(client: Client, model: str, fa_chunk: str, options: dict) -> str:
    he = translate_aya(client, model, fa_chunk, options)
    tries = 0
    while not looks_complete(he) and tries < 2:
        tries += 1
        cont_messages = [{"role": "system", "content": AYA_SYSTEM_PROMPT}, *AYA_FEWSHOT,
            {"role": "assistant", "content": "HE: " + he},
            {"role": "user", "content":
                "Continue the translation from where you stopped. "
                "Do NOT repeat any previous text. Hebrew only (no niqqud). Begin with 'HE:'."}
        ]
        aya_opts = {
            "temperature": options.get("temperature", 0.25),
            "top_p": options.get("top_p", 0.9),
            "top_k": options.get("top_k", 40),
            "repeat_penalty": options.get("repeat_penalty", 1.25),
            "repeat_last_n": options.get("repeat_last_n", 192),
            "num_predict": options.get("num_predict", 2048),
            "seed": options.get("seed", 0),
            "stop": options.get("stop", ["CONTINUE", "Please continue", "请继续", "继续", "继续翻译"]),
        }
        more = client.chat(model=model, messages=cont_messages, options=aya_opts)["message"]["content"].strip()
        if more.startswith("HE:"):
            more = more[3:].lstrip()
        he += " " + postprocess_hebrew(more)
    return he.strip()

# ---- light_fix_hebrew.py (drop this anywhere in your project) ----
import re

# meta/chit-chat we never want in the final text
_RE_META   = re.compile(r"(הנה תרגום לעברית|בסדר גמור|אנ[את] ספק|מה שלומך|אני מקווה|ננסה שוב)", re.I)
# niqqud/ta'amim + bidi marks
_RE_NIQQUD = re.compile(r"[\u0591-\u05C7]")
_RE_BIDI   = re.compile(r"[\u200e\u200f\u202a-\u202e]")
# collapse three-or-more repeated Hebrew words
_RE_REPEAT = re.compile(r"(\b[\u05D0-\u05EA]{2,}\b)(?:\s+\1){2,}")

# Stable, minimal term/gloss fixes (extend as you wish)
_GLOSS = {
    "בשם האל הגאוני": "בשם האל הרחמן והרחום",
    "בשם האל הרחמי": "בשם האל הרחמן והרחום",
    "רבוני העולמים": "ריבון העולמים",
    "רבוני": "ריבון",

    "אל-לאם": "אליף–לאם–מימ",
    "אל לאם": "אליף–לאם–מימ",
    "אליף–לאם–מים": "אליף–לאם–מימ",

    "כס האליל": "הכסא",
    "כיסא האליל": "הכסא",
    "כס האלילה": "הכסא",
    "כיסא האלילה": "הכסא",

    "יום ההשפעה": "יום הדין",

    "מן הגל": "מן העפר",
    "מקרקעין": "מן העפר",
    "ממים קטנים ומבוגרים": "מטיפה מבוזה",
    "וסיים לו את עצמותיו": "ועיצב את גופו",
    "וסיים לו מהנפש שלו": "והפיח בו מרוחו",

    "הוא הרחמים": "והרחום",
    "היוצר היוצר": "היוצר הטוב",

    "גיאנום": "גהנום",
    "גיהנם": "גהנום",  # unify spelling; remove this line if you prefer "גיהנם"
}

# --- add to _GLOSS ---
_GLOSS.update({
    # opening formula & tokens
    "בשם האל הרחמן והרחום, הגאון, הרחמי": "בשם האל הרחמן והרחום",
    "אליף–לאם–מימ. זהו ספר שהורש": "אליף–לאם–מימ. זהו ספר שהורד",
    "שהורש ": "שהורד ",

    # throne / cosmology
    "על הכסאה": "על הכסא",
    "בשש ימים ושש שנים": "בשישה ימים",

    # awkward verb choices
    "נתנו לו צאצא": "עשה את צאצאיו",
    "וסיים לו את עצמותיו": "ועיצב את גופו",
    "וקיבל לכם": "ונתן לכם",
    "לא מודה יותר לאורתו לעם": "מעט מודים בחסדיו",

    # judiciary / day of judgment
    "מחזיקים את המוות עד יום הדין": "מאמינים ביום הדין",

    # sinners scene
    "אילו יסתכלת": "ואילו ראית",
    "האלימים": "הפושעים",
    "כשהגבדים": "כאשר",

    # hellfire lines
    "תקחו את עונש הגיהנום": "טעמו את עונש הגהנום",

    # misc clarity
    "הוא היוצר הטוב הטוב ביותר": "היוצר הטוב",
    "אותינו": "אותותינו",
    "אל המשיח": "אל שער רחמי ריבונם",
    "השפעה שלהם": "שכרם",
    "יחידו בו": "ישכנו בו",
    "כי גניבו הם - האש": "מקומם – האש",

    # messy history lines
    "ואכלנו את בני ישראל": "ועשינו אותו דרך הדרכה לבני ישראל",
    "יפסק לך ה' את הדין": "ריבונך יכריע בדין",

    # bad tail section
    "הם נצלו בהן מתוך אודות האל. הם נעשים עבדים בעונש עונה.":
        "יש בכך אותות לכוח האל ולעונש הכואב לפושעים.",

    # water/land bit
    "אנו משחיקים אותם, ונוצר אותם בכוח": "אנו מזרימים מים אל הארץ החרבה",
    "ארבעת גבעותיהם": "בהמותיהם",
    "ומרותף את עצמם": "וגם הם ניזונים מהם",

    # victory line
    "מה זה הפסד שלכם": "מהו הניצחון",
    "ביום ההצגה": "ביום הניצחון",
    "לא יקבלו אהבה": "לא תועיל להם אמונה",
    "יסתובבו אל הגדרות ותתנו להם את ההמתנה": "ולא יינתן להם חסד",
})


# Pattern fixes for common skewed lines
_PAT_FIXES = [
    (re.compile(r"הוא יעלה את השמים אל הארץ"), "הוא מנהל את ענייני העולם מן השמים אל הארץ"),
    (re.compile(r"השמ[יי]ם מתכננים את דעתם אל הארץ"), "הוא מנהל את ענייני העולם מן השמים אל הארץ"),
]

# --- add to _PAT_FIXES ---
_PAT_FIXES.extend([
    # remove stray quotes at sentence starts like: הוא אמר:"טקסט"
    (re.compile(r'(:)\s*"', re.U), r'\1 "'),
    # compress doubled words like "הטוב הטוב" (extra safety)
    (re.compile(r'\b(\w+)\s+\1\b'), r'\1'),
])

RED_FLAGS = [
    "הגאון", "הרחמי", "בשש שנים", "הכסאה",
    "מחזיקים את המוות", "כשהגבדים", "מה זה הפסד",
    "ואכלנו את בני ישראל"
]
def warn_residuals(s: str):
    hits = [t for t in RED_FLAGS if t in s]
    if hits:
        print("[light-fix] residuals to review:", ", ".join(hits))

def light_fix_hebrew(text: str) -> str:
    s = text or ""
    # 1) strip niqqud/bidi
    s = _RE_NIQQUD.sub("", s)
    s = _RE_BIDI.sub("", s)

    # 2) remove meta/chat lines
    lines = [ln for ln in s.splitlines() if not _RE_META.search(ln.strip())]
    s = " ".join(lines)

    # 3) glossary replacements (surgical)
    for a, b in _GLOSS.items():
        s = s.replace(a, b)

    # 4) regex pattern fixes
    for pat, repl in _PAT_FIXES:
        s = pat.sub(repl, s)

    # 5) collapse 3+ repeated words, tidy punctuation & spaces
    s = _RE_REPEAT.sub(r"\1", s)
    s = re.sub(r'\s*"\s*', '"', s)      # normalize quote spacing
    s = re.sub(r"([.!?])\1+", r"\1", s) # collapse repeated punctuation
    s = re.sub(r"\s+", " ", s).strip()

    return s

# ---------- end-to-end (only small additions) ----------
def translate_fa_to_he_chunked(text_fa: str, model: str, host: str, max_chars:int) -> str:
    client = Client(host=host)
    options = {
        "temperature": 0.2,
        "top_p": 0.9,
        "repeat_penalty": 1.22,
        "num_predict": 4096,
        "seed": 0,
        # used by Aya path if present:
        "top_k": 40,
        "repeat_last_n": 192,
        "stop": ["CONTINUE", "Please continue", "请继续", "继续", "继续翻译"],
    }
    chunks = chunk_persian(text_fa, max_chars=max_chars)
    print(f"[debug] chunk_count={len(chunks)} | max_chars={max_chars}")
    if chunks:
        print(f"[debug] first_chunk_len={len(chunks[0])}")

    out = []

    use_aya = "aya" in (model or "").lower()

    use_gemma = "gemma" in (model or "").lower()

    for i, ch in enumerate(chunks, 1):
        print(f"[debug] translating chunk {i}/{len(chunks)} (len={len(ch)})...")
        if use_aya:
            he = translate_chunk_aya(client, model, ch, options)
        elif use_gemma:
            print(f"[debug] gemma is used...")
            # he = translate_chunk_gemma(client, model, ch, gemma_opts)
            # if looks_bad(he):  # last resort fallback
                # print(f"[debug] looks bad...")
            he = fa_to_he_nllb(ch)

            he = light_fix_hebrew(he)
            warn_residuals(he)

        else:
            he = translate_chunk(client, model, ch, options)
        out.append(he)
    return "\n\n".join(out).strip()



if __name__ == "__main__":
    # main()
    # call_translate.py

    persian_text = """
    به نام خداوند بخشنده بخشایشگر الف لام میم ‌این کتابیست که از سوی پروردگار جهانیان نازل شده و شک و تردیدی درانیست ولی آنان می‌گویند محمد آن را به دروغ به خدا بسته است اما این سخن حقیقت از سوی پروردگارت تا گروهی را انکار کنی که پیش از تو هیچ انکار کنده‌ی برای آنان نیابده شاید پنجیرند و هدایت شوند خداوند کسی است که آسمان‌ها و زمین و آنچه را میان این دوست در شش روز و شش دوران آفرید سپس بر عرش قدرت قرار گره هیچ سرپرست و شفاعت کنده‌ی برای شما جز او نیست آیا متذکر نمی‌شدید امور این جهان را از آسمان به سوی زمین تدبیر می‌کند سپس در روزی که مقدار آن هزار سال از سالهاست که شما می‌شمارید به سوی او بالا می‌آورد و دنیا پایان می‌آورد او خداوندیست که از پنهان ناشکار باخبر است و شکست‌ناپذیر و مهربان است او همان کسیست که هرجا آفرید نیکو آفرید و آفرینش انسان را از جل آغاز کرد سپس نسل او را از اشاره از آب ناچیز و بی قد را دهید سپس اندام او را موزون ساخت و از روح خویش در بی امید و برای شما گوش و چشم‌ها و دل‌ها قرار داد اما کمتر شکر نعمت‌های او را به جامعه داد آن‌ها گفتند آیا هنگامی ‌که ما مردیم و در زمین گم شدیم آفرینش تازهی خواهیم یا ولی آنان لای پروردارشم را امکار می‌کند و می‌خواهند با انکار معاد آزادانه به سخنرانی خویش ادامه دهند بگو فرشته مرگ که بر شما معمول شده روح شما را می‌گیرند سپس شما را به سوی پروردگارتان باز می‌گرداند و اگر بینی مجرمان را هنگامی‌ که در پیشگاه پروردگارشان سربازی رفت کرده می‌گویند پروردگار را آنچه وعده کرده بودی دیدیم و شنیدیم ما را بازگردان تا کار شایسته انجام رهیم ما به قیامت نقیم داریم و اگر می‌خواستیم به هر انسانی هدایت لازمش را از روی اجبار بدهیم می‌دادیم ولی من آن‌ها را آزاد گذارده هم و سخن و عدم حق هست که دوزخ را از افراد بی‌ایمان و گنهکار از جن و انس همهی پرده هم و به آن‌ها می‌گویم بچشید عذاب جهنم را به خاطر این‌که دیدار امروزتان را فراموش کردیم ما نیز شما را فراموش کردیم و باید عذاب جاودان را به خاطر اعمالی که انجام می‌دیدید تن‌ها کسانی که به آیات ما ایمان می‌آورند که هر وقت این آیات به آنان یاد آوری شود به سجده می‌افرند و تسبیح و حمد پروردارشان را به جا می‌آورند و تکبر نمی‌کند فهرو‌هایشان از بستر‌ها در دل شب دور می‌شود و به پا می‌خیزند و روبه در گاه خدا می‌آورند و پروردگار خود را با بیم و امید می‌خواند بعضان که به آنان روزی داده این اتفاق می‌کند هیچ‌کس نمی‌داند چه پاداش‌های مهمی که مای روشنی چشم هاست برای آن‌ها نهاده شده این پاداش کارهایست که انجام می‌دهد آیا کسی که با ایمان باشد همچنان کسیست که فاسق است نه هرگز این دو برابر نیستند اما کسانی که ایمان آوردن و کار‌های شایسته انجام دادند بال‌های بهشت جاویدان از آن آن‌ها خواهد بود این وسیله پذیرای خداوند از آن‌ها است به پاداش آنچه انجام دادند و اما کسانی که پاسخ شدند و از اطاعت خدا سربازدادند جایگاه همیشگی آن‌ها آتش است هر زمان بخواهند از آن خارج شوند آن‌ها را به آن باز می‌گرداند و به آنان گفته می‌شود باشید عذاب آتشی را که انکار می‌کند به آنان از عذاب نزدیک عذاب این دنیا پیش از عذاب بزرگ آخرت می‌چشم شاید باز کرد چه کسی ستمکار پرست از آن کسی که آیات پروردارش به او یاداوری شده و او از آن اعراض کرده است مسلمان ما از مجرمان انتقام خواهیم گره ما به موسا کتاب آسمانی دادیم و شک نداشته باشد که او آیات الهی را دریافت داشت و ما آن را وسیله هدایت بنی‌اسرائیل خوردیم و از آنان امامان و پیشوایانی قرار دادیم که به فرمان ما مردم را هدایت می‌کندند چون شکیبایی نمودند و به آیات ما یقین داشتند البته پروردگار تو میان آنان روز قیامت در آنچه اختلاف داشتند داوری می‌کند و هر کس را به سزای اعمالش می‌رساند آیا برای هدایت آن‌ها همین کافی نیست که افراد زیادی را که در قرن پیش از آنان زندگی داشتند حاکم کردیم این‌ها در مساکن دیگران شده آنان راه می‌روند در این آیاتیست از قدرت خداوند و مجازات دردناک مجرمان آیاهی می‌شوند آیا ندیدن که ما آب را به سوی زمین‌های خشک می‌رانیم و به وسیله آن زراعت‌های نیرویانیم که هم چهار پایانشان از آن می‌خورند و هم خودشان تغییر می‌کند آیا نمی‌بیند آنان می‌گویند اگر راست می‌گوید این پیروزی شما چی خواهد بود بگو روز پیروزی ایمان آوردن سودی به حال کافران نخواهد داشت و به آن‌ها هیچ محبت داده نمی‌شود حال که چونی نزد از آن‌ها روی به جردان و منتظر باش آن‌ها نیز منتظرند تو منتظر رحمت خدا و آن‌ها هم منتظر عذاب
    """
    print(f"[debug] input chars={len(persian_text)}")
    print(f"[debug] first 500 chars of INPUT =\n{persian_text[:500]}")

    hebrew = translate_fa_to_he_chunked(
        persian_text,
        max_chars=250,
        model=config.OLLAMA_MODEL_TAG,  # e.g., "aya:8b"
        host=config.OLLAMA_URL
    )
    print(hebrew)