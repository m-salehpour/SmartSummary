#!/usr/bin/env python3
# fa_to_he_m2m.py
# ------------------------------------------------------------
# Translate Persian (fa) → Hebrew (he) with M2M100.
# - Chunks long input by paragraphs & sentences
# - Runs on CPU, fits easily in ~8 GB RAM
#
# Usage:
#   python fa_to_he_m2m.py --in fa.txt > he.txt
#   cat fa.txt | python fa_to_he_m2m.py
#
# Install:
#   pip install transformers torch sentencepiece
# ------------------------------------------------------------

import argparse
import re
import sys
from typing import List

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


MODEL_NAME = "facebook/m2m100_418M"  # small, reliable MT model


# ---------- Simple Persian-aware chunking ----------

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?؟])\s+')  # split after ., !, ?, Arabic ? (؟)


def split_paragraphs(text: str) -> List[str]:
    """Split by blank lines; keep non-empty paragraphs."""
    parts = re.split(r'\n\s*\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_paragraph(paragraph: str, max_chars: int = 900) -> List[str]:
    """
    Split a paragraph into sentence groups <= max_chars.
    Tries to avoid breaking mid-sentence.
    """
    sents = [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]
    if not sents:
        return [paragraph[:max_chars]]

    chunks, buf = [], ""
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


def chunk_text(text: str, max_chars: int = 900) -> List[List[str]]:
    """
    Return a list of paragraphs, where each paragraph is a list of chunks.
    """
    paragraphs = split_paragraphs(text)
    return [chunk_paragraph(p, max_chars=max_chars) for p in paragraphs]


# ---------- Optional Hebrew cleanup (conservative) ----------

NIQQUD = re.compile(r'[\u0591-\u05C7]')                      # vowels/ta'amim
NON_HEB = re.compile(r'[^\u05D0-\u05EA0-9\s.,;:!?()"\-׳״]+')  # basic whitelist
WS = re.compile(r'[ \t]+')


def clean_he(text: str) -> str:
    # Most outputs are already without niqqud; this is a gentle safeguard.
    text = NIQQUD.sub('', text)
    text = NON_HEB.sub(' ', text)
    text = WS.sub(' ', text)
    return text.strip()


# ---------- Model loader & translator ----------

def load_model(model_name: str = MODEL_NAME):
    tok = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tok, model, device

import re, torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

RE_WORD = re.compile(r"[^\s]+")
def is_degenerate(text: str, max_ratio: float = 0.20) -> bool:
    """True if one word dominates (e.g., 'אלהים' repeated)."""
    words = RE_WORD.findall(text)
    if len(words) < 6:
        return False
    from collections import Counter
    [(w,c)] = Counter(words).most_common(1)
    return c / len(words) >= max_ratio

@torch.no_grad()
def translate_chunk_fa_to_he(tok, model, device, fa_text: str,
                             beams: int = 5, max_new_tokens: int = 512) -> str:
    tok.src_lang = "fa"  # IMPORTANT for M2M100
    enc = tok(fa_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Pass 1: beam search with anti-repeat
    gen = model.generate(
        **enc,
        forced_bos_token_id=tok.get_lang_id("he"),
        num_beams=beams,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=1.2,        # try 1.25 if needed
        length_penalty=1.0,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        renormalize_logits=True,
    )
    he = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    # If collapsed, Pass 2: light sampling (still constrained)
    if is_degenerate(he):
        gen = model.generate(
            **enc,
            forced_bos_token_id=tok.get_lang_id("he"),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=3,
            repetition_penalty=1.25,
            max_new_tokens=max_new_tokens,
            renormalize_logits=True,
        )
        he = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

    return clean_he(he)


TRY_NLLB = True
NLLB_MODEL = "facebook/nllb-200-distilled-600M"  # ~600M; still fine on 8GB
device="cpu"
_nllb_tok = _nllb_model = None
def _load_nllb():
    global _nllb_tok, _nllb_model
    if _nllb_tok is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _nllb_tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device).eval()

def translate_chunk_fa_to_he_nllb(fa_text: str, max_new_tokens=512) -> str:
    _load_nllb()
    enc = _nllb_tok(fa_text, return_tensors="pt", padding=True, truncation=True).to(device)
    gen = _nllb_model.generate(
        **enc,
        forced_bos_token_id=_nllb_tok.lang_code_to_id["heb_Hebr"],
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
    )
    return clean_he(_nllb_tok.batch_decode(gen, skip_special_tokens=True)[0])



def translate_fa_to_he(text_fa: str, max_chars: int = 900,
                       beams: int = 4, max_new_tokens: int = 512) -> str:
    tok, model, device = load_model(MODEL_NAME)
    para_chunks = chunk_text(text_fa, max_chars=max_chars)

    he_paragraphs: List[str] = []
    for chunks in para_chunks:
        he_chunks = []
        for ch in chunks:
            he = translate_chunk_fa_to_he_nllb(ch)
            he_chunks.append(he)
        he_paragraphs.append("\n".join(he_chunks))  # keep paragraph separation
    return "\n\n".join(he_paragraphs).strip()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Translate Persian (fa) → Hebrew (he) with M2M100, chunked.")
    ap.add_argument("--in", dest="inp", help="Input Persian file (UTF-8). Omit to read stdin.")
    ap.add_argument("--max_chars", type=int, default=900, help="Max characters per chunk.")
    ap.add_argument("--beams", type=int, default=4, help="Beam size for generation.")
    ap.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per chunk.")
    args = ap.parse_args()

    text_fa = (open(args.inp, "r", encoding="utf-8").read()
               if args.inp else sys.stdin.read())
    text_fa = text_fa.strip()
    if not text_fa:
        sys.exit("No input text provided.")

    he = translate_fa_to_he(
        text_fa,
        max_chars=args.max_chars,
        beams=args.beams,
        max_new_tokens=args.max_new_tokens,
    )
    print(he)


if __name__ == "__main__":
    # main()
    persian_text = """
         به نام خداوند بخشنده بخشایشگر الف لام میم ‌این کتابیست که از سوی پروردگار جهانیان نازل شده و شک و تردیدی درانیست ولی آنان می‌گویند محمد آن را به دروغ به خدا بسته است اما این سخن حقیقت از سوی پروردگارت تا گروهی را انکار کنی که پیش از تو هیچ انکار کنده‌ی برای آنان نیابده شاید پنجیرند و هدایت شوند خداوند کسی است که آسمان‌ها و زمین و آنچه را میان این دوست در شش روز و شش دوران آفرید سپس بر عرش قدرت قرار گره هیچ سرپرست و شفاعت کنده‌ی برای شما جز او نیست آیا متذکر نمی‌شدید امور این جهان را از آسمان به سوی زمین تدبیر می‌کند سپس در روزی که مقدار آن هزار سال از سالهاست که شما می‌شمارید به سوی او بالا می‌آورد و دنیا پایان می‌آورد او خداوندیست که از پنهان ناشکار باخبر است و شکست‌ناپذیر و مهربان است او همان کسیست که هرجا آفرید نیکو آفرید و آفرینش انسان را از جل آغاز کرد سپس نسل او را از اشاره از آب ناچیز و بی قد را دهید سپس اندام او را موزون ساخت و از روح خویش در بی امید و برای شما گوش و چشم‌ها و دل‌ها قرار داد اما کمتر شکر نعمت‌های او را به جامعه داد آن‌ها گفتند آیا هنگامی ‌که ما مردیم و در زمین گم شدیم آفرینش تازهی خواهیم یا ولی آنان لای پروردارشم را امکار می‌کند و می‌خواهند با انکار معاد آزادانه به سخنرانی خویش ادامه دهند بگو فرشته مرگ که بر شما معمول شده روح شما را می‌گیرند سپس شما را به سوی پروردگارتان باز می‌گرداند و اگر بینی مجرمان را هنگامی‌ که در پیشگاه پروردگارشان سربازی رفت کرده می‌گویند پروردگار را آنچه وعده کرده بودی دیدیم و شنیدیم ما را بازگردان تا کار شایسته انجام رهیم ما به قیامت نقیم داریم و اگر می‌خواستیم به هر انسانی هدایت لازمش را از روی اجبار بدهیم می‌دادیم ولی من آن‌ها را آزاد گذارده هم و سخن و عدم حق هست که دوزخ را از افراد بی‌ایمان و گنهکار از جن و انس همهی پرده هم و به آن‌ها می‌گویم بچشید عذاب جهنم را به خاطر این‌که دیدار امروزتان را فراموش کردیم ما نیز شما را فراموش کردیم و باید عذاب جاودان را به خاطر اعمالی که انجام می‌دیدید تن‌ها کسانی که به آیات ما ایمان می‌آورند که هر وقت این آیات به آنان یاد آوری شود به سجده می‌افرند و تسبیح و حمد پروردارشان را به جا می‌آورند و تکبر نمی‌کند فهرو‌هایشان از بستر‌ها در دل شب دور می‌شود و به پا می‌خیزند و روبه در گاه خدا می‌آورند و پروردگار خود را با بیم و امید می‌خواند بعضان که به آنان روزی داده این اتفاق می‌کند هیچ‌کس نمی‌داند چه پاداش‌های مهمی که مای روشنی چشم هاست برای آن‌ها نهاده شده این پاداش کارهایست که انجام می‌دهد آیا کسی که با ایمان باشد همچنان کسیست که فاسق است نه هرگز این دو برابر نیستند اما کسانی که ایمان آوردن و کار‌های شایسته انجام دادند بال‌های بهشت جاویدان از آن آن‌ها خواهد بود این وسیله پذیرای خداوند از آن‌ها است به پاداش آنچه انجام دادند و اما کسانی که پاسخ شدند و از اطاعت خدا سربازدادند جایگاه همیشگی آن‌ها آتش است هر زمان بخواهند از آن خارج شوند آن‌ها را به آن باز می‌گرداند و به آنان گفته می‌شود باشید عذاب آتشی را که انکار می‌کند به آنان از عذاب نزدیک عذاب این دنیا پیش از عذاب بزرگ آخرت می‌چشم شاید باز کرد چه کسی ستمکار پرست از آن کسی که آیات پروردارش به او یاداوری شده و او از آن اعراض کرده است مسلمان ما از مجرمان انتقام خواهیم گره ما به موسا کتاب آسمانی دادیم و شک نداشته باشد که او آیات الهی را دریافت داشت و ما آن را وسیله هدایت بنی‌اسرائیل خوردیم و از آنان امامان و پیشوایانی قرار دادیم که به فرمان ما مردم را هدایت می‌کندند چون شکیبایی نمودند و به آیات ما یقین داشتند البته پروردگار تو میان آنان روز قیامت در آنچه اختلاف داشتند داوری می‌کند و هر کس را به سزای اعمالش می‌رساند آیا برای هدایت آن‌ها همین کافی نیست که افراد زیادی را که در قرن پیش از آنان زندگی داشتند حاکم کردیم این‌ها در مساکن دیگران شده آنان راه می‌روند در این آیاتیست از قدرت خداوند و مجازات دردناک مجرمان آیاهی می‌شوند آیا ندیدن که ما آب را به سوی زمین‌های خشک می‌رانیم و به وسیله آن زراعت‌های نیرویانیم که هم چهار پایانشان از آن می‌خورند و هم خودشان تغییر می‌کند آیا نمی‌بیند آنان می‌گویند اگر راست می‌گوید این پیروزی شما چی خواهد بود بگو روز پیروزی ایمان آوردن سودی به حال کافران نخواهد داشت و به آن‌ها هیچ محبت داده نمی‌شود حال که چونی نزد از آن‌ها روی به جردان و منتظر باش آن‌ها نیز منتظرند تو منتظر رحمت خدا و آن‌ها هم منتظر عذاب
        """
    he = translate_fa_to_he(
        persian_text,
        max_chars=900,
        beams=4,
        max_new_tokens=512,
    )
    print(he)

