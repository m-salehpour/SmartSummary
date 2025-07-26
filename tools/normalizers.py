import logging
from pathlib import Path
from typing import Optional

from tools.persian_normalize.context_aware_normalizer import pipeline_clean
from tools.persian_normalize.nevis import NeviseCorrector
from tools.persian_normalize.persian_normalizer import persian_normalizer
from hazm import Normalizer as HazmNormalizer
import config
from tools.utils import cleaned_filename

import re
from hebrew import Hebrew


logger = logging.getLogger(__name__)


# Load once at import time
_nevise_corrector = NeviseCorrector(config.NEVISE_VOCAB, config.NEVISE_CKPT)
_hazm_normalizer  = HazmNormalizer()

# ─── English Cleaning Steps ─────────────────────────────────────────────────────────────
# tools/english_normalizer.py
import re
import unicodedata
from typing import Literal

import ftfy
from sacremoses import MosesPunctNormalizer
import contractions

# optional, only if need number normalization
try:
    from word2number import w2n
except Exception:
    w2n = None
try:
    from num2words import num2words
except Exception:
    num2words = None

_PNORM = MosesPunctNormalizer()
_WS = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")  # 3,000 -> 3000
_QUOTES = {
    "“": '"', "”": '"', "‟": '"', "„": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "—": "-", "–": "-", "-": "-", "‒": "-",
    "…": "..."
}
_FILLERS = re.compile(r"\b(uh|um|erm|ah|eh|hmm)\b", flags=re.IGNORECASE)

def normalize_numbers(text: str, mode: Optional[Literal["digits","words"]] = None) -> str:
    if mode is None:
        return text
    if mode == "digits" and w2n:
        # very simple pass: convert isolated number phrases; keep it conservative
        def _wordnum_to_int(m):
            try:
                return str(w2n.word_to_num(m.group(0)))
            except Exception:
                return m.group(0)
        # examples like "three thousand", "seventy"
        return re.sub(r"\b([a-z][a-z -]*)\b", _wordnum_to_int, text, flags=re.IGNORECASE)
    if mode == "words" and num2words:
        def _int_to_words(m):
            try:
                return num2words(int(m.group(0)))
            except Exception:
                return m.group(0)
        return re.sub(r"\b\d+\b", _int_to_words, text)
    return text

def normalize_english(
    text: str,
    *,
    level: Literal["safe","moderate","aggressive"] = "safe",
    numbers: Optional[Literal["digits","words"]] = None,
    lowercase: Optional[bool] = None
) -> str:
    """
    Canonicalize English for ASR WER.
    - level=safe: unicode fix, punct spacing, unify quotes/dashes, thousands commas removal.
    - level=moderate: + expand contractions, optional lowercase.
    - level=aggressive: + remove filler tokens.
    - numbers: None (leave), "digits" (words->digits), or "words" (digits->words).
    """
    if not isinstance(text, str):
        return text

    # 1) Unicode sanity
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)

    # 2) Punctuation normalization (Moses)
    text = _PNORM.normalize(text)

    # 3) Unify quotes/dashes/ellipsis
    for k, v in _QUOTES.items():
        text = text.replace(k, v)

    # 4) Numbers: remove thousands separators for consistent token match
    text = _THOUSANDS.sub("", text)

    if level in ("moderate","aggressive"):
        # 5) Expand contractions
        text = contractions.fix(text)

    # Optional number canonicalization (pick to match reference)
    text = normalize_numbers(text, numbers)

    if level == "aggressive":
        # 6) Remove filler tokens
        text = _FILLERS.sub(" ", text)

    # lowercase decision:
    if lowercase is None:
        lowercase = (level != "safe")
    if lowercase:
        text = text.lower()

    # 7) collapse spaces
    text = _WS.sub(" ", text).strip()
    return text

ORDINALS = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
TO_VARIANTS = {"to-day":"today","to-night":"tonight","to-morrow":"tomorrow"}
THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


def ref_friendly_english(text: str) -> str:
    # 1) robust baseline: lowercase, punctuation normalized, digits
    s = normalize_english(
        text,
        level="safe",     # unicode fix, punct normalization, contractions handled
        numbers="words",
        lowercase=False
    )
    # 2) ordinals -> plain digits
    s = ORDINALS.sub(r"\1", s)
    # 3) drop thousands commas
    s = THOUSANDS.sub("", s)
    # 4) archaic hyphenated forms to modern
    for k, v in TO_VARIANTS.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)
    # 5) remove remaining punctuation (optional but safest for WER)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Hebrew Cleaning Steps ─────────────────────────────────────────────────────────────
THOUSANDS = re.compile(r'(?<=\d),(?=\d)')  # 3,000 -> 3000

def normalize_hebrew(text: str,
                     *,
                     remove_taamim=True,
                     remove_niqqud=True,
                     remove_punct=True,
                     split_maqaf=False) -> str:
    h = Hebrew(text).normalize()                 # fix special/ligature chars
    if remove_taamim:
        h = h.no_taamim()
    if remove_niqqud and not remove_punct:
        h = h.no_niqqud()                        # keep punctuation, drop vowels
    if remove_punct:
        # removes niqqud & punctuation (and optionally maqaf)
        h = h.text_only(remove_maqaf=split_maqaf)

    s = h.string
    s = THOUSANDS.sub('', s)                     # “3,000” → “3000”
    s = s.replace('־', '-')                      # unify maqaf/hyphen if desired
    return re.sub(r'\s+', ' ', s).strip()


# ─── Cleaning Steps ─────────────────────────────────────────────────────────────

def nevis_spell_check(text: str) -> str:
    """
    Run Nevise spell‐checker on Persian text.
    """
    logging.info("🔎 Spell-checking Persian with Nevise…")
    corrected = _nevise_corrector.clean(text)
    logging.debug(f"[Nevise] corrected = {corrected!r}")
    return corrected

def persian_base_normalize(text: str) -> str:
    """
    Run Innocentive’s Persian normalizer.
    """
    normalized = persian_normalizer({"sentence": text}, return_dict=False)
    logging.debug(f"[PersianNormalizer] normalized = {normalized!r}")
    return normalized

def hazm_normalize(text: str) -> str:
    """
    Run Hazm’s normalization on Persian text.
    """
    normalized = _hazm_normalizer.normalize(text)
    logging.debug(f"[Hazm] normalized = {normalized!r}")
    return normalized

# ─── Public API ────────────────────────────────────────────────────────────────

def cleaning(text: str, language: Optional[str] = None) -> Optional[str]:
    """
    High-level text cleaner.
    - If `language == "fa"`, applies:
         1) Nevise spell-check
         2) Innocentive Persian normalizer
         3) Hazm normalizer
    - Otherwise, returns the original text.
    """
    if not isinstance(text, str):
        return None

    if language == "fa": # Persian

        step1 = nevis_spell_check(text)
        step2 = persian_base_normalize(step1)
        step3 = hazm_normalize(step2) #maybe redundant
        return step3

    elif language == "he": return normalize_hebrew(text, split_maqaf=True) # Hebrew

    elif language == "en": return ref_friendly_english(text) # English

    # fallback: no cleaning for other langs
    return text


def _run_llm_clean(input_json: Path, suffix: str) -> Path:
    """
    Runs pipeline_clean.main on the given JSON and returns
    the path to the new JSON (with the given suffix).
    """
    out_json = Path(cleaned_filename(str(input_json), suffix=suffix))
    print(f"\n🔄 Running LLM cleaner ({suffix}) → {out_json}")
    pipeline_clean.main(input_json, out_json)
    print(f"✅ LLM-cleaned transcript ({suffix}) saved to {out_json}")
    return out_json
