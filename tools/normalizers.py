import logging
from pathlib import Path
from typing import Optional

from tools.lm_fa import get_persian_sp_kenlm
from tools.persian_normalize.context_aware_normalizer import pipeline_clean
from tools.persian_normalize.nevis import NeviseCorrector
from tools.persian_normalize.persian_normalizer import persian_normalizer
from hazm import Normalizer as HazmNormalizer
import config
from tools.spellcheck_fa import correct_fa_line, build_symspell_fa, correct_fa_line_safe, \
    correct_fa_line_safer_blockwise
from tools.utils import cleaned_filename

import re
from hebrew import Hebrew


logger = logging.getLogger(__name__)


# Load once at import time
_nevise_corrector = NeviseCorrector(config.NEVISE_VOCAB, config.NEVISE_CKPT)
_hazm_normalizer  = HazmNormalizer()


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

# ─── Persian Pre-clean ─────────────────────────────────────────────────────────
def remove_duplicate_words(input):
    # Regex to matching repeated words
    regex = r'\b(\w+)(?:\W+\1\b)+'
    return re.sub(regex, r'\1', input, flags=re.IGNORECASE)

def handle_duplicates_alphabet(text):
    """Normalizes repeated characters in words based on language rules.

    Rules:
        - Persian/Arabic: Removes ALL consecutive duplicates (سلامم → سلام)
        - English: Allows max 2 repeats (Heeelllo → Heelloo)
        - Numbers: Unchanged

    Args:
        text (str): Input text with mixed languages/numbers.

    Returns:
        str: Text with processed words.

    """

    def process_word(word):
        # Check if the word is a number (do nothing)
        if word.isdigit():
            return word

        # Check if the word is Persian (delete all duplicates)
        if re.search(r'[\u0600-\u06FF]', word):  # Persian/Arabic Unicode range
            processed = re.sub(r'(.)\1+', r'\1', word)
            return processed

        # For English words (keep max 2 duplicates, remove 3+)
        processed = re.sub(r'(.)\1{2,}', r'\1\1', word)
        return processed

    # Split into words and process each one
    tokens = re.findall(r'(\s+|\d+|\w+|[^\w\s])', text)
    return ''.join([process_word(token) for token in tokens])

def filter_allowed_chars(text):
    """Filters text to allow only Persian/English letters, numbers, and common punctuation.

    Allowed:
        - Persian: \u0600-\u06FF + پ (067E), چ (0686), etc.
        - English: A-Za-z
        - Numbers: 0-9 and Persian digits (\u0660-\u0669)
        - Punctuation: English (!@#$) + Persian (،؛؟)

    Args:
        text (str): Raw input text with possible invalid characters.

    Returns:
        str: Sanitized text with disallowed characters removed.

    """

    allowed_pattern = re.compile(
        r'['
        r'\u0600-\u06FF\u067E\u0686\u06AF\u0698' # Persian
        r'A-Za-z'  # English
        r'0-9\u0660-\u0669'  # \number
        r'\s'  # space
        r'!@#$%\^&\*\-_=\+\{\};:\'",<>\.\/?\|~`'  # Enlish puctuation marks
        r'،؛؟«»'  # Persian puctuation marks
        r']',
        flags=re.UNICODE
    )

    cleaned_text = ''.join(allowed_pattern.findall(text))
    return cleaned_text
# ───────────────────────────────────────────────────────────────────

# ─── Public API ────────────────────────────────────────────────────────────────


sym = build_symspell_fa(Path("persian_normalize/wiki_fa_80k.txt"), max_edit_distance=2)
LM_PATH  = "files/jomleh-sp-57218-o3-prune011.probing"  # <- your file
SPM_PATH = "files/jomleh-sp-57218-o3.model"             # <- matching spm model

lm_scorer = get_persian_sp_kenlm(LM_PATH, SPM_PATH)



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
        logger.info(f"[Cleaning] original text [pre-clean] = {text!r}")
        s1 = handle_duplicates_alphabet(text)
        s2 = remove_duplicate_words(s1)
        s3 = filter_allowed_chars(s2)
        logger.info(f"len of text = {len(text)} len of s1 = {s1!r}, s2 = {s2!r}, s3 = {s3!r}")
        logger.info(f"[Cleaning] cleaned text [pre-clean] = {s3!r}")

        logger.info(f"[Cleaning] original text [pre-Nevis] = {text!r}")
        step1 = nevis_spell_check(s3)
        step2 = persian_base_normalize(step1)
        step3 = hazm_normalize(step2) #maybe redundant
        logger.info(f"[Cleaning] cleaned text [Nevised] = {s3!r}")

        # logger.info(f"[Cleaning] cleaned text [Pre KenLM] = {s3!r}")
        # step4 = correct_fa_line(
        #     step3,
        #     sym,
        #     mode="top",
        #     max_edit_distance=2,
        #     min_freq=5,
        #     lm=lm_scorer,  # <- plug it in (None-safe if loading failed)
        # )
        # logger.info(f"[Cleaning] cleaned [KenLM] = {step4!r}")

        return step3

    elif language == "he":
        return normalize_hebrew(text, split_maqaf=True)  # Hebrew

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
