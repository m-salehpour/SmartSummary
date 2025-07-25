import logging
from typing import Optional

from tools.persian_normalize.nevis import NeviseCorrector
from tools.persian_normalize.persian_normalizer import persian_normalizer
from hazm import Normalizer as HazmNormalizer
import config


logger = logging.getLogger(__name__)


# Load once at import time
_nevise_corrector = NeviseCorrector(config.NEVISE_VOCAB, config.NEVISE_CKPT)
_hazm_normalizer  = HazmNormalizer()

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

    if language == "fa":
        step1 = nevis_spell_check(text)
        step2 = persian_base_normalize(step1)
        step3 = hazm_normalize(step2) #maybe redundant
        return step3

    # fallback: no cleaning for other langs
    return text
