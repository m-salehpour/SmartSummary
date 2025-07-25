import logging
from pathlib import Path
from typing import Optional

from tools.persian_normalize.context_aware_normalizer import pipeline_clean
from tools.persian_normalize.nevis import NeviseCorrector
from tools.persian_normalize.persian_normalizer import persian_normalizer
from hazm import Normalizer as HazmNormalizer
import config
from tools.utils import cleaned_filename

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
