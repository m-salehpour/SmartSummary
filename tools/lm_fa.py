# tools/lm_fa.py
from pathlib import Path
from functools import lru_cache

try:
    import kenlm   # pip install https://github.com/kpu/kenlm
except Exception:
    kenlm = None

try:
    import sentencepiece as spm  # pip install sentencepiece
except Exception:
    spm = None


class KenLMSpScorer:
    """
    Callable: score(text) -> float
    Uses SentencePiece to tokenize, then KenLM to score.
    Returns avg log10 per token (higher is better).
    """
    def __init__(self, lm_path: Path, spm_path: Path):
        if kenlm is None:
            raise RuntimeError("kenlm is not installed")
        if spm is None:
            raise RuntimeError("sentencepiece is not installed")
        self.lm = kenlm.Model(str(lm_path))
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_path))

    def __call__(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return float("-1e9")
        toks = self.sp.encode(text, out_type=str)
        # KenLM expects space-separated tokens
        s = " ".join(toks)
        score = self.lm.score(s, bos=True, eos=True)
        return score / max(len(toks), 1)  # length-normalized


@lru_cache(maxsize=1)
def get_persian_sp_kenlm(lm_path: str | Path, spm_path: str | Path):
    """
    Load & cache the jomleh SentencePiece+KenLM scorer.
    Returns a callable or None if anything is missing.
    """
    try:
        return KenLMSpScorer(Path(lm_path), Path(spm_path))
    except Exception:
        return None
