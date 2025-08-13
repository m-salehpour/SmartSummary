# tools/spellcheck_fa.py
from pathlib import Path
import re
from symspellpy import SymSpell, Verbosity

_PERSIAN_LETTERS = r"\u0600-\u06FF\u200c"   # Arabic block + ZWNJ
RE_TOKEN = re.compile(rf"[{_PERSIAN_LETTERS}]+|[A-Za-z]+|\d+|[^\s{_PERSIAN_LETTERS}]+", re.UNICODE)

def build_symspell_fa(
    dict_path: Path,
    max_edit_distance: int = 2,
    prefix_length: int = 7,
) -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    ok = sym.load_dictionary(str(dict_path), term_index=0, count_index=1, encoding="utf-8")
    if not ok:
        raise RuntimeError(f"Failed to load SymSpell dict: {dict_path}")
    return sym

def _should_skip(token: str) -> bool:
    # Don’t “correct” numbers, URLs, emails, Latin, or super-short tokens
    if token.isdigit():
        return True
    if re.match(r"https?://|www\.", token):
        return True
    if "@" in token or "." in token and any(c.isalpha() for c in token):
        return True
    if re.match(r"^[A-Za-z]+$", token):  # English
        return True
    if len(token) <= 1:
        return True
    return False

def correct_fa_line_compound(text: str, sym: SymSpell, max_edit_distance: int = 2) -> str:
    # Useful when spacing/merge errors are common
    res = sym.lookup_compound(text, max_edit_distance=max_edit_distance)
    return (res[0].term if res else text)


import re
from typing import List, Tuple, Optional
from symspellpy import Verbosity

# --- helpers ---------------------------------------------------------------

_PERSIAN_LETTERS = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
_WORD_RE = re.compile(f"([{_PERSIAN_LETTERS}]+)", re.UNICODE)

_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_LATIN_RE = re.compile(r"[A-Za-z]")
_NUM_RE   = re.compile(r"^\d+([.,:/-]\d+)*$")

def _tokenize_preserve_seps(text: str) -> List[Tuple[str, bool]]:
    """Split into [(chunk, is_word)] keeping separators/spaces untouched."""
    out: List[Tuple[str, bool]] = []
    last = 0
    for m in _WORD_RE.finditer(text):
        if m.start() > last:
            out.append((text[last:m.start()], False))   # separator
        out.append((m.group(0), True))                   # word
        last = m.end()
    if last < len(text):
        out.append((text[last:], False))
    return out

def _is_protected_token(tok: str) -> bool:
    """Tokens we should not touch."""
    if len(tok) < 3:
        return True
    if _URL_RE.match(tok) or _EMAIL_RE.match(tok):
        return True
    if _LATIN_RE.search(tok):
        return True
    if _NUM_RE.match(tok):
        return True
    return False

def _normalize_spaces_punct(s: str) -> str:
    """Light, non-destructive spacing fix (don’t collapse original seps)."""
    # Trim stray spaces before Persian/Latin punctuation
    s = re.sub(r"\s+([،؛؟,!?:;])", r"\1", s)
    # Ensure single space after punctuation if followed by a letter
    s = re.sub(r"([،؛؟,!?:;])([^\s])", r"\1 \2", s)
    # Collapse ridiculous space runs
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

# Optional LM scorer interface (if you have it):
# def lm_score(text: str) -> float: ...
# pass it into correct_fa_line(..., lm=lm_score) if available


# --- main API --------------------------------------------------------------

def correct_fa_line(
    text: str,
    symspell,
    mode: str = "top",
    max_edit_distance: int = 2,
    min_freq: int = 5,
    lm: Optional[callable] = None,
) -> str:
    """
    Conservative Persian spell-fix using SymSpell — preserves spacing/separators.
    Will ONLY substitute when:
      - token is Persian, length>=3, not URL/email/latin/number
      - best suggestion edit distance <= max_edit_distance
      - suggestion frequency >= min_freq
      - (optional) LM score improves in a small local window
    """
    if not text or not isinstance(text, str):
        return text

    tokens = _tokenize_preserve_seps(text)
    out_chunks: List[str] = []

    for i, (chunk, is_word) in enumerate(tokens):
        if not is_word:
            out_chunks.append(chunk)
            continue

        word = chunk

        # Skip protected tokens
        if _is_protected_token(word):
            out_chunks.append(word)
            continue

        # Ask SymSpell
        try:
            if mode == "closest":
                suggs = symspell.lookup(word, Verbosity.CLOSEST,
                                        max_edit_distance=max_edit_distance, transfer_casing=True)
            else:
                suggs = symspell.lookup(word, Verbosity.TOP,
                                        max_edit_distance=max_edit_distance, transfer_casing=True)
        except Exception:
            out_chunks.append(word)
            continue

        if not suggs:
            out_chunks.append(word)
            continue

        best = suggs[0]
        # Filter by frequency & distance
        if best.distance > max_edit_distance or best.count < min_freq:
            out_chunks.append(word)
            continue

        candidate = best.term

        # If you have an LM, require local score improvement
        if lm is not None:
            # Build a tiny window "prev + candidate + next" just for safety
            prev_text = tokens[i - 1][0] if i - 1 >= 0 else ""
            next_text = tokens[i + 1][0] if i + 1 < len(tokens) else ""
            old_win = f"{prev_text}{word}{next_text}".strip()
            new_win = f"{prev_text}{candidate}{next_text}".strip()
            try:
                if lm(new_win) < lm(old_win):  # lower = worse (adjust for your LM)
                    out_chunks.append(word)
                    continue
            except Exception:
                # If LM fails for any reason, fall back to no-change
                out_chunks.append(word)
                continue

        out_chunks.append(candidate)

    # Reassemble and do minimal space/punct cleanup
    fixed = "".join(out_chunks)
    fixed = _normalize_spaces_punct(fixed)
    return fixed

# --- drop-in helpers ---
import re
from typing import Callable, Tuple

_NUM_WORDS = r"(?:صفر|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|یازده|دوازده|سیزده|چهارده|پانزده|شانزده|هفده|هجده|نوزده|بیست|سی|چهل|پنجاه|شصت|هفتاد|هشتاد|نود|صد|هزار|میلیون|میلیارد)"
_NUM_PAT = re.compile(rf"(?:\d+|{_NUM_WORDS})(?:\s*(?:و)?\s*(?:\d+|{_NUM_WORDS}))*")

def _mask_numbers(s: str) -> Tuple[str, Callable[[str], str]]:
    """Replace numeric phrases with placeholders and return an unmask() fn."""
    repls = {}
    def repl(m):
        k = f"<<NUM{len(repls)}>>"
        repls[k] = m.group(0)
        return k
    masked = _NUM_PAT.sub(repl, s)
    def unmask(t: str) -> str:
        for k, v in repls.items():
            t = t.replace(k, v)
        return t
    return masked, unmask

def _lm_score(s: str, lm) -> float:
    """Higher is better; if no LM, return 0 so we never force a bad change."""
    try:
        return lm.score(s) if lm else 0.0
    except Exception:
        return 0.0

def correct_fa_line_safe(text: str, sym, lm=None,
                         max_edit_distance: int = 1,
                         min_freq: int = 10,
                         min_gain: float = 1.5) -> str:
    """
    Wraps your existing correct_fa_line:
      - masks numeric phrases
      - uses tighter edit distance & freq
      - only accepts change if LM improves by min_gain
    """
    masked, unmask = _mask_numbers(text)
    # use your existing function with stricter knobs
    cand = correct_fa_line(
        masked, sym,
        mode="top",
        max_edit_distance=max_edit_distance,
        min_freq=min_freq,
        lm=lm,  # your KenLM scorer
    )
    cand = unmask(cand)

    # keep original unless LM clearly prefers the candidate
    if lm:
        orig_s = _lm_score(text, lm)
        cand_s = _lm_score(cand, lm)
        if (cand_s - orig_s) < min_gain:
            return text
    return cand

# add near your helpers
BAN_SUGGESTIONS = {"پنتاگون","دلتا","کامرون","باستانی","فیلمسازی","کافران","ارژنسی","کارسیون","خانوارده","دواستانی"}
ALLOW_COLLOQUIAL = {"ساختمون","کارمون","ویلاسازی","بازسازی","چلتا","پنجاه","پنجاتون","دروازه‌بانی","گردشگری"}

_SENT_SPLIT = re.compile(r"([.!؟!…]+)")

def _sentences(s: str):
    parts = _SENT_SPLIT.split(s)
    cur = ""
    for p in parts:
        cur += p
        if _SENT_SPLIT.fullmatch(p):
            yield cur.strip()
            cur = ""
    if cur.strip():
        yield cur.strip()

def _bad_candidate(orig: str, cand: str) -> bool:
    # reject if banned token appears or allowed colloquial got removed
    o_tokens = set(orig.split())
    c_tokens = set(cand.split())
    if BAN_SUGGESTIONS & c_tokens:
        return True
    if any(w in ALLOW_COLLOQUIAL and w not in c_tokens for w in o_tokens):
        return True
    return False

def correct_fa_line_safer_blockwise(text: str, sym, lm=None,
                                    max_edit_distance=1, min_freq=10,
                                    min_gain_per_100=0.8):
    out = []
    for sent in _sentences(text):
        masked, unmask = _mask_numbers(sent)
        cand = correct_fa_line(masked, sym, mode="top",
                               max_edit_distance=max_edit_distance,
                               min_freq=min_freq, lm=lm)
        cand = unmask(cand)

        # keep original unless (a) not obviously bad and (b) LM improves enough normalized by length
        if lm:
            ol = max(len(sent), 1)
            gain = (_lm_score(cand, lm) - _lm_score(sent, lm)) * (100.0 / ol)
            if gain < min_gain_per_100 or _bad_candidate(sent, cand):
                cand = sent
        out.append(cand)
    return " ".join(out)
