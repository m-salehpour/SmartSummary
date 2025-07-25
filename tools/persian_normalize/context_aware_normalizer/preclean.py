# preclean.py
# deterministic cheap fixes

import regex as re
from hazm import Normalizer

_norm = Normalizer(correct_spacing=False)

RE_YK   = re.compile(r"[ي]")        # Arabic Yeh
RE_KAFE = re.compile(r"[ك]")        # Arabic Kaf
RE_MISPACE = re.compile(r"\s{2,}")
RE_MI  = re.compile(r"\bمی\s+(\S+)")

def preclean(txt: str) -> str:
    txt = _norm.normalize(txt)
    txt = RE_YK.sub("ی", txt)
    txt = RE_KAFE.sub("ک", txt)
    txt = RE_MI.sub(r"می‌\1", txt)
    txt = RE_MISPACE.sub(" ", txt)
    return txt.strip()
