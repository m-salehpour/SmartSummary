from nevis import NeviseCorrector
from pathlib import Path
from tools.persian_normalizer import persian_normalizer
from hazm import Normalizer as HazmNormalizer

def cleaning(text, language=None):
    if not isinstance(text, str):
        return None

    if language == "fa":

        nevise_ckpt     = Path("Nevise/model/model.pth.tar")
        vocab_path      = Path("Nevise/model/vocab.pkl")
        print("🔎  Spell-checking Persian with Nevise…")
        corrector = NeviseCorrector(vocab_path, nevise_ckpt)
        cleaned = corrector.clean(text)
        print(f"\n[normalizer] Nevis clean: {text}\n")

        p_normalizer = persian_normalizer({"sentence": cleaned}, return_dict=False)
        print("\np_normalizer\n", p_normalizer)
        h_normalizer = HazmNormalizer()
        hazm_out = h_normalizer.normalize(p_normalizer)
        print("\nhazm_out\n", hazm_out)

        return hazm_out

