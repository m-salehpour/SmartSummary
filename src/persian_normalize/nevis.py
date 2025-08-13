# ──────────────────────────────────────────────────────────────────────
# 1 « Nevise » spell-checker glue (chatty)
# ──────────────────────────────────────────────────────────────────────

import re
from pathlib import Path

import torch

import src.persian_normalize.Nevise.utils as utils
from hazm import Normalizer
from src.persian_normalize.Nevise.helpers import (
    batch_iter, bert_tokenize_for_valid_examples, get_model_nparams, labelize,
    load_vocab_dict, untokenize_without_unks)
from src.persian_normalize.Nevise.models import SubwordBert
from src.persian_normalize.Nevise.utils import get_sentences_splitters


def _load_nevise(vocab_path: Path, ckpt_path: Path, device: str):
    vocab = load_vocab_dict(str(vocab_path))
    model = SubwordBert(
        3 * len(vocab["chartoken2idx"]),
        vocab["token2idx"][vocab["pad_token"]],
        len(vocab["token_freq"]),
    )
    print(
        f"[Nevise] Loaded SubwordBert model with {get_model_nparams(model):,} parameters"
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"[Nevise] Checkpoint '{ckpt_path.name}' loaded onto {device}")
    return model, vocab


@torch.inference_mode()
def _nevise_correct(
    model: SubwordBert, vocab: dict, device: str, hazm_norm: Normalizer, text: str
) -> str:
    print(f"\n[Nevise] Original sentence: {text!r}")
    sub_sents, _ = get_sentences_splitters(text)
    sub_sents = [
        hazm_norm.normalize(utils.space_special_chars(s))
        for s in sub_sents
        if s.strip()
    ]
    if not sub_sents:
        print("[Nevise] No sub-sentences found, returning original")
        return text.strip()

    fixed = []
    for batch_labels, batch_sentences in batch_iter(
        [(s, s) for s in sub_sents], batch_size=8, shuffle=False
    ):
        print(f"[Nevise] Processing batch: {batch_sentences}")
        b_lbl, b_sent, bert_inp, bert_split = bert_tokenize_for_valid_examples(
            batch_labels, batch_sentences
        )
        if not b_lbl:
            print("[Nevise]  → Skipped (tokenization mismatch)")
            fixed.extend(batch_sentences)
            continue
        bert_inp = {k: v.to(device) for k, v in bert_inp.items()}
        lbl_ids, lens = labelize(b_lbl, vocab)
        lbl_ids, lens = lbl_ids.to(device), lens.to(device)
        loss, preds = model(bert_inp, bert_split, targets=lbl_ids, topk=1)
        preds_txt = untokenize_without_unks(preds, lens.cpu().numpy(), vocab, b_sent)
        for orig, corr in zip(b_sent, preds_txt):
            print(f"[Nevise]   → '{orig}'  →  '{corr}'")
        fixed.extend(preds_txt)

    joined = utils.de_space_special_chars(" ".join(fixed))
    cleaned = re.sub(r"\s+", " ", joined).strip() or text.strip()
    print(f"[Nevise] Final cleaned: {cleaned!r}")
    return cleaned


class NeviseCorrector:
    """Facade for Nevise spell-checking."""

    def __init__(self, vocab_path: Path, ckpt_path: Path, device: str = "auto"):
        self.device = (
            "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        print(f"[NeviseCorrector] Using device: {self.device}")
        self.hazm_norm = Normalizer()
        self.model, self.vocab = _load_nevise(vocab_path, ckpt_path, self.device)

    def clean(self, sentence: str) -> str:
        return _nevise_correct(
            self.model, self.vocab, self.device, self.hazm_norm, sentence
        )
