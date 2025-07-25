from pathlib import Path
import re
import docx
import jiwer
import difflib
import logging
from typing import List, Dict, Union, Any, Optional

logger = logging.getLogger(__name__)

def get_hypothesis_text(segments: List[Dict[str, Any]]) -> str:
    """
    Concatenate ASR segments into a single string.
    """
    return " ".join(seg.get("text", "").strip() for seg in segments)


def load_reference_text(path: Union[str, Path]) -> str:
    """
    Read a .docx file and extract the reference transcript between known markers.
    Raises ValueError if markers not found.
    """
    path = Path(path)
    doc = docx.Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    # Define start/end marker patterns
    patterns = [
        r"English\s*Transcript\s*:\s*(.*?)\s*(?=Hebrew\s*Translation\s*:)",
        r"Original\s*Text\s*:\s*(.*?)\s*(?=Translated\s*Text\s*:)",
        r"Source\s*Text\s*:\s*(.*?)\s*(?=Translation\s*:)",
        r"(?:English Transcript|Original Text)\s*:\s*(.*?)\s*(?:Hebrew Translation|Translated Text)\s*:",
    ]
    for pat in patterns:
        match = re.search(pat, full_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            raw = match.group(1)
            return " ".join(raw.split())
    # No match found, raise with snippet
    snippet = full_text[:500].replace("\n", " ")
    raise ValueError(
        f"Could not find expected markers in '{path}'. Snippet: {snippet}"
    )


def compute_wer_metrics(hyp: str, ref: str) -> Dict[str, Any]:
    """
    Compute WER and related counts between reference and hypothesis.
    Returns a dict with 'wer', 'substitutions', 'deletions', 'insertions'.
    """
    wer_value = jiwer.wer(ref, hyp)
    measures = jiwer.process_characters(ref, hyp)
    return {
        "wer": wer_value,
        "substitutions": measures.substitutions,
        "deletions": measures.deletions,
        "insertions": measures.insertions,
    }


def diff_word_level(ref: str, hyp: str) -> List[str]:
    """
    Return a unified diff of reference vs hypothesis word lists.
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    return list(difflib.unified_diff(
        ref_words,
        hyp_words,
        fromfile="REFERENCE",
        tofile="HYPOTHESIS",
        lineterm=""
    ))


def compare_texts(
    hyp: str,
    ref: str,
    diff: bool = False
) -> Dict[str, Any]:
    """
    Compute and optionally print WER metrics and word-level diff.
    Returns the metrics dict.
    """
    metrics = compute_wer_metrics(hyp, ref)
    logging.info(f"📊 WER: {metrics['wer']:.2%} | S={metrics['substitutions']} D={metrics['deletions']} I={metrics['insertions']}")
    if diff:
        diffs = diff_word_level(ref, hyp)
        logging.info("🔍 Word-level diff:")
        for line in diffs:
            logging.info(line)
    return metrics


def segments_comparison(
    hyp_text: str,
    ground_truth_path: Union[str, Path],
    audio_path: Union[str, Path],
    diff: bool = False,
    print_hyp: bool = False,
    print_ref: bool = False,
    msg: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Full pipeline: extract hypothesis, load reference, normalize, and compare.

    Returns a dict with 'raw' and 'clean' metrics.
    """
    # Extract texts
    ref_text = load_reference_text(ground_truth_path)

    if print_hyp:
        logging.info(f"[Raw hyp] {hyp_text}")
    if print_ref:
        logging.info(f"[Ref] {ref_text}")

    # Raw comparison
    logging.info(f"--- Comparing raw {msg} ---")
    raw_metrics = compare_texts(hyp_text, ref_text, diff)

    clean_metrics = ""

    return {"raw": raw_metrics, "clean": clean_metrics}
