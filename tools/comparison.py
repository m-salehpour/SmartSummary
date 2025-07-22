#!/usr/bin/env python3
import docx       # pip install python-docx
import re
import jiwer      # pip install jiwer
import difflib


def get_hypothesis_text(segments):
    """
    segments: list of dicts with 'text','start','end'
    returns one long string of all text in order
    """
    # simply concatenate segment texts with spaces
    return " ".join(seg["text"].strip() for seg in segments)

def load_reference_text(docx_path: str) -> str:
    """
    Reads a .docx, extracts the chunk of text between one of several marker pairs.
    If no marker is found, raises ValueError and dumps the first 500 characters
    so you can inspect what's actually in the file.
    """
    # Load full text
    doc = docx.Document(docx_path)
    full_text = "\n".join(p.text for p in doc.paragraphs)

    # List of (start_marker, end_marker) regex pairs to try
    patterns = [
        (r"English\s*Transcript\s*:\s*(.*?)\s*(?=Hebrew\s*Translation\s*:)", None),
        (r"Original\s*Text\s*:\s*(.*?)\s*(?=Translated\s*Text\s*:)", None),
        (r"Source\s*Text\s*:\s*(.*?)\s*(?=Translation\s*:)", None),
        (r"English\s*Transcript\s*:\s*(.*)", r"(?=Hebrew\s*Translation\s*:)"),  # alternative lookahead
        (r"(?:English Transcript|Original Text)\s*:\s*(.*?)\s*(?:Hebrew Translation|Translated Text)\s*:", None),
    ]

    for start_pat, end_pat in patterns:
        if end_pat is None:
            # single regex with lookahead in the first group
            combined = start_pat
        else:
            combined = start_pat + end_pat

        m = re.search(combined, full_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            # collapse all whitespace to single spaces
            raw = m.group(1)
            return " ".join(raw.split())

    # nothing matched — raise with a snippet for debugging
    snippet = full_text[:500].replace("\n", " ")
    raise ValueError(
        f"Could not find any of the expected markers in '{docx_path}'.\n"
        f"Here are the first 500 chars of the document:\n\n{snippet}\n\n"
        "Please update your markers or add a new regex to handle this format."
    )


def compare_texts(hyp: str, ref: str, diff: bool):
    """
    Compute WER + insertions/deletions/substitutions,
    and print a word-level diff.
    """
    wer = jiwer.wer(ref, hyp)
    measures = jiwer.process_characters(ref, hyp)
    subs = measures.substitutions
    dels = measures.deletions
    ins = measures.insertions


    # word-level diff
    if diff:
        ref_words = ref.split()
        hyp_words = hyp.split()
        diff = difflib.unified_diff(
            ref_words,
            hyp_words,
            fromfile="REFERENCE",
            tofile="HYPOTHESIS",
            lineterm="",
        )
        print("🔍 Word-level diff:")
        for line in diff:
            print(line)

    print(f"\n📊 WER: {wer:.2%}")
    print(f"   Substitutions: {subs}")
    print(f"   Deletions:     {dels}")
    print(f"   Insertions:    {ins}\n")

