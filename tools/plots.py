# plots.py
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional, Iterable, Union
from collections import defaultdict, Counter
import re
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path




_WORD_RE = re.compile(r"\w+")

# -----------------------------
# Helpers to aggregate metrics
# -----------------------------
def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))

def speaker_word_counts(turns: List[Dict]) -> Dict[str, int]:
    """
    turns: [{'speaker': 'SPEAKER_00', 'text': '...'}, ...]
    returns: {'SPEAKER_00': 123, ...} word counts
    """
    counts = defaultdict(int)
    for t in turns:
        spk = str(t[0] if t[0] else "UNK")
        counts[spk] += _word_count(str(t[1] if t[1] else "UNK"))
    return dict(counts)

def speaker_durations(turns: List[Dict]) -> Dict[str, float]:
    """
    Sum durations per speaker from 'start'/'end' if present.
    returns seconds per speaker.
    """
    durs = defaultdict(float)
    for t in turns:
        spk = str(t[0] if t[0] else "UNK")
        try:
            start = float(t.get("start", 0.0))
            end   = float(t.get("end",   0.0))
            if end > start:
                durs[spk] += (end - start)
        except Exception:
            pass
    return dict(durs)

def remap_hyp_keys(hyp_stats: Dict[str, float], mapping_hyp_to_ref: Dict[str, str]) -> Dict[str, float]:
    """
    Sum hyp stats into ref speaker bins using mapping hyp->ref.
    """
    out = defaultdict(float)
    for hyp_spk, val in hyp_stats.items():
        ref_spk = mapping_hyp_to_ref.get(hyp_spk, f"UNMAPPED:{hyp_spk}")
        out[ref_spk] += val
    return dict(out)

def merge_ref_hyp_series(ref: Dict[str, float], hyp_mapped: Dict[str, float]) -> List[Tuple[str, float, float]]:
    """
    Returns list of (ref_speaker, ref_value, hyp_value_in_ref_bin)
    Speakers present in either side are included.
    """
    speakers = sorted(set(ref) | set(hyp_mapped))
    rows = []
    for spk in speakers:
        rows.append((spk, float(ref.get(spk, 0.0)), float(hyp_mapped.get(spk, 0.0))))
    return rows

# -----------------------------
# Plotting
# -----------------------------
def plot_per_speaker_count(
    ref_turns: List[Dict],
    hyp_turns: List[Dict],
    mapping_hyp_to_ref: Dict[str, str],
    title: str = "Per-speaker word counts (REF vs HYP→REF bins)",
):
    ref_counts = speaker_word_counts(ref_turns)
    hyp_counts = speaker_word_counts(hyp_turns)
    hyp_in_ref_bins = remap_hyp_keys(hyp_counts, mapping_hyp_to_ref)

    rows = merge_ref_hyp_series(ref_counts, hyp_in_ref_bins)
    labels = [r[0] for r in rows]
    ref_vals = [r[1] for r in rows]
    hyp_vals = [r[2] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, ref_vals, width, label="REF")
    ax.bar(x + width/2, hyp_vals, width, label="HYP (mapped)")
    ax.set_title(title)
    ax.set_ylabel("Word count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.show()




def _load_diar_segments_json(path: Union[str, Path]) -> List[dict]:
    """
    Load diarization segments from JSON file.
    Expects a list of dicts with at least: {'speaker', 'start', 'end', ...}.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def _durations_by_hyp_speaker(segments: Iterable[dict]) -> Dict[str, float]:
    """
    Sum durations per HYP speaker from diarization segments.
    Returns seconds per speaker.
    """
    durs: Dict[str, float] = {}
    for s in segments:
        spk = str(s.get("speaker", "UNK"))
        try:
            start = float(s.get("start", 0.0))
            end   = float(s.get("end",   0.0))
        except Exception:
            # bad typing → skip
            continue
        if end > start:
            durs[spk] = durs.get(spk, 0.0) + (end - start)
    return durs


def plot_hyp_durations_mapped_from_json(
    diar_json: Union[str, Path],
    mapping_hyp_to_ref: Dict[str, str],
    title: str = "Per-speaker total duration (HYP only)",
    save_path: Union[str, Path, None] = None,
    show: bool = True,
    min_seconds: float = 0.0,
) -> Tuple[List[str], List[float]]:
    """
    Plot *HYP diarization durations only*, labeling each bar as
    "HYP_SPK → REF_SPK" using the provided mapping.

    Parameters
    ----------
    diar_json : Path-like
        Path to diarization JSON (list of segments with 'speaker','start','end').
    mapping_hyp_to_ref : Dict[str, str]
        Mapping from HYP speaker ids (e.g., 'SPEAKER_03') to REF speaker labels.
        Unmapped HYP speakers will display as "(unmapped)".
    title : str
        Chart title.
    save_path : Path-like or None
        If given, save the figure to this path.
    show : bool
        If True, display the figure via plt.show().
    min_seconds : float
        Filter out speakers with total duration below this threshold.

    Returns
    -------
    (labels, values)
        The x-axis labels and the corresponding duration values (seconds).
    """
    segs = _load_diar_segments_json(diar_json)
    hyp_durs = _durations_by_hyp_speaker(segs)

    # Build display labels "HYP → REF" and filter by min_seconds
    rows = []
    for hyp_spk, secs in hyp_durs.items():
        if secs < min_seconds:
            continue
        ref = mapping_hyp_to_ref.get(hyp_spk, "(unmapped)")
        label = f"{hyp_spk} → {ref}"
        rows.append((label, secs))

    # Sort by duration desc for a nicer chart
    rows.sort(key=lambda x: x[1], reverse=True)

    if not rows:
        raise ValueError("No speakers with positive duration found (after filtering).")

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    # Plot
    x = range(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_ylabel("Seconds")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Annotate bars with seconds
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return labels, values


def plot_hyp_durations_by_ref_bin_from_json(
    diar_json: Union[str, Path],
    mapping_hyp_to_ref: Dict[str, str],
    title: str = "Total HYP duration aggregated into REF speakers",
    save_path: Union[str, Path, None] = None,
    show: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Optional companion chart:
    Aggregate all HYP speaker durations into REF bins via mapping and plot the totals.
    Unmapped HYP speakers are grouped under '(unmapped)'.
    """
    segs = _load_diar_segments_json(diar_json)
    hyp_durs = _durations_by_hyp_speaker(segs)

    ref_bins: Dict[str, float] = {}
    for hyp_spk, secs in hyp_durs.items():
        ref = mapping_hyp_to_ref.get(hyp_spk, "(unmapped)")
        ref_bins[ref] = ref_bins.get(ref, 0.0) + secs

    rows = sorted(ref_bins.items(), key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    x = range(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_ylabel("Seconds")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return labels, values


def plot_wer_by_speaker(wer_by_ref_speaker: Dict[str, float], title: str = "WER by reference speaker"):
    """
    wer_by_ref_speaker: {'SPEAKER_00': 0.18, ...} values in [0,1]
    """
    labels = list(sorted(wer_by_ref_speaker))
    vals = [wer_by_ref_speaker[k] for k in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, vals)
    ax.set_title(title)
    ax.set_ylabel("WER")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    plt.show()

def plot_confusion_matrix(
    sim_matrix: np.ndarray,
    hyp_labels: Sequence[str],
    ref_labels: Sequence[str],
    title: str = "Speaker similarity (HYP rows → REF cols)",
):
    """
    sim_matrix[r, c] ~ similarity between hyp_labels[r] and ref_labels[c]
    """
    fig, ax = plt.subplots()
    im = ax.imshow(sim_matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("REF speakers")
    ax.set_ylabel("HYP speakers")
    ax.set_xticks(np.arange(len(ref_labels)))
    ax.set_xticklabels(ref_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(hyp_labels)))
    ax.set_yticklabels(hyp_labels)
    # annotate cells with values
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
