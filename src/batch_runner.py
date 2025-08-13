# src/batch_runner.py
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from asr import evaluate_transcription
from src.audio_utils import audio_info, resample_to_16k

# --- you already have these somewhere ---
# from src.audio_utils import audio_info, resample_to_16k
# from src.asr import evaluate_transcription

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
REF_EXTS = {".txt", ".docx"}

logger = logging.getLogger(__name__)


import difflib
import logging
import re

logger = logging.getLogger(__name__)


def _norm_stem(p: Path) -> str:
    """
    Normalize filename stem for matching:
    - lowercase
    - remove non-alphanumerics except spaces, underscores, dashes
    - collapse spaces/underscores/dashes to single dash
    """
    s = p.stem.lower()
    # keep letters/numbers/space/_/-
    s = re.sub(r"[^0-9a-zA-Z _-]+", "", s)
    # collapse space/_/- to single dash
    s = re.sub(r"[ _-]+", "-", s).strip("-")
    return s


def pair_audio_and_refs(
    audios: List[Path], refs: List[Path], fuzzy_threshold: float = 0.75
) -> List[Tuple[Path, Path]]:
    """
    Match each audio with a reference file:
      1) Prefer refs whose normalized stem *starts with* the audio's normalized stem.
         If multiple, choose the shortest such ref name.
      2) Otherwise, choose the fuzzy-closest ref (difflib) if similarity >= fuzzy_threshold.
      3) If still nothing, warn and skip.

    Returns list of (audio_path, ref_path) pairs.
    """
    # Precompute normalized stems
    ref_norm_index: Dict[str, List[Path]] = {}
    for r in refs:
        ref_norm_index.setdefault(_norm_stem(r), []).append(r)

    # For convenient startswith search, keep a list of (norm, Path)
    ref_norm_list = [(n, r) for n, paths in ref_norm_index.items() for r in paths]

    pairs: List[Tuple[Path, Path]] = []
    for a in sorted(audios):
        a_norm = _norm_stem(a)

        # 1) startswith matches
        starts = [(n, r) for (n, r) in ref_norm_list if n.startswith(a_norm)]
        if starts:
            # pick the shortest normalized ref name → usually the most "base" file
            chosen = min(starts, key=lambda t: len(t[0]))[1]
            pairs.append((a, chosen))
            continue

        # 2) fuzzy fallback over normalized names
        candidates = [n for (n, _r) in ref_norm_list]
        best = difflib.get_close_matches(
            a_norm, candidates, n=1, cutoff=fuzzy_threshold
        )
        if best:
            best_norm = best[0]
            # If multiple refs share the same normalized stem, choose the shortest actual filename
            ref_candidates = [r for (n, r) in ref_norm_list if n == best_norm]
            chosen = min(ref_candidates, key=lambda p: len(p.stem))
            logger.warning(
                f"ℹ️  Using fuzzy match for audio '{a.name}' → ref '{chosen.name}' (key='{best_norm}')"
            )
            pairs.append((a, chosen))
        else:
            logger.warning(f"⚠️  No ref found for audio: {a.name}")

    return pairs


def _find_pairs(
    folder: Path,
    audio_exts: Iterable[str] = AUDIO_EXTS,
    ref_exts: Iterable[str] = REF_EXTS,
    prefix: Optional[str] = None,
) -> List[Tuple[Path, Path]]:
    """
    Return (audio_path, ref_path) pairs matched by the same stem.
    If `prefix` is provided, only consider files whose stem starts with it.
    """
    folder = Path(folder)
    audios = [
        p for p in folder.iterdir() if p.suffix.lower() in audio_exts and p.is_file()
    ]
    refs = [p for p in folder.iterdir() if p.suffix.lower() in ref_exts and p.is_file()]

    if prefix:
        audios = [p for p in audios if p.stem.startswith(prefix)]
        refs = [p for p in refs if p.stem.startswith(prefix)]

    # map by stem
    return pair_audio_and_refs(audios, refs, fuzzy_threshold=0.75)

    # ref_by_stem: Dict[str, Path] = {p.stem: p for p in refs}
    # pairs: List[Tuple[Path, Path]] = []
    # for a in sorted(audios):
    #     r = ref_by_stem.get(a.stem)
    #     if r:
    #         pairs.append((a, r))
    #     else:
    #         logger.warning(f"⚠️  No ref found for audio: {a.name}")
    # return pairs


def _ensure_16k(audio_path: Path) -> Path:
    """
    If audio is not 16k mono, resample to <stem>_16k.wav in the same folder.
    If the original is already 16k, return it as-is.
    """
    try:
        ch, sr, dur = audio_info(
            str(audio_path)
        )  # your utility: returns (channels, sample_rate, duration_sec)
    except Exception as e:
        logger.exception(f"Failed to read audio info for {audio_path}: {e}")
        raise

    if sr == 16000 and ch in (1, "1"):
        logger.info(f"🎧 Audio already 16 kHz mono → {audio_path.name}")
        return audio_path

    # avoid resampling again if a 16k sibling already exists
    target = audio_path.with_name(f"{audio_path.stem}_16k.wav")
    if target.exists():
        logger.info(f"↪️  Using existing resampled file: {target.name}")
        return target

    logger.info(f"🔄 Resampling to 16 kHz mono → {target.name} (from {sr} Hz, {ch} ch)")
    out = resample_to_16k(str(audio_path))  # your utility returns the new path as str
    return Path(out)


def run_batch_from_folder(
    folder: str,
    limit: Optional[int] = None,
    prefix: Optional[str] = None,
    *,
    # evaluate_transcription passthrough (set your own sensible defaults)
    model_size: str,
    device: str,
    compute_type: str,
    batch_size: int,
    diff: bool = False,
    print_hyp: bool = False,
    print_ref: bool = False,
    llm_clean: bool = False,
    asr_options: Optional[Dict] = None,
    vad_method: Optional[str] = None,
    vad_options: Optional[Dict] = None,
    print_progress: bool = True,
    # comparison/loader hints you added earlier
    fallback: Optional[str] = None,
    strip_speakers: Optional[bool] = None,
    script_hint: Optional[str] = None,
    language: Optional[str] = None,
) -> List[Dict]:
    """
    Scan `folder` for (audio, ref) pairs and run ASR on up to `limit` pairs.
    - Matches by identical stem (e.g., foo.mp3 ↔ foo.txt / foo.docx).
    - Resamples audio to 16 kHz mono if needed, writing <stem>_16k.wav.
    - Calls your evaluate_transcription for each pair.

    Returns a list of result dicts from evaluate_transcription (and minimal metadata).
    """
    folder_p = Path(folder).expanduser().resolve()
    assert folder_p.is_dir(), f"Not a folder: {folder_p}"
    logger.info(f"looking into folder: {folder_p}")

    pairs = _find_pairs(folder_p, prefix=prefix)
    if not pairs:
        logger.warning(f"⚠️  No (audio, ref) pairs found in {folder_p}")
        return []

    logger.info(f"limit is {limit}")
    logger.info(f"pairs length: {len(pairs)}")
    logger.info(f"pairs: {pairs}")

    if limit is not None:
        pairs = pairs[: max(0, int(limit))]

    logger.info(f"🗂  Running batch on {len(pairs)} file(s) in: {folder_p}")
    results: List[Dict] = []

    for idx, (audio_path, ref_path) in enumerate(pairs, start=1):
        logger.info(f"[{idx}/{len(pairs)}] 🎬 {audio_path.name}  ↔  {ref_path.name}")

        try:
            # 1) Ensure 16 kHz mono
            audio_16k = _ensure_16k(audio_path)

            # 2) Run your experiment (evaluate_transcription)
            res = evaluate_transcription(
                audio_path=str(audio_16k),
                ground_truth_path=str(ref_path),
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                diff=diff,
                print_hyp=print_hyp,
                print_ref=print_ref,
                llm_clean=llm_clean,
                asr_options=asr_options,
                vad_method=vad_method,
                vad_options=vad_options,
                print_progress=print_progress,
                fallback=fallback,
                strip_speakers=strip_speakers,
                script_hint=script_hint,
                language=language,
            )
            results.append(
                {
                    "audio": str(audio_path),
                    "audio_used": str(audio_16k),
                    "ref": str(ref_path),
                    "asr_result": res,
                }
            )
        except Exception as e:
            logger.exception(f"✗ Failed on {audio_path.name}: {e}")

    logger.info("✅ Batch completed.")
    return results
