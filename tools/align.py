# align.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import whisperx

# Optional: import your config if you keep paths there
try:
    import config  # noqa: F401
except Exception:
    config = None  # type: ignore

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Alignment: load model + run alignment on WhisperX segments
# ---------------------------------------------------------------------------

def load_alignment_model(
    language_code: str,
    device: str = "cpu",
    *,
    model_name: Optional[str] = None,
    model_dir: Optional[PathLike] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load the WhisperX alignment model (wav2vec2 / torchaudio pipeline) for a given language.

    Parameters
    ----------
    language_code : str
        ISO code detected/used by ASR (e.g., "en", "fa", "he").
    device : str
        "cpu" or "cuda" (e.g., "cuda:0").
    model_name : Optional[str]
        Override default alignment model. If None, WhisperX will choose a default per language.
    model_dir : Optional[PathLike]
        Where to cache/load the alignment model (works offline).

    Returns
    -------
    (model, metadata)
        model: the alignment model (HF/torchaudio)
        metadata: dict with keys like {"language", "dictionary", "type"}
    """
    md = str(model_dir) if model_dir else None
    logger.info(f"[alignment] loading align model for lang={language_code!r} (cache={md})")
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code, device=device, model_name=model_name, model_dir=md
    )
    return model_a, metadata


def align_segments(
    segments: Iterable[Dict[str, Any]],
    audio: Union[PathLike, "np.ndarray", "torch.Tensor"],
    *,
    language_code: str,
    device: str = "cpu",
    model_name: Optional[str] = None,
    model_dir: Optional[PathLike] = None,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> Dict[str, Any]:
    """
    Align WhisperX segments to word (and optionally char) timestamps.

    Parameters
    ----------
    segments : iterable of dict
        Items with at least {"text","start","end"}.
    audio : str | Path | np.ndarray | torch.Tensor
        Audio path or already-loaded waveform (mono 16k expected by whisperx.load_audio).
    language_code : str
        Language code for alignment model.
    device : str
        "cpu" or "cuda".
    model_name : Optional[str]
        Override the default language alignment model.
    model_dir : Optional[PathLike]
        Cache directory for alignment assets (offline-friendly).
    interpolate_method : str
        How to fill missing sub-segment times; "nearest" (default) or "linear".
    return_char_alignments : bool
        If True, include per-character timing in the result.
    print_progress / combined_progress : bool
        Progress printing flags passed through to align().

    Returns
    -------
    result : dict
        {"segments": [...], "word_segments": [...]}
    """
    # Load alignment model
    model_a, metadata = load_alignment_model(
        language_code=language_code, device=device, model_name=model_name, model_dir=model_dir
    )

    # Load audio if path given
    if isinstance(audio, (str, Path)):
        audio_wave = whisperx.load_audio(str(audio))
    else:
        audio_wave = audio

    # Run alignment
    logger.info("[alignment] running forced alignment …")
    result = whisperx.align(
        transcript=list(segments),
        model=model_a,
        align_model_metadata=metadata,
        audio=audio_wave,
        device=device,
        interpolate_method=interpolate_method,
        return_char_alignments=return_char_alignments,
        print_progress=print_progress,
        combined_progress=combined_progress,
    )
    logger.info("[alignment] done.")
    return result


def align_from_asr_result(
    asr_result: Dict[str, Any],
    audio_path: PathLike,
    *,
    device: str = "cpu",
    model_name: Optional[str] = None,
    model_dir: Optional[PathLike] = None,
    **align_kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper: take the dict returned by ASR (with 'segments' and 'language'),
    and return an aligned result.

    Parameters
    ----------
    asr_result : dict
        Must include ["segments"] and ["language"].
    audio_path : str | Path
        Path to the original audio file.
    device, model_name, model_dir
        See `align_segments`.
    align_kwargs
        Extra kwargs forwarded to `align_segments` (e.g., interpolate_method, return_char_alignments).

    Returns
    -------
    dict
        Alignment result as returned by `whisperx.align`.
    """
    segments = asr_result.get("segments")
    language = asr_result.get("language")
    if not segments or not language:
        raise ValueError("asr_result must include 'segments' and 'language'")

    return align_segments(
        segments=segments,
        audio=audio_path,
        language_code=language,
        device=device,
        model_name=model_name,
        model_dir=model_dir,
        **align_kwargs,
    )


# ---------------------------------------------------------------------------
# Diarization (optional): get speaker turns and attach to aligned words
# ---------------------------------------------------------------------------

def diarize_audio(
    audio: PathLike,
    *,
    device: str = config.DEVICE,
    use_auth_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run diarization to get speaker segments.

    Returns
    -------
    diarize_segments : list of dict
        Each element typically has {"start","end","speaker"} after WhisperX post-processing.
    """
    logger.info("[diarize] loading diarization pipeline …")
    pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=use_auth_token, device=device)
    # If you know #speakers, pass min/max for better VAD clustering
    if min_speakers is not None or max_speakers is not None:
        diar = pipeline(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    else:
        diar = pipeline(audio)
    logger.info("[diarize] diarization complete.")
    return diar


def attach_speakers(
    diarize_segments: List[Dict[str, Any]],
    aligned_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach speaker labels to aligned words/segments.

    Returns the same aligned_result dict with speaker information added.
    """
    logger.info("[diarize] assigning speakers to words …")
    out = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    logger.info("[diarize] speaker assignment done.")
    return out


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_alignment_result(
    aligned: Dict[str, Any],
    audio_path: PathLike,
    *,
    output_dir: PathLike = config.JSON_ASR_OUTPUT_DIR,
    suffix: str = "_aligned",
) -> Path:
    """
    Save alignment JSON next to ASR output convention.

    The filename is derived from the audio name: <stem>_aligned.json by default.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    out = out_dir / f"{stem}{suffix}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Aligned JSON saved to {out}")
    return out


# def save_diarization_json(
#     diarize: Any,
#     audio_file: str,
#     out_path: str | None = None,
# ) -> Path:
#     """
#     Normalize diarization output to a list of {start, end, speaker} dicts
#     and save as JSON next to the audio (or to out_path if given).
#
#     Supports:
#       - pandas.DataFrame with columns ['start','end','speaker'] (extra cols ignored)
#       - pyannote.core.Annotation
#       - iterable of dicts/tuples containing start/end/speaker
#     """
#     records: List[Dict[str, Any]] = []
#
#     # Case 1: pandas DataFrame
#     try:
#         import pandas as pd  # type: ignore
#         if isinstance(diarize, pd.DataFrame):
#             required = {"start", "end", "speaker"}
#             missing = required - set(diarize.columns)
#             if missing:
#                 raise ValueError(f"DataFrame missing columns: {missing}")
#             # Select and cast
#             df = diarize[["start", "end", "speaker"]].copy()
#             df["start"] = df["start"].astype(float)
#             df["end"] = df["end"].astype(float)
#             df["speaker"] = df["speaker"].astype(str)
#             records = df.to_dict(orient="records")
#     except Exception:
#         # Not a DataFrame or something went wrong; fall through to other cases
#         pass
#
#     # Case 2: pyannote Annotation
#     if not records and hasattr(diarize, "itertracks"):
#         # pyannote.core.Annotation
#         # itertracks(yield_label=True) -> (segment, track, label)
#         for segment, _, label in diarize.itertracks(yield_label=True):
#             records.append({
#                 "start": float(segment.start),
#                 "end": float(segment.end),
#                 "speaker": str(label),
#             })
#
#     # Case 3: list/iterable of dicts or tuples
#     if not records:
#         try:
#             # Try to coerce from iterable
#             for row in diarize:
#                 if isinstance(row, dict):
#                     start = float(row["start"])
#                     end = float(row["end"])
#                     spk = str(row.get("speaker", row.get("label", "SPEAKER_00")))
#                 else:
#                     # Assume tuple-like: (start, end, speaker)
#                     start = float(row[0])
#                     end = float(row[1])
#                     spk = str(row[2])
#                 records.append({"start": start, "end": end, "speaker": spk})
#         except Exception as e:
#             raise TypeError(
#                 "Unsupported diarization object. Expected a pandas DataFrame, a pyannote Annotation, "
#                 "or an iterable of dicts/tuples containing start/end/speaker."
#             ) from e
#
#     # Write JSON
#     audio_path = Path(audio_file)
#     out = Path(out_path) if out_path else audio_path.with_suffix(".diarize.json")
#     out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
#     print(f"✅ Diarization written to {out}")
#     return out

# --- helpers ---------------------------------------------------------------
def _coerce_record_like(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a single diarization row JSON-serializable.
    Handles possible 'segment' objects (pyannote.core.Segment).
    """
    out = dict(obj)

    # If a pyannote Segment is present, extract start/end and drop the object
    seg = out.get("segment", None)
    if seg is not None:
        try:
            out.setdefault("start", float(getattr(seg, "start")))
            out.setdefault("end", float(getattr(seg, "end")))
        except Exception:
            # best effort; if segment can't provide start/end, stringify it
            out["segment"] = str(seg)
        else:
            out.pop("segment", None)

    # Normalize start/end as floats if present
    if "start" in out:
        try:
            out["start"] = float(out["start"])
        except Exception:
            pass
    if "end" in out:
        try:
            out["end"] = float(out["end"])
        except Exception:
            pass

    # Speaker/label normalization
    # Prefer 'speaker', fall back to 'label'
    if "speaker" in out:
        out["speaker"] = str(out["speaker"])
    elif "label" in out:
        out["speaker"] = str(out["label"])
    # Do not keep both keys in the final JSON
    out.pop("label", None)

    return out


def _df_to_records(df) -> List[Dict[str, Any]]:
    """
    Convert a pandas.DataFrame from diarization into JSON-serializable records.
    Works whether the DF has explicit start/end/speaker or a 'segment' column.
    """
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        records.append(_coerce_record_like(rec))
    return records


def _annotation_to_records(annotation) -> List[Dict[str, Any]]:
    """
    Convert a pyannote.core.Annotation into JSON-serializable records.
    """
    records: List[Dict[str, Any]] = []
    try:
        for segment, _, label in annotation.itertracks(yield_label=True):
            records.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(label),
            })
    except Exception as e:
        logger.exception("Failed to convert Annotation to records")
        raise
    return records


def save_diarization_json(
    diarization_result: Any,
    audio_path: Union[str, Path],
    out_dir: Union[str, Path],
    diarization_json_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Save diarization output to JSON.
    - Accepts pandas.DataFrame, pyannote Annotation, or list[dict].
    - Returns the file path on success, or None on failure.
    """
    try:
        audio_path = Path(audio_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resolve target path
        if diarization_json_path is None:
            diar_json_path = out_dir / f"{audio_path.stem}.diarize.json"
        else:
            p = Path(diarization_json_path)
            diar_json_path = (p / f"{audio_path.stem}.diarize.json") if p.is_dir() else p
        diar_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Build records in a JSON-serializable shape
        records: List[Dict[str, Any]]

        # Case 1: pandas.DataFrame
        try:
            import pandas as pd  # type: ignore
            if isinstance(diarization_result, pd.DataFrame):
                records = _df_to_records(diarization_result)
            else:
                raise TypeError  # fall through to other cases
        except Exception:
            # Case 2: pyannote Annotation (has itertracks)
            if hasattr(diarization_result, "itertracks"):
                records = _annotation_to_records(diarization_result)
            # Case 3: list/iterable of dicts or mixed
            elif isinstance(diarization_result, (list, tuple)):
                records = [_coerce_record_like(r if isinstance(r, dict) else {"value": r})
                           for r in diarization_result]
            # Case 4: single dict
            elif isinstance(diarization_result, dict):
                records = [_coerce_record_like(diarization_result)]
            else:
                # Final fallback: stringify
                records = [{"value": str(diarization_result)}]

        # Dump JSON
        diar_json_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if verbose:
            logger.info(f"📄 Diarization JSON: {diar_json_path}")
        return diar_json_path

    except Exception:
        logger.exception("⚠️  Failed to save diarization JSON (continuing).")
        return None

def align_and_diarize(
    *,
    # --- required inputs ---
    asr_result: Dict[str, Any],
    audio_path: Union[str, Path],

    # --- alignment options ---
    device: str = config.DEVICE,
    model_dir: Optional[Union[str, Path]] = None,
    return_char_alignments: bool = False,

    # --- diarization options ---
    do_diarize: bool = True,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,

    # --- output / runtime options ---
    out_dir: Union[str, Path] = ".",
    save_diarization_as_json: bool = False,
    diarization_json_path: Optional[Union[str, Path]] = None,
    parallel: bool = True,               # try to overlap diarization with alignment
    verbose: bool = True,                # chatty logs
) -> Dict[str, Any]:
    """
    Align word timings from an existing ASR result and (optionally) run diarization,
    then attach speaker labels and write the aligned JSON.

    Returns a dict:
        {
          "aligned": <aligned_result_dict>,
          "diarization": <diarization_result or None>,
          "aligned_json": <path to saved alignment json>,
          "diarization_json": <path or None>,
        }

    Notes on parallelism:
      - If `parallel=True` and `do_diarize=True`, diarization is started in a background
        thread while alignment runs. This can reduce overall wall time, but both steps
        will compete for CPU/GPU memory. Set `parallel=False` if you run out of RAM/VRAM.
    """
    from concurrent.futures import ThreadPoolExecutor

    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info("🎯 Starting align_and_diarize")
        logger.info(f"• audio_path={audio_path}")
        logger.info(f"• device={device}  • return_char_alignments={return_char_alignments}")
        logger.info(f"• do_diarize={do_diarize}  • parallel={parallel}")
        if do_diarize:
            logger.info(f"• diarization: min_speakers={min_speakers}  max_speakers={max_speakers}")

    # Validate ASR result
    if not isinstance(asr_result, dict) or "segments" not in asr_result:
        raise ValueError("`asr_result` must be a dict containing a 'segments' list.")

    # Prepare outputs
    aligned_result: Optional[Dict[str, Any]] = None
    diarization_result = None
    aligned_json_path: Optional[Path] = None
    diar_json_path: Optional[Path] = None

    # Kick off diarization early if parallel & requested
    diar_future = None
    if do_diarize and parallel:
        if verbose:
            logger.info("🚀 Launching diarization in background …")
        try:
            executor = ThreadPoolExecutor(max_workers=2)
            diar_future = executor.submit(
                diarize_audio,
                audio=str(audio_path),
                device=device,
                use_auth_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception:
            logger.exception("Failed to start background diarization; will fallback to sequential.")
            diar_future = None

    # Alignment (foreground)
    try:
        if verbose:
            logger.info("⏱  Aligning words from ASR segments …")
        aligned_result = align_from_asr_result(
            asr_result=asr_result,
            audio_path=str(audio_path),
            device=device,
            model_dir=(str(model_dir) if model_dir else None),
            return_char_alignments=return_char_alignments,
        )
        if verbose:
            logger.info("✅ Alignment complete.")
    except Exception:
        logger.exception("❌ Alignment failed.")
        raise

    # Diarization (if not already running in background)
    if do_diarize and diar_future is None:
        try:
            if verbose:
                logger.info("⏱  Running diarization …")
            diarization_result = diarize_audio(
                audio=str(audio_path),
                device=device,
                use_auth_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            if verbose:
                logger.info("✅ Diarization complete.")
        except Exception:
            logger.exception("❌ Diarization failed.")
            # continue without diarization
            diarization_result = None

    # If diarization was running in background, collect it now
    if do_diarize and diar_future is not None:
        try:
            diarization_result = diar_future.result()
            if verbose:
                logger.info("✅ Diarization (background) complete.")
        except Exception:
            logger.exception("❌ Diarization (background) failed.")
            diarization_result = None
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    # Attach speakers (if we have diarization)
    if do_diarize and diarization_result is not None:
        try:
            if verbose:
                logger.info("🔗 Attaching speaker labels to aligned words …")
            aligned_result = attach_speakers(diarization_result, aligned_result)
            if verbose:
                logger.info("✅ Speaker labels attached.")
        except Exception:
            logger.exception("⚠️  attach_speakers failed; continuing with alignment only.")

    # Save alignment JSON
    try:
        if verbose:
            logger.info("💾 Saving aligned result …")
        aligned_json_path = save_alignment_result(
            aligned_result,
            audio_path=str(audio_path),
            output_dir=str(out_dir),
        )
        if verbose:
            logger.info(f"📄 Aligned JSON: {aligned_json_path}")
    except Exception:
        logger.exception("❌ Failed to save aligned result.")
        raise

    # Optionally save diarization JSON
    if save_diarization_as_json and diarization_result is not None:
        try:
            save_diarization_json(
                diarization_result=diarization_result,  # DataFrame / Annotation / list[dict]
                audio_path=audio_path,  # Path-like
                out_dir=out_dir,  # directory
                diarization_json_path=None,  # or omit it
                verbose=True,
            )

            if verbose:
                logger.info(f"📄 Diarization JSON: {diar_json_path}")
        except Exception:
            logger.exception("⚠️  Failed to save diarization JSON (continuing).")


    # # Optionally save diarization JSON
    # if save_diarization_as_json and diarization_result is not None:
    #     try:
    #         diar_json_path = Path(diarization_json_path) if diarization_json_path else (
    #             out_dir / f"{audio_path.stem}.diarize.json"
    #         )
    #         # Try to convert to JSON-serializable
    #         to_write: Any = diarization_result
    #         try:
    #             # If it's a pandas.DataFrame (common for pyannote), serialize rows
    #             import pandas as pd  # type: ignore
    #
    #             if isinstance(diarization_result, pd.DataFrame):
    #                 to_write = diarization_result.to_dict(orient="records")
    #         except Exception:
    #             pass
    #
    #         # Normalize types for JSON (floats & strings)
    #         def _coerce_row(row: Dict[str, Any]) -> Dict[str, Any]:
    #             out = dict(row)
    #             if "start" in out:
    #                 out["start"] = float(out["start"])
    #             if "end" in out:
    #                 out["end"] = float(out["end"])
    #             if "speaker" in out and not isinstance(out["speaker"], str):
    #                 out["speaker"] = str(out["speaker"])
    #             return out
    #
    #         if isinstance(to_write, list) and to_write and isinstance(to_write[0], dict):
    #             to_write = [_coerce_row(r) for r in to_write]
    #
    #         diar_json_path.write_text(
    #             json.dumps(to_write, ensure_ascii=False, indent=2),
    #             encoding="utf-8",
    #         )
    #         if verbose:
    #             logger.info(f"📄 Diarization JSON: {diar_json_path}")
    #     except Exception:
    #         logger.exception("⚠️  Failed to save diarization JSON (continuing).")

    return {
        "aligned": aligned_result,
        "diarization": diarization_result,
        "aligned_json": str(aligned_json_path) if aligned_json_path else None,
        "diarization_json": str(diar_json_path) if diar_json_path else None,
    }


# ---------------------------------------------------------------------------
# Example CLI usage (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align WhisperX segments (and optionally diarize).")
    parser.add_argument("--audio", type=Path, required=True, help="Path to audio file")
    parser.add_argument("--asr-json", type=Path, required=True, help="Path to ASR JSON (must include 'segments' & 'language')")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_dir", type=Path, default=None, help="Cache dir for align models (offline)")
    parser.add_argument("--char", action="store_true", help="Return char alignments")
    parser.add_argument("--diarize", action="store_true", help="Also run diarization and attach speakers")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token for diarization models (if required)")
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default="asr_outputs")

    args = parser.parse_args()

    # load ASR JSON
    asr = json.loads(Path(args.asr_json).read_text(encoding="utf-8"))

    # align
    aligned = align_from_asr_result(
        asr, args.audio,
        device=args.device,
        model_dir=args.model_dir,
        return_char_alignments=args.char,
    )

    # diarize (optional)
    if args.diarize:
        dia = diarize_audio(
            args.audio,
            device=args.device,
            use_auth_token=args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        aligned = attach_speakers(dia, aligned)

    # save
    save_alignment_result(aligned, args.audio, output_dir=args.out_dir)
