#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import time
from typing import Optional, Dict, Any

# ─── 1) Environment & config ──────────────────────────────────────────────────
import init_env    # runs init_env.py top‐level setup
import config      # runs config.py top‐level setup
from config import FW_LARGE_V2_DIR

# Audio / pipeline utils (kept even if unused in some paths)
from tools.align import align_and_diarize
from tools.audio_utils import audio_info, resample_to_16k, maybe_resample_to_16k
from tools.batch_runner import run_batch_from_folder

# ─── 2) Your tools ────────────────────────────────────────────────────────────
from tools.profiler import profile_resources
from asr import evaluate_transcription

# ─── 3) Logging ───────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# lightweight fallback if tqdm isn't available
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, total=None, desc=None, unit=None):
            self.total = total
        def update(self, n=1): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): self.close()


# ─── 4) Centralized option builders ───────────────────────────────────────────

def build_asr_options(**overrides: Any) -> Dict[str, Any]:
    """
    Central source of truth for ASR options.
    Use overrides to tweak per-call without duplicating the whole dict.
    """
    opts: Dict[str, Any] = {
        "no_repeat_ngram_size": 3,
        "beam_size": 10,
        "best_of": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperatures": [0.0, 0.2],  # start deterministic; broaden only if stuck
        "compression_ratio_threshold": 2.3,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.30,  # lenient enough to keep quiet intros
        "condition_on_previous_text": True,  # improves cohesion in long speech
        "suppress_blank": True,
        "suppress_tokens": [-1, 1, 2, 3],  # use model defaults
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "hotwords": None,
        "repetition_penalty": 1.05,
    }
    opts.update(overrides)
    return opts


def build_vad_options(**overrides: Any) -> Dict[str, Any]:
    """
    Central source of truth for VAD options.
    """
    opts: Dict[str, Any] = {
        "chunk_size": 30,
        "vad_onset": 0.15,
        "vad_offset": 0.25,
    }
    opts.update(overrides)
    return opts


def _log_kv(title: str, mapping: Dict[str, Any]) -> None:
    logger.info("%s:\n%s", title, json.dumps(mapping, ensure_ascii=False, indent=2))


DEFAULT_MODEL = FW_LARGE_V2_DIR.resolve().as_posix()
DEFAULT_DEVICE = config.DEVICE
DEFAULT_COMPUTE = config.QUANT_TYPE_FLOAT_32
DEFAULT_BATCH = config.BATCH_SIZE
DEFAULT_FALLBACK = config.FALLBACK_POLICY_FULL


# ─── 5) Batch ASR runner (authoritative calling pattern) ──────────────────────

@profile_resources
def run_asr_batch_from_folder(
    folder: str,
    limit: Optional[int] = None,
    prefix: Optional[str] = None,
    model_size: Optional[str] = DEFAULT_MODEL,
    device: Optional[str] = DEFAULT_DEVICE,
    compute_type: Optional[str] = DEFAULT_COMPUTE,
    batch_size: Optional[int] = DEFAULT_BATCH,
    diff: bool = False,
    print_hyp: bool = False,
    print_ref: bool = False,
    llm_clean: bool = False,
    print_progress: bool = True,
    fallback: str = DEFAULT_FALLBACK,
    strip_speakers: bool = False,
    script_hint: Optional[str] = None,   # e.g., "latin", "hebrew", "arabic"
    language: Optional[str] = None,
) -> None:
    """
    Batch-process every audio in a folder using the canonical ASR invocation pattern.
    """
    asr_opt = build_asr_options()
    vad_opt = build_vad_options()

    _log_kv("[run_asr_batch_from_folder] asr_options", asr_opt)
    _log_kv("[run_asr_batch_from_folder] vad_options", vad_opt)
    logger.info(
        "[run_asr_batch_from_folder] args:\n"
        f"folder={folder}\nlimit={limit}\nprefix={prefix}\n"
        f"model_size={model_size}\ndevice={device}\ncompute_type={compute_type}\n"
        f"batch_size={batch_size}\ndiff={diff}\nprint_hyp={print_hyp}\nprint_ref={print_ref}\n"
        f"llm_clean={llm_clean}\nprint_progress={print_progress}\nfallback={fallback}\n"
        f"strip_speakers={strip_speakers}\nscript_hint={script_hint}\nlanguage={language}"
    )

    run_batch_from_folder(
        folder=folder,
        limit=limit,
        prefix=prefix,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        diff=diff,
        print_hyp=print_hyp,
        print_ref=print_ref,
        llm_clean=llm_clean,
        asr_options=asr_opt,
        vad_method="silero",
        vad_options=vad_opt,
        print_progress=print_progress,
        fallback=fallback,
        strip_speakers=strip_speakers,
        script_hint=script_hint,
        language=language,
    )


# ─── 6) Single-file experiment pipeline ───────────────────────────────────────

@profile_resources
def run_experiment(
    is_asr_enabled: bool = True,
    is_align_and_diarize_enabled: bool = False,
    audio_file: Optional[str] = None,
    ground_truth_file: Optional[str] = None,
    fallback: Optional[str] = DEFAULT_FALLBACK,
    *,
    model_size: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE,
    batch_size: int = DEFAULT_BATCH,
    diff: bool = False,
    print_hyp: bool = True,
    print_ref: bool = True,
    llm_clean: bool = False,
    print_progress: bool = True,
    strip_speakers: bool = True,
    script_hint: Optional[str] = None,
    language: Optional[str] = None,
    resample_if_needed: bool = True,  # only resample when not already 16 kHz
) -> dict:
    """
    Two coarse steps: (1) ASR + comparisons, (2) optional align + diarize.
    Uses the same ASR/VAD option style as run_asr_batch_from_folder.
    """
    if not audio_file or not ground_truth_file:
        raise ValueError("audio_file and ground_truth_file are required")

    start_ts = time.perf_counter()
    logger.info("🚀 Starting run_experiment")

    # Centralized options
    asr_opt = build_asr_options()
    vad_opt = build_vad_options()

    _log_kv("[run_experiment] asr_options", asr_opt)
    _log_kv("[run_experiment] vad_options", vad_opt)

    # Optional pre-resample (only if not already 16k)
    if resample_if_needed:
        try:
            before = audio_info(audio_file)
            audio_file = maybe_resample_to_16k(audio_file)
            after = audio_info(audio_file)
            logger.info(f"[run_experiment] audio_info before={before}, after={after}")
        except Exception:
            logger.exception("[run_experiment] optional resample check failed (continuing).")

    asr_result = {}
    res = {}

    with tqdm(total=2 if is_align_and_diarize_enabled else 1, desc="Pipeline", unit="step") as pbar:
        if is_asr_enabled:
            # ---- STEP 1: ASR + comparisons ----------------------------------
            logger.info("🎧 [1/2] ASR + comparisons → evaluate_transcription(...)")
            t0 = time.perf_counter()
            try:
                asr_result = evaluate_transcription(
                    audio_path=audio_file,
                    ground_truth_path=ground_truth_file,
                    model_size=model_size,
                    device=device,
                    compute_type=compute_type,
                    batch_size=batch_size,
                    diff=diff,
                    print_hyp=print_hyp,
                    print_ref=print_ref,
                    llm_clean=llm_clean,
                    asr_options=asr_opt,
                    vad_method="silero",
                    vad_options=vad_opt,            # ← fixed: was asr_opt
                    print_progress=print_progress,
                    fallback=fallback,
                    strip_speakers=strip_speakers,
                    script_hint=script_hint,
                    language=language,
                )
                logger.info("✅ ASR step completed in %.1fs", time.perf_counter() - t0)
                pbar.update(1)
            except Exception:
                logger.exception("❌ ASR step failed")
                raise

        if is_align_and_diarize_enabled:
            # ---- STEP 2: Align + Diarize ------------------------------------
            logger.info("🧩 [2/2] Alignment + Diarization → align_and_diarize(...)")
            t1 = time.perf_counter()
            try:
                res = align_and_diarize(
                    asr_result=asr_result,
                    audio_path=audio_file,
                    device=device,
                    do_diarize=True,
                    hf_token=config.HF_TOKEN,
                    out_dir=config.JSON_ASR_OUTPUT_DIR,
                    save_diarization_as_json=True,
                    parallel=True,
                    verbose=True,
                )
                logger.info("✅ Align+Diarize step completed in %.1fs", time.perf_counter() - t1)
                pbar.update(1)
            except Exception:
                logger.exception("❌ Align+Diarize step failed")
                raise

    elapsed = time.perf_counter() - start_ts
    logger.info("✅ Done. Total elapsed: %.1fs", elapsed)

    if is_align_and_diarize_enabled:
        # Log a compact digest of returned keys/types (avoid dumping large blobs)
        logger.info(
            json.dumps(
                {k: (str(v)[:200] if isinstance(v, str) else type(v).__name__) for k, v in res.items()},
                ensure_ascii=False,
                indent=2,
            )
        )

    return asr_result if not is_align_and_diarize_enabled else res


# ─── 7) Optional: tiny CLI façade (kept non-invasive) ─────────────────────────
if __name__ == "__main__":

    audio_file = "../Data/Test/hebrew_thirdsource.mp3_16k.wav"
    ground_truth_file = "../Data/Test/hebrew_thirdsource.txt"

    result = run_experiment(
        is_asr_enabled=True,
        is_align_and_diarize_enabled=True,  # set False to skip diarization
        audio_file=audio_file,
        ground_truth_file=ground_truth_file,
        fallback=config.FALLBACK_POLICY_FULL,
        model_size=config.FW_LARGE_V3_DIR.resolve().as_posix(),
        device=config.DEVICE,
        compute_type=config.QUANT_TYPE_FLOAT_32,
        batch_size=config.BATCH_SIZE,
        diff=False,
        print_hyp=True,
        print_ref=True,
        llm_clean=True,
        print_progress=True,
        strip_speakers=True,
        # script_hint="hebrew",
        # language="he",
        resample_if_needed=True,  # only resamples if not already 16 kHz
    )

    # result is ASR dict by default; if align+diarize=True, it returns that result instead
    print("Keys in result:", list(result.keys()))

    # import argparse

    # parser = argparse.ArgumentParser(description="ASR batch/experiment runner")
    # sub = parser.add_subparsers(dest="cmd", required=False)
    #
    # p_batch = sub.add_parser("batch", help="Run ASR over a folder")
    # p_batch.add_argument("--folder", required=True)
    # p_batch.add_argument("--limit", type=int)
    # p_batch.add_argument("--prefix")
    # p_batch.add_argument("--model_size", default=DEFAULT_MODEL)
    # p_batch.add_argument("--device", default=DEFAULT_DEVICE)
    # p_batch.add_argument("--compute_type", default=DEFAULT_COMPUTE)
    # p_batch.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    # p_batch.add_argument("--diff", action="store_true")
    # p_batch.add_argument("--print_hyp", action="store_true")
    # p_batch.add_argument("--print_ref", action="store_true")
    # p_batch.add_argument("--llm_clean", action="store_true")
    # p_batch.add_argument("--fallback", default=DEFAULT_FALLBACK)
    # p_batch.add_argument("--strip_speakers", action="store_true")
    # p_batch.add_argument("--script_hint")
    # p_batch.add_argument("--language")
    # p_batch.add_argument("--no_progress", action="store_true")
    #
    # p_exp = sub.add_parser("experiment", help="Run single-file experiment")
    # p_exp.add_argument("--audio_file", required=True)
    # p_exp.add_argument("--ground_truth_file", required=True)
    # p_exp.add_argument("--align", action="store_true", help="Enable align+diarize")
    # p_exp.add_argument("--model_size", default=DEFAULT_MODEL)
    # p_exp.add_argument("--device", default=DEFAULT_DEVICE)
    # p_exp.add_argument("--compute_type", default=DEFAULT_COMPUTE)
    # p_exp.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    # p_exp.add_argument("--fallback", default=DEFAULT_FALLBACK)
    # p_exp.add_argument("--script_hint")
    # p_exp.add_argument("--language")
    # p_exp.add_argument("--no_resample", action="store_true")
    #
    # args = parser.parse_args()
    #
    # if args.cmd == "batch":
    #     run_asr_batch_from_folder(
    #         folder=args.folder,
    #         limit=args.limit,
    #         prefix=args.prefix,
    #         model_size=args.model_size,
    #         device=args.device,
    #         compute_type=args.compute_type,
    #         batch_size=args.batch_size,
    #         diff=args.diff,
    #         print_hyp=args.print_hyp,
    #         print_ref=args.print_ref,
    #         llm_clean=args.llm_clean,
    #         print_progress=not args.no_progress,
    #         fallback=args.fallback,
    #         strip_speakers=args.strip_speakers,
    #         script_hint=args.script_hint,
    #         language=args.language,
    #     )
    # elif args.cmd == "experiment":
    #     run_experiment(
    #         is_asr_enabled=True,
    #         is_align_and_diarize_enabled=args.align,
    #         audio_file=args.audio_file,
    #         ground_truth_file=args.ground_truth_file,
    #         model_size=args.model_size,
    #         device=args.device,
    #         compute_type=args.compute_type,
    #         batch_size=args.batch_size,
    #         fallback=args.fallback,
    #         script_hint=args.script_hint,
    #         language=args.language,
    #         resample_if_needed=not args.no_resample,
    #     )
    # else:
    #     parser.print_help()
