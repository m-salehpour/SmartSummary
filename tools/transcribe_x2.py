#!/usr/bin/env python3
import json

from typing import Optional

# ─── 1) Environment & config ──────────────────────────────────────────────────
import init_env    # runs init_env.py top‐level setup
import config      # runs config.py top‐level setup
from config import FW_LARGE_V2_DIR
from tools.align import align_and_diarize
from tools.batch_runner import run_batch_from_folder

# ─── 2) Imports of your tools ─────────────────────────────────────────────────
from tools.profiler import profile_resources
from asr import evaluate_transcription

import logging

logger = logging.getLogger(__name__)


import time

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

logger = logging.getLogger(__name__)

@profile_resources
def run_asr_batch_from_folder(folder: str,
                              limit: Optional[int] = None,
                              prefix: Optional[str] = None,
                              model_size: Optional[str] = FW_LARGE_V2_DIR.resolve().as_posix(),
                              device: Optional[str] = config.DEVICE,
                              compute_type: Optional[str] = config.QUANT_TYPE_FLOAT_32,
                              batch_size: Optional[int] = config.BATCH_SIZE,
                              diff: bool = False,
                              print_hyp: bool = False,
                              print_ref: bool = False,
                              llm_clean: bool = False,
                              print_progress: bool = True,
                              fallback: str = config.FALLBACK_POLICY_FULL,
                              strip_speakers: bool = False,
                              script_hint: Optional[str] = None,  # e.g., "latin", "hebrew", "arabic"
                              ):
    run_batch_from_folder(
        folder=folder,
        limit=limit,  # or an int like 5
        prefix=prefix,  # or e.g. "response_1730995"

        model_size=model_size,
        device=device,
        # compute_type=config.QUANT_TYPE,
        compute_type=compute_type,
        batch_size=batch_size,
        diff=diff,
        print_hyp=print_hyp,
        print_ref=print_ref,
        llm_clean=llm_clean,  # set False if you want to skip LLM passes
        asr_options={"no_speech_threshold": 0.30},
        vad_method="silero",
        vad_options={"chunk_size": 30, "vad_onset": 0.30, "vad_offset": 0.20},
        print_progress=print_progress,
        fallback=fallback,
        strip_speakers=strip_speakers,
        script_hint=script_hint,
    )


@profile_resources
def run_experiment(is_asr_enabled: bool = True,
                   is_align_and_diarize_enabled: bool = False,
                   audio_file: str = None,
                   ground_truth_file: str = None,
                   fallback: str = None,):

    if not audio_file or not ground_truth_file:
        raise Exception("audio_file and ground_truth_file are required")

    start_ts = time.perf_counter()
    logger.info("🚀 Starting run_experiment")

    # 2 coarse steps: ASR (+ comparisons) and Align+Diarize
    with tqdm(total=2, desc="Pipeline", unit="step") as pbar:

        if is_asr_enabled:
            # ---- STEP 1: ASR + comparisons --------------------------------------
            logger.info("🎧 [1/2] ASR + comparisons → evaluate_transcription(...)")
            t0 = time.perf_counter()
            try:
                asr_result = evaluate_transcription(
                    audio_path=audio_file,
                    ground_truth_path=ground_truth_file,
                    model_size=FW_LARGE_V2_DIR.resolve().as_posix(),
                    device=config.DEVICE,
                    # compute_type=config.QUANT_TYPE,
                    compute_type=config.QUANT_TYPE_FLOAT_32,
                    batch_size=config.BATCH_SIZE,
                    diff=False,
                    print_hyp=True,
                    print_ref=True,
                    llm_clean=False,  # set False if you want to skip LLM passes
                    asr_options={"no_speech_threshold": 0.30},
                    vad_method="silero",
                    vad_options={"chunk_size": 30, "vad_onset": 0.30, "vad_offset": 0.20},
                    print_progress=True,
                    fallback=fallback,
                    strip_speakers=True,
                    # script_hint="latin",
                )
                logger.info("✅ ASR step completed in %.1fs", time.perf_counter() - t0)
                pbar.update(1)
            except Exception:
                logger.exception("❌ ASR step failed")
                raise

        if is_align_and_diarize_enabled:
            # ---- STEP 2: Align + Diarize ----------------------------------------
            logger.info("🧩 [2/2] Alignment + Diarization → align_and_diarize(...)")
            t1 = time.perf_counter()
            try:
                res = align_and_diarize(
                    asr_result=asr_result,
                    audio_path=audio_file,
                    device=config.DEVICE,
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

        # ---- Final report --------------------------------------------------------
        elapsed = time.perf_counter() - start_ts
        logger.info("✅ Done. Total elapsed: %.1fs", elapsed)

        if is_align_and_diarize_enabled:
            logger.info(
                json.dumps(
                    {k: (str(v)[:200] if isinstance(v, str) else type(v).__name__) for k, v in res.items()},
                    ensure_ascii=False,
                    indent=2,
                )
            )

    return asr_result if not is_align_and_diarize_enabled else res


if __name__ == "__main__":

    folder_farsi_youtube = "../Data/Training/Farsi/Quran" #"../Data/Training/Farsi/youtube_transcripts",
    prefix_test = "farsi_secondsource" # "response_1730994879150"
    limit_f = None # 1

    run_asr_batch_from_folder(folder=folder_farsi_youtube,
                              limit=limit_f,
                              fallback=config.FALLBACK_POLICY_FULL,
                              model_size=FW_LARGE_V2_DIR.resolve().as_posix(),
                              device=config.DEVICE,
                              compute_type=config.QUANT_TYPE_INT_8,
                              batch_size=config.BATCH_SIZE,
                              llm_clean=False,
                              prefix=prefix_test,
                              )

    # audio_file = "../Data/Training/Farsi/youtube_transcripts/response_1731000306032_output-farsi.mp3"
    # ground_truth_file = "../Data/Training/Farsi/youtube_transcripts/response_1731000306032_output-farsi.txt"
    #
    #
    # print(audio_info(audio_file))
    # # → (2, 48000, 12.875)
    #
    # wav16 = resample_to_16k(audio_file)
    # print("input file is resampled and saved to: ", wav16)
    # print(audio_info(wav16))
    # # → (1, 16000, 12.875)
    #
    # run_experiment(is_asr_enabled=True,
    #                is_align_and_diarize_enabled=False,
    #                audio_file=str(wav16),
    #                ground_truth_file=ground_truth_file,
    #                fallback=config.FALLBACK_POLICY_FULL)



# model medium farsi_secondsource
##only hazm 33%
##only p_norm 30%
##both + nevis 29.23% upto 31%

# model medium farsi_secondsource
##both + nevis 31%


# align_whisper_output(result=result, audio_path=audio_file, ground_truth_path=ground_truth_file, diff=True, print_hype_text=True , print_ref_text=True )
