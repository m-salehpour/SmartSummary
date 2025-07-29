# asr.py
import json
import logging
from pathlib import Path
from typing import Optional, Dict

import whisperx

import config
from init_env import FW_MEDIUM_DIR, HF_ROOT

# keep your existing imports
from normalizers import cleaning as no_llm_clean
from tools.normalizers import _run_llm_clean  # still available if you call it directly
from tools.utils import save_transcription_result, cleaned_filename

# keep comparison primitives for compatibility (some may be unused now)

# NEW: import the orchestration helpers
from tools.compare_runner import (
    run_basic_comparisons,
    run_no_llm_clean_and_compare,
    run_llm_clean_from_raw,
    run_llm_clean_from_no_llm, get_hypothesis_text, load_reference_text, segments_comparison, _compare_strs,
)

logger = logging.getLogger(__name__)

# ---- Defaults for ASR VAD ----
DEFAULT_ASR_OPTIONS: Dict = {
    "no_speech_threshold": 0.30,
    "condition_on_previous_text": False,
}
DEFAULT_VAD_METHOD: str = "silero"
DEFAULT_VAD_OPTIONS: Dict = {
    "chunk_size": 60,
    "vad_onset": 0.30,
    "vad_offset": 0.20,
}

def transcribe_audio(
    audio_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
    batch_size: int,
) -> dict:
    """
    Load WhisperX model, transcribe `audio_path`, and return the raw result dict.
    """
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    return model.transcribe(audio, batch_size=batch_size)


def evaluate_transcription(
    audio_path: str,
    ground_truth_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "float32",
    batch_size: int = 16,
    diff: bool = False,
    print_hyp: bool = False,
    print_ref: bool = False,
    llm_clean: bool = False,
    asr_options: Optional[Dict] = None,
    vad_method: Optional[str] = None,
    vad_options: Optional[Dict] = None,
    print_progress: bool = True,
    fallback: str = config.FALLBACK_POLICY_FULL,
    strip_speakers: bool = False,
    script_hint: Optional[str] = None,  # e.g., "latin", "hebrew", "arabic"
):
    """
    1) Transcribe audio → raw segments
    2) Save raw JSON
    3) Compare raw, no-LLM-clean, LLM-clean, and LLM(no-LLM) outputs vs. reference
    """
    audio_p = Path(audio_path)
    ref_p   = Path(ground_truth_path)

    # Merge shallow copies (caller values override defaults)
    merged_asr = {**DEFAULT_ASR_OPTIONS, **(asr_options or {})}
    used_vad_method = vad_method or DEFAULT_VAD_METHOD
    merged_vad = {**DEFAULT_VAD_OPTIONS, **(vad_options or {})}

    logger.info(f"[evaluate_transcription] loading model from...: {FW_MEDIUM_DIR}")

    # 1) Transcribe (fully offline)
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        asr_options=merged_asr,
        vad_method=used_vad_method,
        vad_options=merged_vad,
        download_root=str(HF_ROOT),   # local HF cache
        local_files_only=True,        # no network
    )
    audio = whisperx.load_audio(str(audio_p))
    result = model.transcribe(audio, batch_size=batch_size, print_progress=print_progress)

    lang     = result.get("language", "unknown")
    raw_segs = result["segments"]

    # 2) Save raw transcript JSON
    raw_json = save_transcription_result(result, audio_path)
    print(f"✅ Raw transcript saved to {raw_json}")

    # 3) Basic comparisons (RAW)
    base = run_basic_comparisons(
        raw_segments=raw_segs,
        ref_path=ref_p,
        lang=lang,
        diff=diff,
        print_hyp=print_hyp,
        print_ref=print_ref,
        fallback=fallback,
        strip_speakers=strip_speakers,
        script_hint=script_hint,
    )
    ref_text   = base["ref_text"]
    hyp_raw    = base["hyp_raw"]

    # 4) No-LLM cleaned comparison
    hyp_no_llm = no_llm_clean(hyp_raw, lang).strip()
    no_llm_res = run_no_llm_clean_and_compare(
        hyp_no_llm=hyp_no_llm,
        ref_text=ref_text,
        diff=diff,
        print_hyp=print_hyp,
    )

    # 5) LLM-based variants (optional)
    if llm_clean:
        _ = run_llm_clean_from_raw(
            raw_json_path=Path(raw_json),
            ref_text=ref_text,
            diff=diff,
            print_hyp=print_hyp,
            suffix="_llm_from_raw",
        )

        _ = run_llm_clean_from_no_llm(
            raw_json_path=Path(raw_json),
            hyp_no_llm=hyp_no_llm,
            ref_text=ref_text,
            diff=diff,
            print_hyp=print_hyp,
            suffix_intermediate="_no_llm_cleaned",
            suffix_final="_llm_from_no_llm",
        )

    return result

#
# def align_whisper_output(
#     result,
#     audio_path: str,
#     ground_truth_path: str,
#     device: str     = "cpu",
#     diff: bool      = False,
#     print_hype_text: bool     = False,
#     print_ref_text: bool      = False,
# ):
#
#     audio = whisperx.load_audio(audio_path)
#
#     # 2. Align whisper output
#     model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, )
#     result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
#
#     segments_comparison(result["segments"], ground_truth_path, audio_path, diff=diff, print_hyp=print_hype_text, print_ref=print_ref_text, msg="aligned", lang=result["language"])
#
