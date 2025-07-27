import json
import logging
from pathlib import Path

import whisperx
from init_env import FW_MEDIUM_DIR, HF_ROOT

from normalizers import cleaning as no_llm_clean
from tools.normalizers import _run_llm_clean
from tools.utils import (
    save_transcription_result,
    cleaned_filename,
)

from tools.comparison import (
    segments_comparison,
    get_hypothesis_text,
    load_reference_text,
    _compare_strs,
)

logger = logging.getLogger(__name__)


# ---- Defaults for ASR VAD ----
from typing import Optional, Dict, Union
DEFAULT_ASR_OPTIONS: Dict = {
    "no_speech_threshold": 0.30,          # default 0.6 – be less eager to drop
    "condition_on_previous_text": False,  # fine either way for the opener
}
DEFAULT_VAD_METHOD: str = "silero"        # often more permissive than pyannote
DEFAULT_VAD_OPTIONS: Dict = {
    "chunk_size": 60,   # larger chunks → fewer tiny early cuts
    "vad_onset": 0.30,  # default ~0.50 – lower so quiet speech is kept
    "vad_offset": 0.20, # default ~0.363 – release later to avoid early cut
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
    print_progress: bool = True
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

    # 1) Transcribe
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        asr_options=merged_asr,
        vad_method=used_vad_method,
        vad_options=merged_vad,
        download_root=str(HF_ROOT),   # <-- use local HF cache
        local_files_only=True,        # <-- do not hit the network
    )

    audio = whisperx.load_audio(str(audio_p))
    result = model.transcribe(audio, batch_size=batch_size, print_progress=print_progress)

    lang     = result.get("language", "unknown")
    raw_segs = result["segments"]

    # 2) Save raw transcript JSON
    raw_json = save_transcription_result(result, audio_path)
    print(f"✅ Raw transcript saved to {raw_json}")

    # prepare reference text once
    ref_text = load_reference_text(str(ref_p)).strip()
    if print_ref:
        print("\n[REF TEXT]\n", ref_text)

    # 3) RAW comparison
    hyp_raw = get_hypothesis_text(raw_segs).strip()
    if print_hyp:
        print("\n[HYP RAW]\n", hyp_raw)
    print("\n=== RAW TRANSCRIPTION ===")
    _compare_strs(hyp_raw, ref_text, diff=diff)

    # 4) NO-LLM clean comparison
    hyp_no_llm = no_llm_clean(hyp_raw, lang).strip()
    if print_hyp:
        print("\n[HYP NO-LLM CLEAN]\n", hyp_no_llm)
    print("\n=== NO-LLM CLEANED ===")
    _compare_strs(hyp_no_llm, ref_text, diff=diff)

    if llm_clean:
        # 5A) LLM clean of RAW → compare
        llm_raw_json = _run_llm_clean(Path(raw_json), suffix="_llm_from_raw")
        llm_raw_data = json.loads(llm_raw_json.read_text(encoding="utf-8"))
        hyp_llm_raw  = get_hypothesis_text(llm_raw_data["segments"]).strip()
        if print_hyp:
            print("\n[HYP LLM(from raw)]\n", hyp_llm_raw)
        print("\n=== LLM-CLEANED FROM RAW ===")
        _compare_strs(hyp_llm_raw, ref_text, diff=diff)

        # 5B) LLM clean of NO-LLM → compare
        # reuse the no-LLM cleaned transcript JSON (or generate a JSON from hyp_no_llm first)
        # for simplicity assume we can write hyp_no_llm segments back to a JSON:
        no_llm_json = cleaned_filename(raw_json, suffix="_no_llm_cleaned")
        # write out a minimal JSON structure:
        with open(no_llm_json, "w", encoding="utf-8") as f:
            json.dump({"segments": [{"text": hyp_no_llm}]}, f, indent=2)
        llm_no_llm_json = _run_llm_clean(Path(no_llm_json), suffix="_llm_from_no_llm")
        llm_no_llm_data = json.loads(llm_no_llm_json.read_text(encoding="utf-8"))
        hyp_llm_no_llm  = get_hypothesis_text(llm_no_llm_data["segments"]).strip()
        if print_hyp:
            print("\n[HYP LLM(from no-LLM)]\n", hyp_llm_no_llm)
        print("\n=== LLM-CLEANED FROM NO-LLM ===")
        _compare_strs(hyp_llm_no_llm, ref_text, diff=diff)

    return result

def align_whisper_output(
    result,
    audio_path: str,
    ground_truth_path: str,
    device: str     = "cpu",
    diff: bool      = False,
    print_hype_text: bool     = False,
    print_ref_text: bool      = False,
):

    audio = whisperx.load_audio(audio_path)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, )
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    segments_comparison(result["segments"], ground_truth_path, audio_path, diff=diff, print_hyp=print_hype_text, print_ref=print_ref_text, msg="aligned", lang=result["language"])

