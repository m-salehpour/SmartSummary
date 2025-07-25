import json
from pathlib import Path

import whisperx
from tools.comparison import segments_comparison, get_hypothesis_text
import config
from persian_normalize.context_aware_normalizer import pipeline_clean
from normalizers import cleaning as no_llm_clean
from tools.utils import (
    save_transcription_result,
    cleaned_filename,
)


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


def compare_segments(
    segments,
    ground_truth_path: Path,
    audio_path: Path,
    diff: bool,
    print_hyp: bool,
    print_ref: bool,
    msg: str,
    lang: str,
):
    print(f"\n--- Comparing {msg} ({lang}) ---\n")
    segments_comparison(
        segments,
        str(ground_truth_path),
        str(audio_path),
        diff=diff,
        print_hyp=print_hyp,
        print_ref=print_ref,
        msg=msg,
        lang=lang,
    )


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
):
    """
    1) Transcribe audio → raw segments
    2) Save raw JSON
    3) Compare raw, no-LLM-clean, and LLM-clean outputs vs. reference
    """
    audio_p = Path(audio_path)
    ref_p   = Path(ground_truth_path)

    # 1) Transcribe
    result = transcribe_audio(audio_p, model_size, device, compute_type, batch_size)
    lang   = result.get("language", "unknown")
    segments = result["segments"]

    # 2) Save raw transcript JSON
    raw_json = save_transcription_result(result, audio_path)
    print(f"✅ Raw transcript saved to {raw_json}")

    segments = get_hypothesis_text(segments)

    # 3) Compare raw segments
    compare_segments(segments, ref_p, audio_p, diff, print_hyp, print_ref, "raw transcription", lang)

    # 4) Compare no-LLM-clean segments
    cleaned_no_llm = no_llm_clean(segments, lang)
    compare_segments(cleaned_no_llm, ref_p, audio_p, diff, print_hyp, print_ref, "no-LLM cleaned", lang)

    # 5) Run LLM-based cleaner and compare
    llm_json = cleaned_filename(raw_json, suffix="_llm_cleaned")
    print(f"\n🔄 Running LLM cleaner → {llm_json}")
    pipeline_clean.main(Path(raw_json), Path(llm_json))
    print(f"✅ LLM-cleaned transcript saved to {llm_json}")

    with open(llm_json, encoding="utf-8") as f:
        llm_result = json.load(f)

    segments = get_hypothesis_text(llm_result["segments"])
    compare_segments(
        segments,
        ref_p,
        audio_p,
        diff,
        print_hyp,
        print_ref,
        "LLM-normalized transcription",
        lang,
    )

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