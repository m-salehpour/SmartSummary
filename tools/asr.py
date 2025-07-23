import whisperx
from tools.comparison import get_hypothesis_text, load_reference_text, compare_texts, segments_comparison

def evaluate_transcription(
    audio_path: str,
    ground_truth_path: str,
    model_size: str = "small",
    device: str     = "cpu",
    compute_type: str = "float32",
    batch_size: int = 16,
    diff: bool      = False,
    print_hype_text: bool     = False,
    print_ref_text: bool      = False,
):
    """
    1) Load Whisper ASR, transcribe `audio_path` → segments
    2) Build hypothesis text
    3) Load reference text from ground_truth_path
    4) Compute & print WER (+ optional diff)
    """
    # 1. load & transcribe
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    segments_comparison(result["segments"], ground_truth_path, audio_path, diff=diff, print_hype_text=print_hype_text, print_ref_text=print_ref_text, msg="transcription", lang=result["language"])

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

    segments_comparison(result["segments"], ground_truth_path, audio_path, diff=diff, print_hype_text=print_hype_text, print_ref_text=print_ref_text, msg="aligned", lang=result["language"])