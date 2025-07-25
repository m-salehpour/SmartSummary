#!/usr/bin/env python3

# ─── 1) Environment & config ──────────────────────────────────────────────────
import init_env    # runs init_env.py top‐level setup
import config      # runs config.py top‐level setup

# ─── 2) Imports of your tools ─────────────────────────────────────────────────
from tools.profiler import profile_resources
from asr import evaluate_transcription

# result = evaluate_transcription(audio_path=audio_file, ground_truth_path=ground_truth_file, diff=False, print_hype_text=True , print_ref_text=True, model_size="medium" )

# @profile_resources
def run_experiment():

    # audio_file = "../Data/Training/Farsi/Quran/farsi_secondsource.mp3"
    # ground_truth_file = "../Data/Training/Farsi/Quran/farsi_secondsource_transcript-farsi_translation-hebrew.docx"

    audio_file = "../Data/Training/Farsi/Quran/farsi_thirdsource.mp3"
    ground_truth_file = "../Data/Training/Farsi/Quran/farsi_thirdsource_transcript-farsi_translation-hebrew.docx"

    # audio_file = "../Data/Training/Hebrew/hebrew_firstsource.mp4"
    # ground_truth_file = "../Data/Training/Hebrew/hebrew_firstsource_transcript-hebrew_translation-english.docx"

    return evaluate_transcription(
        audio_path=audio_file,
        ground_truth_path=ground_truth_file,
        diff=False,
        print_hyp=True,
        print_ref=True,
        model_size="medium"
    )

if __name__ == "__main__":
    # Now execute:
    run_experiment()


# model medium farsi_secondsource
##only hazm 33%
##only p_norm 30%
##both + nevis 29.23% upto 31%

# model medium farsi_secondsource
##both + nevis 31%


# align_whisper_output(result=result, audio_path=audio_file, ground_truth_path=ground_truth_file, diff=True, print_hype_text=True , print_ref_text=True )
