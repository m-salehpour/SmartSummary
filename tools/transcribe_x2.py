#!/usr/bin/env python3
import json
from pathlib import Path

# ─── 1) Environment & config ──────────────────────────────────────────────────
import init_env    # runs init_env.py top‐level setup
import config      # runs config.py top‐level setup
from init_env import FW_MEDIUM_DIR
from tools.align import align_from_asr_result, save_alignment_result, diarize_audio, save_diarization_json, \
    align_and_diarize

# ─── 2) Imports of your tools ─────────────────────────────────────────────────
from tools.profiler import profile_resources
from asr import evaluate_transcription

import logging

logger = logging.getLogger(__name__)


# result = evaluate_transcription(audio_path=audio_file, ground_truth_path=ground_truth_file, diff=False, print_hype_text=True , print_ref_text=True, model_size="medium" )

@profile_resources
def run_experiment():

    # audio_file = "../Data/Training/Farsi/Quran/farsi_secondsource.mp3"
    # ground_truth_file = "../Data/Training/Farsi/Quran/farsi_secondsource_transcript-farsi_translation-hebrew.docx"

    # audio_file = "../Data/Training/Farsi/Quran/farsi_thirdsource.mp3"
    # ground_truth_file = "../Data/Training/Farsi/Quran/farsi_thirdsource_transcript-farsi_translation-hebrew.docx"

    # audio_file = "../Data/Training/Hebrew/hebrew_firstsource.mp4"
    # ground_truth_file = "../Data/Training/Hebrew/hebrew_firstsource_transcript-hebrew_translation-english.docx"

    # audio_file = "../Data/Training/English/Churchill/english_firstsourcecommons_13_churchill_64kb.mp3"
    # ground_truth_file = "../Data/Training/English/Churchill/english_firstsourcecommons_13_churchill_transcript-english_translation_hebrew.docx"

    audio_file = "../Data/Training/English/Aridia Conference Call/Aridia Conference Call.mp4"
    ground_truth_file = "../Data/Training/English/Aridia Conference Call/Aridia Conference Transcript.docx"

    asr_result = evaluate_transcription(
        audio_path=audio_file,
        ground_truth_path=ground_truth_file,
        model_size=FW_MEDIUM_DIR.resolve().as_posix(),
        device=config.DEVICE,
        compute_type=config.QUANT_TYPE,
        batch_size=config.BATCH_SIZE,
        diff=False,
        print_hyp=True,
        print_ref=True,
        llm_clean=True,  # set False if you want to skip LLM passes
        asr_options={"no_speech_threshold": 0.30},
        vad_method="silero",
        vad_options={"chunk_size": 60, "vad_onset": 0.30, "vad_offset": 0.20},
        print_progress=True,
        fallback=config.FALLBACK_POLICY_DIALOGUE,
        strip_speakers=True,
        script_hint="latin",
    )


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

    logger.info("✅ Done.")
    logger.info(json.dumps(
        {k: (str(v)[:200] if isinstance(v, str) else type(v).__name__) for k, v in res.items()},
        ensure_ascii=False, indent=2
    ))


    # # now align
    # diarize = diarize_audio(
    #         audio_file,
    #         device=config.DEVICE,
    #         use_auth_token=config.HF_TOKEN,
    #     )
    # logging.info(f"diarization result: {diarize}")
    #
    # save_diarization_json(diarize, audio_file, config.JSON_ASR_OUTPUT_DIR)
    # logging.info(f"✅ Diarized transcript written saved.")

    # save_alignment_result(aligned, audio_file)

    return res
    #
    # wav = whisperx.load_audio(audio)
    # align_model, align_meta = load_alignment_model(
    #     language_code=result.get("language", "en"),
    #     device=config.DEVICE,
    #     download_root=str(config.HF_ROOT),
    #     local_files_only=True,
    # )
    #
    # aligned = run_alignment(
    #     segments=result["segments"],
    #     audio=wav,
    #     align_model=align_model,
    #     align_metadata=align_meta,
    #     device=config.DEVICE,
    # )
    #
    # out = Path(audio).with_suffix(".aligned.json")
    # out.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    # print(f"✅ Aligned transcript written to {out}")

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
