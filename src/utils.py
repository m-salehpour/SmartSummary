# src/utils.py
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def save_transcription_result(
    result: dict,
    audio_path: str,
    output_dir: str = "transcripts_json",
    prefix: str = "",
) -> str:
    """
    Write a WhisperX transcription `result` dict out as JSON.

    Arguments:
      result:      The dict returned from model.transcribe(...)
      audio_path:  Path to the source audio file (used to name the JSON)
      output_dir:  Directory where the JSON will be written (created if needed)

    Returns:
      The full path to the written JSON file.
    """
    # 1) ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) derive filename from the audio file (e.g. "meeting.wav" -> "meeting.json")
    base = os.path.basename(audio_path)
    name, _ = os.path.splitext(base)
    json_path = os.path.join(output_dir, f"{prefix}_{name}.json")

    # 3) dump the transcription result
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logging.info(f"[save_transcription_result] Wrote {json_path}")

    return json_path


def cleaned_filename(path: str, suffix: str = "_cleaned") -> str:
    """
    Given a filepath like "asr_outputs/farsi_thirdsource.json",
    return "asr_outputs/farsi_thirdsource_cleaned.json".
    """
    p = Path(path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")
