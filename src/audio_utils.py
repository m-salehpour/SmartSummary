# --- audio_utils.py ----------------------------------------------------------
from pathlib import Path
from typing import Tuple

import torchaudio  # pip install torchaudio
from torchaudio import functional as F


def audio_info(path: str | Path) -> Tuple[int, int, float]:
    """
    Return (num_channels, sample_rate, duration_sec) for *any* audio file
    that ffmpeg/sox/torchaudio can open.

    >>> ch, sr, dur = audio_info("clip.mp3")
    >>> print(ch, sr, dur)
    1 22050 37.42
    """
    meta = torchaudio.info(str(path))
    num_channels = meta.num_channels
    sample_rate = meta.sample_rate
    duration_sec = meta.num_frames / sample_rate
    return num_channels, sample_rate, duration_sec


def resample_to_16k(
    src_path: str | Path,
    dst_path: str | Path | None = None,
    target_sr: int = 16_000,
    mono: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Convert *any* input file to 16 kHz (optionally mono) WAV.

    Returns the output-file path (Path object).

    >>> out = resample_to_16k("clip.mp3")           # clip_16k.wav
    >>> print(out)
    clip_16k.wav
    """
    src_path = Path(src_path)
    dst_path = (
        Path(dst_path)
        if dst_path is not None
        else src_path.with_suffix("").with_suffix(f"{src_path.suffix}_16k.wav")
    )

    if dst_path.exists() and not overwrite:
        return dst_path

    # 1) load with original sample-rate
    waveform, sr = torchaudio.load(str(src_path))

    # 2) (optional) stereo→mono
    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3) resample if needed
    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)

    # 4) save
    torchaudio.save(str(dst_path), waveform, sample_rate=target_sr)
    return dst_path


def maybe_resample_to_16k(audio_file):
    ch, sr, dur = audio_info(audio_file)
    print(f"input info: (channels={ch}, sr={sr}, dur={dur})")

    # be tolerant in case sr is float (e.g., 16000.0)
    if int(round(sr)) == 16000:
        print("Already 16 kHz; skipping resample.")
        return audio_file  # no change
    else:
        wav16 = resample_to_16k(audio_file)
        print("Input file was resampled and saved to:", wav16)
        print("output info:", audio_info(wav16))
        return wav16
