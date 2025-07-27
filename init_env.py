from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
import sys, subprocess, logging
from pathlib import Path
from download_utils import _download_with_gdown

import logging

# ─── Configure root logger for INFO+ to stdout ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ─── Prepend Homebrew to PATH ──────────────────────────────────────────────────
# so that subprocesses can find /opt/homebrew/bin/ffmpeg, etc.
import os
homebrew_bin = "/opt/homebrew/bin"
os.environ["PATH"] = f"{homebrew_bin}:{os.environ.get('PATH', '')}"
import nest_asyncio
nest_asyncio.apply()


# ────────────────────────────────────────────────────────────────────────────────
# Config: flip this via environment variable BOOTSTRAP_OFFLINE=1 on first run
# ────────────────────────────────────────────────────────────────────────────────
BOOTSTRAP = os.getenv("BOOTSTRAP_OFFLINE", "1").lower() in {"1", "true", "yes"}

# Project paths
ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"              # all cached models live here
HF_ROOT      = MODELS_ROOT / "hf"          # huggingface snapshots
TORCH_HUB    = MODELS_ROOT / "torch_hub"   # torch hub cache (e.g., silero)
NLTK_DATA    = MODELS_ROOT / "nltk_data"   # punkt, etc.

# Local dirs for the 2 models you asked
FW_MEDIUM_DIR = HF_ROOT / "Systran" / "faster-whisper-medium"
BERT_FA_DIR  = HF_ROOT / "HooshvareLab" / "bert-fa-base-uncased"


# Nevise assets (Persian)
NEVISE_DIR   = ROOT / "tools" / "persian_normalize" / "Nevise" / "model"
NEVISE_VOCAB = NEVISE_DIR / "vocab.pkl"
NEVISE_CKPT  = NEVISE_DIR / "model.pth.tar"

# Optional: make runtime default to local caches & offline
os.environ.setdefault("HF_HOME", str(HF_ROOT))
os.environ.setdefault("TORCH_HOME", str(TORCH_HUB))
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# You can flip these to "1" later when you want hard offline behavior by default
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

# Expose Nevise tokenizer path to your code (helpers.py reads this)
os.environ.setdefault("NEVISE_BERT_DIR", str(BERT_FA_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("init")


def _ensure_dirs():
    for p in [MODELS_ROOT, HF_ROOT, TORCH_HUB, NLTK_DATA, FW_MEDIUM_DIR.parent, BERT_FA_DIR.parent, NEVISE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _pip_install(package: str):
    log.info(f"pip install {package} …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


def _snapshot(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None):
    """
    Cache a HF repo to local_dir. If it already exists with files, we skip.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        _pip_install("huggingface_hub>=0.24.0")
        from huggingface_hub import snapshot_download

    if any(local_dir.rglob("*")):
        log.info(f"Already cached: {repo_id} -> {local_dir}")
        return

    log.info(f"Caching HF repo: {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    log.info(f"✅ Cached: {repo_id}")


def _cache_nltk():
    try:
        import nltk  # noqa
    except Exception:
        _pip_install("nltk")
        import nltk  # noqa

    import nltk
    log.info("Caching NLTK punkt …")
    NLTK_DATA.mkdir(parents=True, exist_ok=True)
    nltk.download("punkt", download_dir=str(NLTK_DATA))
    # newer NLTK versions sometimes need this table:
    try:
        nltk.download("punkt_tab", download_dir=str(NLTK_DATA))
    except Exception:
        pass
    log.info("✅ Cached NLTK punkt")



def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def is_silero_cached(torch_cache: Path) -> bool:
    # Look for a cached Torch Hub repo (has hubconf.py) directly under torch_cache
    for sub in torch_cache.glob("*"):
        if (sub / "hubconf.py").exists() and "silero" in sub.name.lower():
            return True
    return False

def cache_silero_vad(torch_hub_dir: Path, force: bool = False) -> None:
    """
    Pre-fetch Silero VAD via torch.hub so it can be used offline.
    This loads 'silero_vad' (the correct hub entry) which returns (model, utils).
    """
    import torch

    # Ensure hub dir is where you expect; PyTorch will use <TORCH_HOME>/hub/...
    os.environ.setdefault("TORCH_HOME", str(torch_hub_dir))
    torch.hub.set_dir(str(torch_hub_dir))

    logging.info("Caching Silero VAD via torch.hub (online)…")

    # NOTE:
    #   - model='silero_vad' is the only valid callable we need to load
    #   - trust_repo=True is required because hubconf executes repo code
    #   - source='github' makes it explicit (optional)
    #   - onnx=False keeps it in PyTorch (same as WhisperX default)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=bool(force),
        trust_repo=True,
        source="github",
        onnx=False,
    )

    # Sanity-check: the utils tuple should contain expected callables
    # required_utils = {"get_speech_timestamps", "read_audio", "VADIterator", "collect_chunks", "save_audio"}
    # available_utils = set(name for name in dir(utils) if not name.startswith("_"))
    # missing = required_utils - available_utils
    # if missing:
    #     raise RuntimeError(f"Silero VAD utils missing expected members: {sorted(missing)}")

    logging.info("✅ Silero VAD cached successfully (model + utils)")


def bootstrap_offline_assets():
    """
    One-time online bootstrap:
      1) ensure folders
      2) download/copy models locally:
         - Systran/faster-whisper-medium (full snapshot)
         - HooshvareLab/bert-fa-base-uncased (tokenizer files are enough)
      3) download Nevise assets (vocab & checkpoint) from Google Drive
      4) cache NLTK punkt
    """
    _ensure_dirs()

    # 2a) faster-whisper-medium (CTRANSLATE2 weights + tokenizer etc.)
    _snapshot(
        repo_id="Systran/faster-whisper-medium",
        local_dir=FW_MEDIUM_DIR,
        allow_patterns=None,  # keep full model so WhisperX can find everything
    )

    # 2b) Persian BERT tokenizer (only tokenizer artifacts to keep it medium)
    _snapshot(
        repo_id="HooshvareLab/bert-fa-base-uncased",
        local_dir=BERT_FA_DIR,
        allow_patterns=[
            "tokenizer.json",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ],
    )

    # 3) Nevise vocab & checkpoint (Google Drive)
    NEVISE_DIR.mkdir(parents=True, exist_ok=True)
    # IDs you provided:
    #   vocab.pkl  -> 1Ki5WGR4yxftDEjROQLf9Br8KHef95k1F
    #   model.pth.tar -> 1nKeMdDnxIJpOv-OeFj00UnhoChuaY5Ns
    _download_with_gdown("1Ki5WGR4yxftDEjROQLf9Br8KHef95k1F", NEVISE_CKPT)
    _download_with_gdown("1nKeMdDnxIJpOv-OeFj00UnhoChuaY5Ns", NEVISE_VOCAB)

    cache_silero_vad(TORCH_HUB, force=False)


    # 4) NLTK punkt
    _cache_nltk()

    log.info("🎉 Offline bootstrap complete. You can now run with HF_HUB_OFFLINE=1.")


# Run bootstrap if the flag is set
if BOOTSTRAP:
    # ensure we’re online for bootstrap; if not, this will just fail clearly
    log.info("BOOTSTRAP_OFFLINE=1 → caching models/assets for offline use …")
    try:
        bootstrap_offline_assets()
    except Exception as e:
        log.exception("Bootstrap failed: %s", e)
        raise SystemExit(2)

# Optional: after bootstrap, force offline by default (you can also set these in your shell)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["BOOTSTRAP_OFFLINE"] = "0"
