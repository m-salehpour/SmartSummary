# config.py
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cpu"
BATCH_SIZE = 16
DEFAULT_MODEL = "medium"
LARGE_V2_MODEL = "large-v2"
QUANT_TYPE_FLOAT_32 = "float32"   # or int8
QUANT_TYPE_INT_8 = "int8"   # or int8
OLLAMA_MODEL_TAG  = "gemma3:4b-it-q4_K_M"      # tweak freely
OLLAMA_URL = "http://localhost:11434"
JSON_ASR_OUTPUT_DIR = "asr_outputs"
MODEL_DIR    = Path("persian_normalize/Nevise/model")

BOOTSTRAP = os.getenv("BOOTSTRAP_OFFLINE", "1").lower() in {"1", "true", "yes"}
ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"              # all cached models live here
HF_ROOT      = MODELS_ROOT / "hf"          # huggingface snapshots
TORCH_HUB    = MODELS_ROOT / "torch_hub"   # torch hub cache (e.g., silero)
NLTK_DATA    = MODELS_ROOT / "nltk_data"   # punkt, etc.
FW_MEDIUM_DIR = HF_ROOT / "Systran" / "faster-whisper-medium"
FW_LARGE_V2_DIR = HF_ROOT / "Systran" / "faster-whisper-large-v2"
BERT_FA_DIR  = HF_ROOT / "HooshvareLab" / "bert-fa-base-uncased"
NEVISE_DIR   = ROOT / "tools" / "persian_normalize" / "Nevise" / "model"

NEVISE_VOCAB = NEVISE_DIR / "vocab.pkl"
NEVISE_CKPT  = NEVISE_DIR / "model.pth.tar"
FALLBACK_POLICY_FULL = "full"         #   "full"     -> return whole document (after cleanup)
FALLBACK_POLICY_DIALOGUE = "dialogue" #   "dialogue" -> strip "Speaker: " labels, then return
FALLBACK_POLICY_FIRST = "first"       # "first"    -> first non-empty block (>= min_chars)
FALLBACK_POLICY_NONE = None           # "none"     -> raise ValueError (strict mode)
YOUTUBE_CACHE_PATH = Path("/Users/pouya/PycharmProjects/SmartSummary/Data/Training/Farsi/youtube_files")
YOUTUBE_URLS = [
    "https://youtu.be/FtBfFN_AGTA?si=AYS-56wUinfQF-hE",
    "https://youtu.be/6j4QBircNnU?si=ALyFAn648z5QZt7T",
    "https://youtube.com/shorts/bRg54hrWbcE?si=8O-u7_KDOKGBsAVI",
    "https://youtube.com/shorts/1CCd-PToFFc?si=dcnpvsz3WukCrQlV",
    "https://youtube.com/shorts/ucSbHCiq-DU?si=dPaSTx98Z-BBUrZF",
    "https://youtu.be/K1k-vthv86Y?si=6iozFpO1xtBumHEW",
    "https://youtu.be/FFWXZmq1MTU?si=U82c_0AgHi4rm5-l",
    "https://youtu.be/EMAoQuv_6OU?si=HC_oVXNyJG-AxT0n",
]

