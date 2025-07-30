# config.py
from pathlib import Path
from init_env import NEVISE_CKPT, NEVISE_VOCAB

from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cpu"
BATCH_SIZE = 16
DEFAULT_MODEL = "medium"
QUANT_TYPE = "float32"   # or int8
OLLAMA_MODEL_TAG  = "gemma3:4b-it-q4_K_M"      # tweak freely
OLLAMA_URL = "http://localhost:11434"
JSON_ASR_OUTPUT_DIR = "asr_outputs"
MODEL_DIR    = Path("persian_normalize/Nevise/model")
NEVISE_CKPT  = NEVISE_CKPT
NEVISE_VOCAB = NEVISE_VOCAB
FALLBACK_POLICY_FULL = "full"         #   "full"     -> return whole document (after cleanup)
FALLBACK_POLICY_DIALOGUE = "dialogue" #   "dialogue" -> strip "Speaker: " labels, then return
FALLBACK_POLICY_FIRST = "first"       # "first"    -> first non-empty block (>= min_chars)
FALLBACK_POLICY_NONE = None           # "none"     -> raise ValueError (strict mode)






