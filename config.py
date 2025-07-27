# config.py
from pathlib import Path
from init_env import NEVISE_CKPT, NEVISE_VOCAB

DEVICE = "cpu"
BATCH_SIZE = 16
DEFAULT_MODEL = "small"
QUANT_TYPE = "float32"   # or int8
OLLAMA_MODEL_TAG  = "gemma3:4b-it-q4_K_M"      # tweak freely
OLLAMA_URL = "http://localhost:11434"
JSON_ASR_OUTPUT_DIR = "asr_outputs"
MODEL_DIR    = Path("persian_normalize/Nevise/model")
NEVISE_CKPT  = NEVISE_CKPT
NEVISE_VOCAB = NEVISE_VOCAB