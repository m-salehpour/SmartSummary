# SmartSummary (offline‑ready ASR + normalization)

This repository provides a CPU‑only speech‑to‑text pipeline (WhisperX / faster‑whisper) with language‑aware normalization (Persian, Hebrew, English) and optional LLM post‑cleaning. It supports **fully offline** execution on modest hardware.

---

## Quick start

### Requirements
- Python 3.11
- ffmpeg available on PATH
(e.g., macOS/Homebrew on Apple Silicon: add /opt/homebrew/bin to PATH)

   - It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:
    ```bash
    # on Ubuntu or Debian
    sudo apt update && sudo apt install ffmpeg
    
    # on Arch Linux
    sudo pacman -S ffmpeg
    
    # on MacOS using Homebrew (https://brew.sh/)
    brew install ffmpeg
    
    # on Windows using Chocolatey (https://chocolatey.org/)
    choco install ffmpeg
    
    # on Windows using Scoop (https://scoop.sh/)
    scoop install ffmpeg
    ```

```bash
# 0) Create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 1) Install dependencies
pip install -U pip
pip install -r requirements.txt

# 2) One‑time bootstrap (requires internet ONCE)
export BOOTSTRAP_OFFLINE=1
python -m init

# 3) (Optional) start your local LLM if you plan to use LLM cleaning
# e.g., Ollama with gemma3:4b-it-q4_K_M
#   ollama pull gemma3:4b-it-q4_K_M
#   ollama serve

# 4) Run a transcription experiment (now works offline)
python -m tools.transcribe_x2 \
  --audio "../Data/Training/English/Churchill/english_firstsourcecommons_13_churchill_64kb.mp3" \
  --ref   "../Data/Training/English/Churchill/english_firstsourcecommons_13_churchill_transcript-english_translation_hebrew.docx"
```

After step (2) you can go offline and run step (4) repeatedly.

---

## Why this bootstrap?

Some libraries try to download models at runtime (Hugging Face Hub, torch.hub, NLTK, etc.). The bootstrap step **pre‑caches** everything in `./models/` so that later runs don’t touch the network.

---

## What gets cached

* **Whisper (faster‑whisper)**

  * Default path: `models/hf/Systran/faster-whisper-small` (or `medium` if you choose)
* **Persian BERT tokenizer** used by Nevise spell‑corrector

  * Path: `models/hf/HooshvareLab/bert-fa-base-uncased`
* **Nevise resources** (context‑aware Persian correction)

  * `models/persian_nevise/vocab.pkl`
  * `models/persian_nevise/model.pth.tar`
* **Silero VAD** (via torch.hub)

  * Torch cache dir: `models/torch_home`
  * Torch hub dir:  `models/torch_hub`
* **NLTK punkt data** (sentence tokenization)

  * `models/nltk_data`

> The bootstrap uses environment variables so runtime components (WhisperX, Nevise, etc.) find these caches without trying to download.

---

## One‑time bootstrap (online)

```bash
export BOOTSTRAP_OFFLINE=1
python -m init
```

This will:

1. Create `models/` tree if missing
2. Snapshot/download:

   * `Systran/faster-whisper-small` (or your configured model)
   * `HooshvareLab/bert-fa-base-uncased` (tokenizer files)
   * Nevise vocab & checkpoint files (from pre‑defined URLs)
   * NLTK `punkt` & `punkt_tab`
   * Pre‑warm **Silero VAD** into `models/torch_hub`
3. Write a small cache marker so later runs can detect the offline setup.

When it finishes you’ll see:

```
🎉 Offline bootstrap complete. You can now run with HF_HUB_OFFLINE=1.
```

---

## Fully offline runs

Once bootstrapped, you can force offline mode:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/models/hf_cache"   # optional, if you want a dedicated HF cache
export NLTK_DATA="$PWD/models/nltk_data"
export TORCH_HOME="$PWD/models/torch_home"
export TORCH_HUB="$PWD/models/torch_hub"

# Run
python -m tools.transcribe_x2 --audio <path> --ref <path>
```

The code points faster‑whisper to the **local** copy via `download_root=…` and sets all “\*\_OFFLINE” flags so no network calls are made.

---

## Directory layout

```
SmartSummary/
├─ models/
│  ├─ hf/
│  │  ├─ Systran/faster-whisper-small/        # or faster-whisper-medium, etc.
│  │  └─ HooshvareLab/bert-fa-base-uncased/
│  ├─ hf_cache/                               # (optional) HF cache dir
│  ├─ persian_nevise/
│  │  ├─ vocab.pkl
│  │  └─ model.pth.tar
│  ├─ torch_home/
│  ├─ torch_hub/                              # silero‑vad code cache
│  └─ nltk_data/
├─ tools/
│  ├─ asr.py
│  ├─ normalizers.py
│  ├─ persian_normalize/
│  │  ├─ Nevise/
│  │  └─ nevis.py
│  ├─ hebrew_normalize/
│  │  └─ …
│  ├─ english_normalize/
│  │  └─ …
│  ├─ profiler.py
│  ├─ compare.py
│  └─ transcribe_x2.py
├─ init.py
├─ config.py
└─ README.md
```

---

## Configuration knobs

### `config.py`

* `DEVICE = "cpu"`
* `BATCH_SIZE = 16`
* `DEFAULT_MODEL = "small"`  # or "medium"
* `QUANT_TYPE = "float32"`   # faster‑whisper compute type
* `JSON_ASR_OUTPUT_DIR = "asr_outputs"`
* `MODEL_DIR` / `NEVISE_CKPT` / `NEVISE_VOCAB`  # local Nevise files
* `OLLAMA_URL` and `OLLAMA_MODEL_TAG` (if you use LLM cleaning)

### Environment variables

* `BOOTSTRAP_OFFLINE=1` → triggers downloads into `./models/`
* `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` → disable network at runtime
* `HF_HOME` → where HF caches go (optional)
* `NLTK_DATA`, `TORCH_HOME`, `TORCH_HUB` → point libraries to local caches

---

## Troubleshooting

### 1) "Cannot find an appropriate cached snapshot folder…"

* You likely didn’t run `python -m init` with internet once.
* Or the model name changed. Check `config.DEFAULT_MODEL` and that the corresponding directory exists under `models/hf/Systran/`.

### 2) Silero VAD tries to download

* Ensure `TORCH_HUB` points to `models/torch_hub` and that the bootstrap pre‑warmed it.
* If needed, re‑run only the VAD warm‑up:

  ```bash
  python - <<'PY'
  import os, torch
  os.environ['TORCH_HUB'] = os.path.abspath('models/torch_hub')
  os.environ['TORCH_HOME'] = os.path.abspath('models/torch_home')
  torch.hub.set_dir(os.environ['TORCH_HUB'])
  torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True, trust_repo=True)
  print('Silero cached at:', torch.hub.get_dir())
  PY
  ```

### 3) SSL / cert issues

* If your Python has strict SSL and you’re on a corporate proxy, you can set in code *only for bootstrap*:

  ```python
  import ssl; ssl._create_default_https_context = ssl._create_unverified_context
  ```

### 4) NLTK `punkt` offline

* Ensure `NLTK_DATA` env var points to `models/nltk_data` **before** importing NLTK.

### 5) Logging verbosity

* Default logging is INFO. You can override via:

  ```bash
  export PYTHONWARNINGS=ignore
  export LOGLEVEL=INFO   # or DEBUG
  ```

### 6) Re‑run bootstrap safely

* You can re‑run `BOOTSTRAP_OFFLINE=1 python -m init` any time; it is idempotent and will skip files that already exist.

---

## Notes / tips

* **Disk usage**: Medium Whisper ≈ \~500–800 MB; tokenizer ≈ \~400 MB; Nevise ≈ \~1.5 GB; Silero hub \~few MB.
* **CPU‑only**: Use `compute_type="float32"` (default). `int8` lowers RAM but may reduce accuracy.
* **Determinism**: For comparable WER runs, keep `vad_options` and `asr_options` constant.
* **LLM cleaning is optional**: If Ollama is not running, the pipeline falls back to non‑LLM normalization.

