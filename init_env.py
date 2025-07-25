# from dotenv import load_dotenv
# import os
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")


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
