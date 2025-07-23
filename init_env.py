# from dotenv import load_dotenv
# import os
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# ─── Prepend Homebrew to PATH ──────────────────────────────────────────────────
# so that subprocesses can find /opt/homebrew/bin/ffmpeg, etc.
import os
homebrew_bin = "/opt/homebrew/bin"
os.environ["PATH"] = f"{homebrew_bin}:{os.environ.get('PATH', '')}"
