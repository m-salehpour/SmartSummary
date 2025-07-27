import time
import socket
import hashlib
from pathlib import Path
import logging as log

def _is_online(host="www.google.com", timeout=2.5) -> bool:
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _looks_like_html_error(path: Path) -> bool:
    try:
        head = path.read_bytes()[:2048].lower()
    except Exception:
        return True
    return (b"<html" in head) or (b"quota" in head and b"google drive" in head)

def _download_with_requests_gdrive(file_id: str, out_path: Path, timeout=30):
    """Fallback: download from Google Drive with confirm-token handling."""
    log.info(f"[_download_with_requests_gdrive] Downloading {file_id}")
    import requests
    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}
    r = session.get(URL, params=params, stream=True, timeout=timeout)
    r.raise_for_status()

    # If a confirm token is required, it usually appears as a cookie
    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        params["confirm"] = token
        r = session.get(URL, params=params, stream=True, timeout=timeout)
        r.raise_for_status()

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(out_path)

def _download_with_gdown(file_id: str, out_path: Path, sha256: str | None = None, max_retries: int = 3):
    """
    Robust Google Drive download using gdown with validation & retries.
    - file_id: the Drive file ID (the long hash), not the full URL.
    - out_path: target file path.
    - sha256: optional checksum to verify integrity.
    """
    log.info(f"[_download_with_gdown] Downloading {file_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        if sha256:
            if _sha256(out_path) == sha256:
                log.info(f"Already present (sha256 OK): {out_path.name}")
                return
            else:
                log.warning(f"Checksum mismatch, re-downloading: {out_path.name}")
                out_path.unlink(missing_ok=True)
        else:
            log.info(f"Already present: {out_path.name}")
            return

    if not _is_online():
        raise RuntimeError(f"Offline: cannot download {out_path.name}. Connect once to cache it.")

    # Try gdown first
    try:
        import gdown  # type: ignore
    except Exception:
        # Don't try to install here if you might be offline; fail clearly instead
        raise RuntimeError("gdown is not installed and cannot be auto-installed while offline.")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Downloading {out_path.name} via gdown (attempt {attempt}/{max_retries}) …")
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.unlink(missing_ok=True)

            # Use the id= form (more robust than manually building the URL)
            ret = gdown.download(id=file_id, output=str(tmp), quiet=False, fuzzy=True, use_cookies=False)
            if not ret or not tmp.exists() or tmp.stat().st_size == 0 or _looks_like_html_error(tmp):
                raise RuntimeError("gdown returned empty/HTML content (quota/private/bad id?)")

            if sha256 and _sha256(tmp) != sha256:
                raise RuntimeError("sha256 mismatch after download")

            tmp.replace(out_path)
            log.info(f"✅ Downloaded: {out_path.name}")
            return
        except Exception as e:
            last_err = e
            log.warning(f"gdown failed: {e}")
            time.sleep(2 * attempt)

    # Fallback to requests-based downloader (handles confirm token)
    log.info("Falling back to requests-based Google Drive downloader …")
    try:
        _download_with_requests_gdrive(file_id, out_path)
        if out_path.stat().st_size == 0 or _looks_like_html_error(out_path):
            raise RuntimeError("requests fallback produced empty/HTML file.")
        if sha256 and _sha256(out_path) != sha256:
            raise RuntimeError("sha256 mismatch after requests fallback download")
        log.info(f"✅ Downloaded: {out_path.name}")
        return
    except Exception as e:
        raise RuntimeError(f"Failed to download {out_path.name}: {e} (last gdown err: {last_err})") from e
