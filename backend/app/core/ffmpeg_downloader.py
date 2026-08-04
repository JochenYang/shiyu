"""
FFmpeg auto-downloader — downloads static ffmpeg binary on first use.

Follows the same pattern as ``model_downloader.py``:
  - Downloads to ``~/.shiyu/bin/``
  - Runs in background thread
  - Provides status for frontend progress display

Supported platforms:
  - Windows: BtbN GPL zip (contains ffmpeg.exe + ffprobe.exe)
  - macOS:   evermeet.cx static ffmpeg zip
  - Linux:   johnvansickle.com static ffmpeg tar.xz
"""
import os
import shutil
import stat
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

import requests

# ── 平台感知 ──────────────────────────────────────────────────────────

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

EXE_NAME = "ffmpeg.exe" if IS_WIN else "ffmpeg"

# ── 下载源 ────────────────────────────────────────────────────────────

FFMPEG_SOURCES: dict = {}

# Windows: BtbN 静态构建 zip
if IS_WIN:
    FFMPEG_SOURCES = {
        "primary": (
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
            "ffmpeg-master-latest-win64-gpl.zip"
        ),
        # GitHub proxy mirror (accessible from CN)
        "mirror": (
            "https://ghproxy.net/https://github.com/BtbN/FFmpeg-Builds/"
            "releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        ),
        # 从 zip 中提取 ffmpeg.exe 的路径
        "bin_zip_path": "ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe",
    }
elif IS_MAC:
    FFMPEG_SOURCES = {
        "primary": (
            "https://evermeet.cx/ffmpeg/ffmpeg-7.1.zip"
        ),
        "bin_zip_path": "ffmpeg",  # zip root 下直接是 ffmpeg
    }
elif IS_LINUX:
    FFMPEG_SOURCES = {
        "primary": (
            "https://johnvansickle.com/ffmpeg/releases/"
            "ffmpeg-release-amd64-static.tar.xz"
        ),
        # tar.xz 中的路径：ffmpeg-7.1-amd64-static/ffmpeg
        "bin_tar_path": None,  # 在下载时动态确定
    }

# ── 全局下载状态 ──────────────────────────────────────────────────────

_download_state: dict = {"status": "idle", "progress": 0, "error": ""}
_lock = threading.Lock()


# ── 路径 ──────────────────────────────────────────────────────────────


def get_shiyu_bin_dir() -> Path:
    """Get `~/.shiyu/bin/` directory."""
    p = Path.home() / ".shiyu" / "bin"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_expected_ffmpeg_path() -> Path:
    """Full path to the expected ffmpeg binary under `~/.shiyu/bin/`."""
    return get_shiyu_bin_dir() / EXE_NAME


# ── 状态查询 ─────────────────────────────────────────────────────────


def get_download_status() -> dict:
    """Return current download status."""
    with _lock:
        if get_expected_ffmpeg_path().is_file():
            _verify_ffmpeg()  # 检查是否真的是可用的二进制
            return {"status": "ready", "progress": 100, "error": ""}
        return dict(_download_state)


def _verify_ffmpeg() -> bool:
    """Quick verify that ffmpeg binary works."""
    exe = get_expected_ffmpeg_path()
    if not exe.is_file() or exe.stat().st_size < 1024 * 1024:  # need at least 1MB
        return False
    return True


# ── 下载 + 解压 ──────────────────────────────────────────────────────


def _download_file(url: str, dst: Path) -> bool:
    """Download a file with streaming progress."""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        with _lock:
                            _download_state["progress"] = int(downloaded / total * 90)
        tmp.rename(dst)
        return True
    except Exception as e:
        print(f"[FFmpegDownloader] Download failed: {e}")
        return False


def _extract_zip(zip_path: Path, inner_path: str, dst: Path) -> bool:
    """Extract a single file from a zip archive."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the actual path in the zip
            normalized_target = inner_path.replace("\\", "/")
            matched = None
            for name in zf.namelist():
                if name.endswith(normalized_target) or name == normalized_target:
                    matched = name
                    break
            if matched is None:
                print(f"[FFmpegDownloader] '{inner_path}' not found in zip")
                return False
            with zf.open(matched) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
        _set_executable(dst)
        return True
    except Exception as e:
        print(f"[FFmpegDownloader] Zip extraction failed: {e}")
        return False


def _extract_tar_xz(tar_path: Path, dst: Path) -> bool:
    """Extract ffmpeg from a .tar.xz archive (Linux)."""
    try:
        with tarfile.open(tar_path, "r:xz") as tar:
            # Find ffmpeg inside the archive
            for member in tar.getmembers():
                if member.name.endswith("/ffmpeg") and not member.isdir():
                    with tar.extractfile(member) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                    _set_executable(dst)
                    return True
        print(f"[FFmpegDownloader] ffmpeg not found in tar.xz")
        return False
    except Exception as e:
        print(f"[FFmpegDownloader] tar.xz extraction failed: {e}")
        return False


def _set_executable(path: Path):
    """Make file executable (no-op on Windows)."""
    if not IS_WIN:
        try:
            st = path.stat()
            path.chmod(st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass


# ── 后台下载线程 ──────────────────────────────────────────────────────


def _download_worker():
    """Background worker that downloads ffmpeg."""
    global _download_state
    try:
        with _lock:
            _download_state["status"] = "downloading"
            _download_state["progress"] = 0
            _download_state["error"] = ""

        bin_dir = get_shiyu_bin_dir()
        archive_name = "ffmpeg-master-latest-win64-gpl.zip"
        archive_path = bin_dir / archive_name

        # Step 1: download archive (try primary, then mirror)
        urls_to_try = [FFMPEG_SOURCES["primary"]]
        if "mirror" in FFMPEG_SOURCES:
            urls_to_try.append(FFMPEG_SOURCES["mirror"])

        ok = False
        for url in urls_to_try:
            print(f"[FFmpegDownloader] Downloading {archive_name}...")
            ok = _download_file(url, archive_path)
            if ok:
                break
            print(f"[FFmpegDownloader] Failed, trying next source...")

        if not ok:
            raise RuntimeError("All download sources failed")

        with _lock:
            _download_state["progress"] = 90

        # Step 2: extract ffmpeg binary
        dst = bin_dir / EXE_NAME
        if IS_MAC or IS_WIN:
            inner = FFMPEG_SOURCES["bin_zip_path"]
            ok = _extract_zip(archive_path, inner, dst)
        else:
            ok = _extract_tar_xz(archive_path, dst)

        if not ok:
            raise RuntimeError("Extraction failed")

        # Step 3: cleanup archive
        archive_path.unlink(missing_ok=True)

        # Step 4: verify
        if not dst.is_file() or dst.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"Extracted binary invalid: {dst.stat().st_size} bytes")

        print(f"[FFmpegDownloader] ffmpeg ready at {dst}")
        with _lock:
            _download_state["status"] = "ready"
            _download_state["progress"] = 100
            _download_state["error"] = ""

    except Exception as e:
        print(f"[FFmpegDownloader] Failed: {e}")
        with _lock:
            _download_state["status"] = "error"
            _download_state["error"] = str(e)


# ── 主入口 ────────────────────────────────────────────────────────────


def ensure_ffmpeg() -> dict:
    """Ensure ffmpeg is available. Downloads in background if needed.

    Returns:
        dict with status.
    """
    if _verify_ffmpeg():
        with _lock:
            _download_state = {"status": "ready", "progress": 100, "error": ""}
        return {"status": "ready", "progress": 100, "error": ""}

    with _lock:
        if _download_state.get("status") == "downloading":
            return dict(_download_state)
        _download_state = {"status": "downloading", "progress": 0, "error": ""}

    thread = threading.Thread(target=_download_worker, daemon=True)
    thread.start()
    return {"status": "downloading", "progress": 0, "error": ""}
