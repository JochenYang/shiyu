"""
Model downloader - downloads SenseVoice model on first launch.
"""
import os
import shutil
import tarfile
import threading
import requests
from pathlib import Path

# Global state for download progress (thread-safe singleton)
_download_state = {"status": "idle", "progress": 0, "error": ""}

# Model files needed
MODEL_FILES = [
    "model.onnx",
    "tokens.json",
    "config.yaml",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
]

# Download sources
# Small config files are on ModelScope (iic/SenseVoiceSmall) — fastest in CN
MODELSCOPE_BASE = "https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master"

# Model ONNX file: OpenASR/sensevoice-small-onnx on HF Mirror (accessible from CN)
HF_MIRROR_BASE = "https://hf-mirror.com/OpenASR/sensevoice-small-onnx/resolve/main"

# Fallback for model.onnx: extract from GitHub release tar.bz2
GITHUB_RELEASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
)


def get_shiyu_data_dir() -> Path:
    """Get the Shiyu user data directory (~/.shiyu)."""
    return Path.home() / ".shiyu"


def get_model_dir() -> Path:
    """Get the model directory (~/.shiyu/models/sensevoice-small)."""
    return get_shiyu_data_dir() / "models" / "sensevoice-small"


def model_exists() -> bool:
    """Check if ALL model files exist and are valid (>1KB each)."""
    model_dir = get_model_dir()
    if not model_dir.exists():
        return False
    for fname in MODEL_FILES:
        fpath = model_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 1024:
            return False
    # Validate model.onnx is actually the large model (>10MB)
    onnx_path = model_dir / "model.onnx"
    if onnx_path.stat().st_size < 10 * 1024 * 1024:
        return False
    return True


def _try_migrate_from_bundle() -> bool:
    """Try to find model files from the bundled app resources and copy to user dir.

    Checks common install paths:
    - Windows: <exe_dir>/resources/models/sensevoice-small/
    - macOS: <app>/Contents/Resources/resources/models/sensevoice-small/
    - dev: <project>/models/sensevoice-small/
    """
    # Get the directory where this script lives
    script_dir = Path(__file__).resolve().parent.parent.parent.parent  # backend/

    # Possible source directories to check
    candidates = [
        # Development mode: <project>/models/sensevoice-small/
        script_dir.parent / "models" / "sensevoice-small",
        # Production Windows/macOS: <exe>/resources/models/sensevoice-small/
        # (Tauri resources are relative to executable)
        Path(os.getcwd()).parent / "models" / "sensevoice-small",
        # Also try the tauri resource path
        Path(__file__).resolve().parent.parent.parent.parent.parent / "resources" / "models" / "sensevoice-small",
    ]

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    for src in candidates:
        if not src.exists():
            continue

        # ALL required files must exist in the bundle source
        required_files_ok = True
        for fname in MODEL_FILES:
            src_file = src / fname
            if not src_file.exists() or src_file.stat().st_size < 1024:
                required_files_ok = False
                break
        if not required_files_ok:
            continue

        # Valid model.onnx must be >10MB (not a stub)
        onnx_src = src / "model.onnx"
        if onnx_src.stat().st_size < 10 * 1024 * 1024:
            continue

        print(f"[ModelDownloader] Found bundled model at {src}, migrating...")
        for fname in MODEL_FILES:
            src_file = src / fname
            dst_file = model_dir / fname
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"[ModelDownloader] Failed to copy {fname}: {e}")
                return False
        print(f"[ModelDownloader] Migration complete!")
        return True

    return False


def _cleanup_wrong_locations():
    """Move model.onnx from wrong directory if misplaced one level up."""
    wrong = get_shiyu_data_dir() / "models" / "model.onnx"
    correct = get_model_dir() / "model.onnx"
    if wrong.exists() and wrong.stat().st_size > 10 * 1024 * 1024:
        if not correct.exists():
            try:
                wrong.rename(correct)
                print(f"[ModelDownloader] Moved model.onnx from {wrong} to {correct}")
                return
            except Exception as e:
                print(f"[ModelDownloader] Failed to move model.onnx: {e}")
        # Remove the wrong-located file if correct one already exists or move failed
        try:
            wrong.unlink(missing_ok=True)
        except Exception as e:
            print(f"[ModelDownloader] Failed to remove wrong-located model.onnx: {e}")


def _extract_onnx_from_github_release(dst: Path) -> bool:
    """Download and extract model.int8.onnx from GitHub release tar.bz2.

    This is a fallback when HF Mirror is unavailable.
    """
    try:
        print(f"[ModelDownloader] Downloading from GitHub release (163MB)...")
        resp = requests.get(GITHUB_RELEASE_URL, stream=True, timeout=120)
        resp.raise_for_status()

        # Save to temporary tar.bz2
        tmp_bz2 = dst.with_suffix(".tar.bz2")
        with open(tmp_bz2, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Extract model.int8.onnx from the archive
        with tarfile.open(tmp_bz2, "r:bz2") as tar:
            found = False
            for member in tar.getmembers():
                if member.name.endswith("model.int8.onnx"):
                    print(f"[ModelDownloader] Extracting {member.name}...")
                    tmp_onnx = dst.with_suffix(".int8.onnx.tmp")
                    with open(tmp_onnx, "wb") as f:
                        f.write(tar.extractfile(member).read())
                    tmp_onnx.rename(dst)  # rename to model.onnx
                    found = True
                    break
            if not found:
                raise RuntimeError("model.int8.onnx not found in GitHub release archive")

        # Cleanup
        tmp_bz2.unlink(missing_ok=True)
        print(f"[ModelDownloader] Successfully extracted model to {dst}")
        return True

    except Exception as e:
        print(f"[ModelDownloader] GitHub release extraction failed: {e}")
        # Cleanup partial files
        tmp_bz2 = dst.with_suffix(".tar.bz2")
        if tmp_bz2.exists():
            tmp_bz2.unlink(missing_ok=True)
        return False


def _download_file(url: str, dst: Path, progress_weight: float = 1.0) -> bool:
    """Download a single file with streaming and progress tracking.

    Args:
        url: Download URL
        dst: Destination file path
        progress_weight: Weight of this download in overall progress (0.0-1.0)
    """
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        tmp = dst.with_suffix(f"{dst.suffix}.tmp")
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update global progress during download of large files
                    if total > 1024 * 1024:  # Only for files > 1MB
                        pct = downloaded / total if total > 0 else 0
                        _download_state["progress"] = int(pct * progress_weight * 100)
        tmp.rename(dst)
        return True
    except Exception as e:
        print(f"[ModelDownloader] Failed to download {url}: {e}")
        return False


def _download_worker():
    """Background worker that downloads model files."""
    global _download_state

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # First, clean up any model.onnx in wrong location
        _cleanup_wrong_locations()

        # Progress weights: model.onnx is ~228MB, small files are each <1MB
        # model.onnx gets 85% of progress bar, small files share 15%
        _download_state["progress"] = 0
        _download_state["status"] = "downloading"

        for fname in MODEL_FILES:
            dst = model_dir / fname

            # Skip if already exists (e.g. migrated from bundle)
            if dst.exists() and dst.stat().st_size > 1024:
                print(f"[ModelDownloader] {fname} already exists, skipping")
                continue

            print(f"[ModelDownloader] Downloading {fname}...")
            success = False

            if fname == "model.onnx":
                # model.onnx: 85% weight, primary from HF Mirror, fallback GitHub
                onnx_url = f"{HF_MIRROR_BASE}/model.int8.onnx"
                print(f"[ModelDownloader]   from HF Mirror: {onnx_url}")
                success = _download_file(onnx_url, dst, progress_weight=0.85)

                if not success:
                    print(f"[ModelDownloader]   fallback to GitHub release...")
                    success = _extract_onnx_from_github_release(dst)

                if success:
                    _download_state["progress"] = 85
            else:
                # Small files: 3.75% each, primary from ModelScope
                ms_url = f"{MODELSCOPE_BASE}/{fname}"
                print(f"[ModelDownloader]   from ModelScope: {ms_url}")
                success = _download_file(ms_url, dst, progress_weight=0.0375)

                # Fallback: try HF Mirror
                if not success:
                    hf_url = f"{HF_MIRROR_BASE}/{fname}"
                    print(f"[ModelDownloader]   fallback to HF Mirror: {hf_url}")
                    success = _download_file(hf_url, dst, progress_weight=0.0375)

                if success:
                    _download_state["progress"] = min(_download_state["progress"] + 3, 99)

            if not success:
                raise RuntimeError(f"Failed to download {fname} from all sources")

            print(f"[ModelDownloader] {fname} downloaded, overall progress: {_download_state['progress']}%")

        _download_state["status"] = "ready"
        _download_state["progress"] = 100
        _download_state["error"] = ""
        print(f"[ModelDownloader] All files downloaded successfully!")

    except Exception as e:
        _download_state["status"] = "error"
        _download_state["error"] = str(e)
        print(f"[ModelDownloader] Download failed: {e}")


def ensure_model():
    """Ensure model exists. Returns immediately, downloads in background if needed.

    Returns:
        dict with model status
    """
    global _download_state

    # First, clean up any model.onnx in wrong location before checking
    _cleanup_wrong_locations()

    if model_exists():
        _download_state = {"status": "ready", "progress": 100, "error": ""}
        return _download_state

    # Don't restart if already downloading
    if _download_state.get("status") == "downloading":
        return _download_state

    # Try migration from bundled resources
    if _try_migrate_from_bundle():
        if model_exists():
            _download_state = {"status": "ready", "progress": 100, "error": ""}
            return _download_state

    # Start background download
    _download_state = {"status": "downloading", "progress": 0, "error": ""}
    thread = threading.Thread(target=_download_worker, daemon=True)
    thread.start()

    return _download_state


def get_download_status() -> dict:
    """Get current download status."""
    if model_exists():
        return {"status": "ready", "progress": 100, "error": ""}
    return dict(_download_state)
