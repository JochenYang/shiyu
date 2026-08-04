"""
Silero-VAD ONNX 检测器 — 纯 ONNX Runtime，无需 PyTorch。

提供两个核心功能：
  1. ``detect_speech(waveform, sr)`` → 语音活跃区间列表
  2. ``get_pause_boundaries(waveform, sr)`` → 用于断句的停顿时间戳

模型自动下载至 ``~/.shiyu/models/silero-vad/``，约 20MB。
"""
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime
import requests

# ── 模型信息 ──────────────────────────────────────────────────────────

MODEL_RELATIVE_PATH = "models/silero-vad/silero_vad.onnx"

# PyPI wheel 源（国内可访问，~9MB）
PYPI_WHEEL_URL = (
    "https://pypi.org/packages/py3/s/silero-vad/"
    "silero_vad-6.2.1-py3-none-any.whl"
)

# 最后兜底：直接下载 URL（大多数已失效或不可用）
SOURCES = [
    "https://hf-mirror.com/snakers4/silero-vad/resolve/master/files/silero_vad.onnx",
    "https://huggingface.co/snakers4/silero-vad/resolve/master/files/silero_vad.onnx",
]

# Silero-VAD v5 参数
SR = 16000                       # 模型固定采样率
WINDOW_SAMPLES = 512             # 每帧采样数 (32ms @ 16kHz)
HOP_SAMPLES = 160                # 帧移 (10ms @ 16kHz)

# ── 全局状态 ──────────────────────────────────────────────────────────

_download_state: dict = {"status": "idle", "progress": 0, "error": ""}
_lock = threading.Lock()
_detector: Optional["VADDetector"] = None

# ── 路径 ──────────────────────────────────────────────────────────────


def get_model_path() -> Path:
    """Full path to the Silero-VAD ONNX model file."""
    return Path.home() / ".shiyu" / MODEL_RELATIVE_PATH


def get_download_status() -> dict:
    """Return download status."""
    with _lock:
        if get_model_path().is_file() and get_model_path().stat().st_size > 1024 * 1024:
            return {"status": "ready", "progress": 100, "error": ""}
        return dict(_download_state)


# ── 模型获取 ─────────────────────────────────────────────────────────


def _copy_from_pip() -> bool:
    """Copy silero_vad.onnx from silero-vad pip package if already installed."""
    try:
        import silero_vad
        import shutil
        pkg_dir = os.path.dirname(silero_vad.__file__)
        src = os.path.join(pkg_dir, "data", "silero_vad.onnx")
        if os.path.isfile(src):
            dst = get_model_path()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[VAD] Copied from pip package: {src}")
            return True
    except Exception:
        pass
    return False


def _install_via_pip() -> bool:
    """Install silero-vad with ``--no-deps`` via pip (no torch pulled in)."""
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "silero-vad"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return _copy_from_pip()
        print(f"[VAD] pip install failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"[VAD] pip install error: {e}")
    return False


def _extract_from_wheel() -> bool:
    """Download PyPI wheel directly and extract silero_vad.onnx from it.

    Uses ``urllib.request`` (stdlib) for reliable streaming — ``requests``
    streaming can hang on large payloads in some environments.
    """
    import zipfile, io, urllib.request
    dst = get_model_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[VAD] Downloading wheel from PyPI (~9MB)...")
        resp = urllib.request.urlopen(PYPI_WHEEL_URL, timeout=120)
        data = resp.read()
        print(f"[VAD] Downloaded {len(data) // 1024} KB")
        content = io.BytesIO(data)
        with zipfile.ZipFile(content) as zf:
            for name in zf.namelist():
                if name.endswith("silero_vad.onnx"):
                    with zf.open(name) as src, open(dst, "wb") as f:
                        f.write(src.read())
                    print(f"[VAD] Extracted {name} from wheel ({dst.stat().st_size // 1024} KB)")
                    return True
        print(f"[VAD] silero_vad.onnx not found in wheel")
    except Exception as e:
        print(f"[VAD] Wheel extraction failed: {e}")
    return False


def _download_model():
    """Acquire Silero-VAD ONNX model via multi-tier strategy.

    Tier 1 — already installed via ``silero-vad`` pip package (no download).
    Tier 2 — ``pip install --no-deps silero-vad`` (skip torch).
    Tier 3 — direct PyPI wheel download → extract ONNX.
    Tier 4 — static URL mirrors (last resort, most are broken).
    """
    global _download_state
    dst = get_model_path()
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Tier 1: already installed
    if _copy_from_pip():
        _mark_ready(dst)
        return

    # Tier 2: pip install --no-deps (9MB, no torch)
    if _install_via_pip():
        _mark_ready(dst)
        return

    # Tier 3: download wheel → extract ONNX directly
    if _extract_from_wheel():
        _mark_ready(dst)
        return

    # Tier 4: static URL mirrors (most are dead, kept for backward compat)
    for url in SOURCES:
        try:
            print(f"[VAD] Downloading from {url}")
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            with _lock:
                                _download_state["progress"] = int(downloaded / total * 100)
            tmp.rename(dst)
            _mark_ready(dst)
            return
        except Exception as e:
            print(f"[VAD] Download failed from {url}: {e}")

    with _lock:
        _download_state["status"] = "error"
        _download_state["error"] = "All download sources failed"


def _mark_ready(dst: Path):
    """Mark download as complete."""
    with _lock:
        _download_state["status"] = "ready"
        _download_state["progress"] = 100
        _download_state["error"] = ""
    print(f"[VAD] Model ready at {dst}")


# ── VAD 参数 ─────────────────────────────────────────────────────────┐
# v6.2 ONNX 模型按 512-sample 块处理，每块产生一个概率
FRAME_SEC = WINDOW_SAMPLES / SR  # 0.032s per frame (32ms)


# ── VAD 检测器（纯 ONNX Runtime，无需 silero-vad pip 包） ───────────


class VADDetector:
    """Silero-VAD ONNX inference wrapper — pure ONNX Runtime, no torch/pip.

    Manually implements chunked ONNX inference with LSTM state management,
    avoiding dependency on the ``silero-vad`` pip package.

    Usage::

        detector = VADDetector()
        intervals = detector.detect_speech(waveform, sr)
        pauses = detector.get_pause_boundaries(waveform, sr)
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or get_model_path()
        self._session: Optional[onnxruntime.InferenceSession] = None

    # ── lazy init ─────────────────────────────────────────────────

    def _ensure_model(self):
        """Ensure model file exists and ONNX session is loaded.

        模型缺失时直接抛 FileNotFoundError, 不再同步下载/轮询等待 —
        下载由模块级 ``ensure_model()`` 后台线程负责 (main.py lifespan 触发),
        避免首次 transcribe 被阻塞至多 60s. 调用方 (zipformer_engine
        ``_align_with_vad`` 等) 已有 try/except 降级, 行为安全.
        """
        if self._session is not None:
            return
        path = self._model_path
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            raise FileNotFoundError(
                f"VAD model not available at {path}; "
                "model may still be downloading in background"
            )
        self._session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"],
        )

    # ── 核心推理：chunked ONNX with LSTM state ─────────────────────

    def _get_speech_probs(self, waveform: np.ndarray) -> np.ndarray:
        """Run Silero-VAD chunked ONNX inference.

        v6.2 model requires 512-sample chunks with 64-sample context
        overlap, maintaining LSTM state across chunks.

        Args:
            waveform: 1-D float32 array, 16kHz mono.

        Returns:
            float32 array of speech probabilities, one per 512-sample frame.
        """
        self._ensure_model()
        wav = np.ascontiguousarray(waveform.astype(np.float32))

        context = np.zeros((1, 64), dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr_arr = np.array(SR, dtype=np.int64)
        probs: List[float] = []

        for start in range(0, len(wav), WINDOW_SAMPLES):
            chunk = wav[start:start + WINDOW_SAMPLES]
            if len(chunk) < WINDOW_SAMPLES:
                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
            chunk = chunk.reshape(1, -1)

            # Prepend context from previous chunk → [1, 64+512] = [1, 576]
            inp = np.concatenate([context, chunk], axis=1).astype(np.float32)

            out, state = self._session.run(
                ["output", "stateN"],
                {"input": inp, "state": state, "sr": sr_arr},
            )
            probs.append(float(out[0, 0]))

            # Update context: last 64 samples of current chunk
            context = chunk[:, -64:]

        return np.array(probs, dtype=np.float32)

    # ── 公共 API ──────────────────────────────────────────────────

    def detect_speech(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.25,
    ) -> List[Tuple[float, float]]:
        """Detect speech intervals in audio.

        Args:
            waveform: Audio waveform (1-D float32).
            sample_rate: Sample rate of waveform.
            threshold: Speech probability threshold.
            min_speech_duration: Minimum speech segment in seconds.
            min_silence_duration: Merge silences shorter than this (seconds).

        Returns:
            List of ``(start_sec, end_sec)`` speech intervals.
        """
        wav = self._prepare_waveform(waveform, sample_rate)
        probs = self._get_speech_probs(wav)

        # Threshold-based speech/non-speech
        speech_flags = probs > threshold

        intervals: List[Tuple[float, float]] = []
        in_speech = False
        start_frame = 0

        for i, is_speech in enumerate(speech_flags):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                duration = (i - start_frame) * FRAME_SEC
                if duration >= min_speech_duration:
                    intervals.append((
                        start_frame * FRAME_SEC,
                        i * FRAME_SEC,
                    ))
                in_speech = False

        if in_speech:
            duration = (len(speech_flags) - start_frame) * FRAME_SEC
            if duration >= min_speech_duration:
                intervals.append((
                    start_frame * FRAME_SEC,
                    len(speech_flags) * FRAME_SEC,
                ))

        # Merge short silence gaps
        if len(intervals) < 2:
            return intervals

        merged: List[Tuple[float, float]] = [intervals[0]]
        for curr in intervals[1:]:
            gap = curr[0] - merged[-1][1]
            if gap < min_silence_duration:
                merged[-1] = (merged[-1][0], curr[1])
            else:
                merged.append(curr)
        return merged

    def get_pause_boundaries(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        min_speech_duration: float = 0.1,
        min_pause_duration: float = 0.5,
    ) -> List[float]:
        """Find pause boundaries suitable for sentence segmentation.

        A pause boundary is the *middle* of a silence gap that is long
        enough to indicate a sentence break.

        Args:
            waveform: Audio waveform (1-D float32).
            sample_rate: Sample rate of waveform.
            min_speech_duration: Minimum speech segment (seconds) to keep.
            min_pause_duration: Minimum silence (seconds) to be a sentence break.

        Returns:
            Sorted list of timestamps (seconds) where sentences break.
        """
        speech = self.detect_speech(
            waveform, sample_rate,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_pause_duration,
        )
        if len(speech) < 2:
            return []

        boundaries: List[float] = []
        for i in range(1, len(speech)):
            gap_start, gap_end = speech[i - 1][1], speech[i][0]
            if (gap_end - gap_start) >= min_pause_duration:
                boundaries.append((gap_start + gap_end) / 2.0)
        return boundaries

    # ── 辅助 ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Ensure waveform is correct format for Silero-VAD (16kHz mono float32)."""
        wav = np.asarray(waveform, dtype=np.float32)
        if sample_rate != SR:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=SR)
            except ImportError:
                ratio = SR / sample_rate
                new_len = int(len(wav) * ratio)
                indices = np.linspace(0, len(wav) - 1, new_len)
                wav = np.interp(indices, np.arange(len(wav)), wav).astype(np.float32)
        return wav


# ── 便捷入口 ──────────────────────────────────────────────────────────


def get_detector() -> "VADDetector":
    """Return the module-level singleton VADDetector instance.

    Creates and pre-warms (loads ONNX session) on first call,
    returns the cached instance on subsequent calls.
    """
    global _detector
    if _detector is None:
        _detector = VADDetector()
        _detector._ensure_model()
    return _detector


def ensure_model() -> dict:
    """Ensure Silero-VAD model is available. Downloads in background if needed."""
    global _download_state
    model_path = get_model_path()
    if model_path.is_file() and model_path.stat().st_size > 1024 * 1024:
        # Pre-warm the detector singleton when model is already ready
        get_detector()
        return {"status": "ready", "progress": 100, "error": ""}

    with _lock:
        if _download_state.get("status") == "downloading":
            return dict(_download_state)
        _download_state = {"status": "downloading", "progress": 0, "error": ""}

    thread = threading.Thread(target=_download_model, daemon=True)
    thread.start()
    return {"status": "downloading", "progress": 0, "error": ""}
