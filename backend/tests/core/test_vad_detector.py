"""
VADDetector 模型加载策略测试 — 模型缺失时快速失败, 不做同步下载/轮询.

背景: v2.2 修复 P2 缺陷, _ensure_model 不再同步 _download_model() 并轮询
等待最多 60s (首次 transcribe 被阻塞). 下载由模块级 ensure_model() 后台
线程负责 (main.py lifespan 触发), 此处只负责快速抛错 + 降级.

覆盖:
  1. 模型缺失 → 直接抛 FileNotFoundError, 不下载、不等待
  2. 模型文件过小 (<1MB, 未下载完/损坏) → 同样抛 FileNotFoundError
  3. 模型就绪 → 正常加载 ONNX session (仅一次) 并完成 detect_speech
  4. ensure_model() 公共行为不变: 模型缺失时仍后台启动下载线程
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core import vad_detector
from app.core.vad_detector import VADDetector


class TestEnsureModelFailsFastWhenMissing:
    """模型缺失 → 快速抛 FileNotFoundError, 不做同步下载/轮询等待."""

    def test_missing_model_raises_file_not_found_fast(self, tmp_path):
        """模型文件不存在 → 抛 FileNotFoundError, 且毫秒级失败而非轮询 60s."""
        detector = VADDetector(model_path=tmp_path / "silero_vad.onnx")
        start = time.monotonic()
        with pytest.raises(FileNotFoundError, match="silero_vad.onnx"):
            detector._ensure_model()
        assert time.monotonic() - start < 5.0

    def test_missing_model_does_not_trigger_download(self, tmp_path, monkeypatch):
        """模型缺失时不应调用 _download_model (下载由后台线程负责)."""
        calls = []
        monkeypatch.setattr(vad_detector, "_download_model", lambda: calls.append(1))
        detector = VADDetector(model_path=tmp_path / "silero_vad.onnx")
        with pytest.raises(FileNotFoundError):
            detector._ensure_model()
        assert calls == []

    def test_too_small_model_raises_file_not_found(self, tmp_path):
        """文件存在但 <1MB (未下载完/损坏) → 视为不可用, 抛 FileNotFoundError."""
        tiny = tmp_path / "silero_vad.onnx"
        tiny.write_bytes(b"\x00" * 1024)
        detector = VADDetector(model_path=tiny)
        with pytest.raises(FileNotFoundError, match="silero_vad.onnx"):
            detector._ensure_model()


class TestEnsureModelLoadsWhenReady:
    """模型文件就绪 → 正常加载 session 并完成推理."""

    @staticmethod
    def _make_ready_model(tmp_path) -> object:
        """构造一个大小合法 (≥1MB) 的占位模型文件."""
        model = tmp_path / "silero_vad.onnx"
        model.write_bytes(b"\x00" * (1024 * 1024 + 1))
        return model

    def test_ready_model_loads_session_once(self, tmp_path):
        """有效模型文件 → 创建 ONNX session, 二次调用复用缓存."""
        model = self._make_ready_model(tmp_path)
        fake_session = MagicMock()
        with patch.object(
            vad_detector.onnxruntime, "InferenceSession", return_value=fake_session
        ) as mock_factory:
            detector = VADDetector(model_path=model)
            detector._ensure_model()
            detector._ensure_model()
            assert mock_factory.call_count == 1
            assert detector._session is fake_session

    def test_detect_speech_runs_with_ready_model(self, tmp_path):
        """session 就绪 → detect_speech 走完整分块推理并返回语音区间."""
        model = self._make_ready_model(tmp_path)
        fake_session = MagicMock()
        fake_session.run.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )
        with patch.object(
            vad_detector.onnxruntime, "InferenceSession", return_value=fake_session
        ):
            detector = VADDetector(model_path=model)
            waveform = np.sin(
                2 * np.pi * 440 * np.arange(16000) / 16000
            ).astype(np.float32)
            intervals = detector.detect_speech(waveform, sample_rate=16000)

        assert fake_session.run.call_count > 10  # 512-sample 分块推理确实执行
        assert intervals, "全语音正弦波应至少返回一个语音区间"
        assert intervals[0][0] == 0.0
        assert intervals[0][1] > 0.5


class TestEnsureModelPublicBehavior:
    """模块级 ensure_model() 公共行为保持不变 — 缺失时仍后台下载."""

    def test_ensure_model_starts_background_download_when_missing(
        self, tmp_path, monkeypatch
    ):
        """模型缺失 → 返回 downloading 状态, 并启动后台下载线程."""
        monkeypatch.setattr(
            vad_detector, "get_model_path", lambda: tmp_path / "silero_vad.onnx"
        )
        started = threading.Event()

        def fake_download():
            started.set()

        monkeypatch.setattr(vad_detector, "_download_model", fake_download)
        status = vad_detector.ensure_model()
        assert status["status"] == "downloading"
        assert started.wait(2.0), "后台下载线程应被启动"
