"""Tests for SenseVoice engine fallback and availability checks.

Covers:
1. _ctc_forced_decode 在 ctc_aligner 模块缺失时回退贪婪解码 (v2.2 已删除该模块)
2. SenseVoiceTranscriber.is_available() 检查 onnxruntime + 模型文件
3. transcribe_waveform_with_vad 重叠窗口拼接按时间过滤重复段
"""
import builtins
import sys

import numpy as np
import pytest
from unittest.mock import patch

from app.core.sensevoice_engine import SenseVoiceEngine, SenseVoiceTranscriber


# ─────────────────────────────────────────────────────────────────────
# 1. _ctc_forced_decode 回退 (ctc_aligner 已移除)
# ─────────────────────────────────────────────────────────────────────

def test_ctc_forced_decode_falls_back_to_greedy_when_aligner_missing(monkeypatch):
    """ctc_aligner 模块缺失 (v2.2 已删除) → 回退贪婪解码, 不抛 ModuleNotFoundError."""
    engine = SenseVoiceEngine.__new__(SenseVoiceEngine)
    engine.sp_model = object()  # 非 None → 走 forced alignment 分支

    fake_pairs = [(1, 0), (2, 5)]
    monkeypatch.setattr(
        SenseVoiceEngine, "_ctc_greedy_decode",
        lambda self, logits, input_length: fake_pairs,
    )

    # 屏蔽 app.core.ctc_aligner 导入, 模拟模块已删除的环境
    real_import = builtins.__import__

    def _block_ctc_aligner(name, *args, **kwargs):
        if name == "app.core.ctc_aligner":
            raise ModuleNotFoundError("No module named 'app.core.ctc_aligner'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_ctc_aligner)

    result = engine._ctc_forced_decode(np.zeros((10, 5)), 10)
    assert result == fake_pairs


# ─────────────────────────────────────────────────────────────────────
# 2. is_available() 失败路径
# ─────────────────────────────────────────────────────────────────────

class TestSenseVoiceIsAvailableFailures:
    @staticmethod
    def _make_transcriber(tmp_path, model_bytes=12_000_000, with_tokens=True):
        """用 __new__ 绕过 SenseVoiceEngine 实例化 (构造会读 tokens.json + 加载 ONNX)."""
        if with_tokens:
            (tmp_path / "tokens.json").write_text("[]")
        if model_bytes:
            (tmp_path / "model.onnx").write_bytes(b"\x00" * model_bytes)
        t = SenseVoiceTranscriber.__new__(SenseVoiceTranscriber)
        t._model_path = str(tmp_path / "model.onnx")
        t._tokens_path = str(tmp_path / "tokens.json")
        return t

    def test_is_available_false_when_onnxruntime_missing(self, tmp_path):
        """onnxruntime 缺失 → not available."""
        t = self._make_transcriber(tmp_path)
        with patch.dict(sys.modules, {"onnxruntime": None}):
            ok, reason = t.is_available()
        assert ok is False
        assert "onnxruntime" in reason.lower()

    def test_is_available_false_when_model_missing(self, tmp_path):
        """缺 model.onnx → not available."""
        t = self._make_transcriber(tmp_path, model_bytes=0)
        ok, reason = t.is_available()
        assert ok is False
        assert "model.onnx" in reason

    def test_is_available_false_when_model_too_small(self, tmp_path):
        """model.onnx 太小 (< 10MB) → not available."""
        t = self._make_transcriber(tmp_path, model_bytes=1000)  # 1KB
        ok, reason = t.is_available()
        assert ok is False
        assert "too small" in reason or "missing" in reason

    def test_is_available_false_when_tokens_missing(self, tmp_path):
        """tokens.json 缺失 → not available."""
        t = self._make_transcriber(tmp_path, with_tokens=False)
        ok, reason = t.is_available()
        assert ok is False
        assert "tokens.json" in reason


# ─────────────────────────────────────────────────────────────────────
# 3. is_available() 成功路径 (用大文件模拟)
# ─────────────────────────────────────────────────────────────────────

class TestSenseVoiceIsAvailableSuccess:
    def test_is_available_true_when_full_model_present(self, tmp_path):
        """model.onnx (>=10MB) + tokens.json 都存在 → available."""
        t = TestSenseVoiceIsAvailableFailures._make_transcriber(tmp_path)
        ok, reason = t.is_available()
        assert ok is True, f"expected available, got reason: {reason}"


# ─────────────────────────────────────────────────────────────────────
# 4. 重叠窗口拼接去重
# ─────────────────────────────────────────────────────────────────────

def _make_overlap_engine(monkeypatch, fake_transcribe):
    """构造 SenseVoiceEngine 并 mock 掉音频特征与转写依赖."""
    import librosa
    import app.core.audio_processor as ap

    # 能量 VAD: 返回空区间 → 退化为均匀切分, 行为确定
    monkeypatch.setattr(librosa.effects, "split", lambda *a, **k: [])
    monkeypatch.setattr(ap, "compute_fbank_kaldi", lambda *a, **k: np.zeros((100, 80), dtype=np.float32))
    monkeypatch.setattr(ap, "apply_lfr", lambda fbank, m, n: fbank)
    monkeypatch.setattr(ap, "apply_cmvn", lambda features, cmvn: features)
    monkeypatch.setattr(SenseVoiceEngine, "transcribe_with_timestamps", fake_transcribe)

    engine = SenseVoiceEngine.__new__(SenseVoiceEngine)
    return engine


def test_overlap_windows_clip_cross_boundary_segments(monkeypatch):
    """重叠窗口 (overlap_sec>0) 拼接: 完全落在重叠区的段丢弃,
    跨边界段 (起点重叠、终点超出) 裁剪起点保留句尾, 不丢内容."""
    calls = {"n": 0}

    def fake_transcribe(self, features, chunk_duration, language, external_text=None):
        calls["n"] += 1
        if calls["n"] == 1:
            # 窗口 1: 段一直延伸到窗口末尾附近
            return [{"start": 0.0, "end": 11.0, "text": "窗口一内容"}]
        # 窗口 2: 首段起点落在窗口 1 已覆盖区间内但终点超出 → 裁剪保留句尾;
        #         第二段完全在新区间 → 原样保留
        return [
            {"start": 0.0, "end": 11.0, "text": "窗口二重复内容"},
            {"start": 12.0, "end": 14.0, "text": "窗口二新内容"},
        ]

    engine = _make_overlap_engine(monkeypatch, fake_transcribe)

    # 20s @ 16kHz → 两个 10s chunk, 3s 重叠窗口
    waveform = np.zeros(20 * 16000, dtype=np.float32)
    segments = engine.transcribe_waveform_with_vad(
        waveform=waveform,
        sample_rate=16000,
        audio_cfg={},
        duration_sec=20.0,
        language="zh",
        overlap_sec=3.0,
    )

    assert len(segments) == 3
    assert segments[0]["text"] == "窗口一内容"
    assert segments[0]["start"] == 0.0
    # 跨边界段: 起点被裁剪到 covered_end (窗口 1 段的终点)
    assert segments[1]["text"] == "窗口二重复内容"
    assert segments[1]["start"] == 11.0
    assert segments[1]["end"] > 11.0
    assert segments[2]["text"] == "窗口二新内容"
    assert segments[2]["start"] > segments[1]["end"]
    # 保留段之间无时间重叠
    assert segments[0]["end"] <= segments[1]["start"]


def test_non_overlap_path_keeps_all_segments(monkeypatch):
    """非重叠路径 (overlap_sec=0) 不做时间过滤, 保留全部段."""
    calls = {"n": 0}

    def fake_transcribe(self, features, chunk_duration, language, external_text=None):
        calls["n"] += 1
        return [
            {"start": 0.0, "end": 5.0, "text": f"窗口{calls['n']}-a"},
            {"start": 5.0, "end": 8.0, "text": f"窗口{calls['n']}-b"},
        ]

    engine = _make_overlap_engine(monkeypatch, fake_transcribe)

    waveform = np.zeros(20 * 16000, dtype=np.float32)
    segments = engine.transcribe_waveform_with_vad(
        waveform=waveform,
        sample_rate=16000,
        audio_cfg={},
        duration_sec=20.0,
        language="zh",
        overlap_sec=0.0,
    )

    assert len(segments) == 4
    assert [s["text"] for s in segments] == ["窗口1-a", "窗口1-b", "窗口2-a", "窗口2-b"]
