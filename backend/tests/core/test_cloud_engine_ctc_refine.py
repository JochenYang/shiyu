"""Tests for cloud_engine._refine_timestamps_with_ctc fallback.

Verifies that when ctc_aligner (removed in v2.2) is missing, CTC timestamp
refinement degrades to None so the caller keeps the original segments
instead of crashing with a 500.
"""
import builtins
import sys
import wave
from types import SimpleNamespace

import numpy as np


def _write_wav(path, sr=16000, seconds=1.0):
    """用标准库写一个 440Hz 正弦波 wav, 避免额外依赖."""
    samples = (np.sin(2 * np.pi * 440 * np.arange(int(sr * seconds)) / sr) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """假 ONNX session: 任意输入返回固定 logits, 避免真实模型依赖."""

    def get_inputs(self):
        return [_FakeInput("speech"), _FakeInput("speech_lengths"),
                _FakeInput("language"), _FakeInput("text_norm")]

    def run(self, output_names, inputs):
        return [np.zeros((1, 100, 5), dtype=np.float32)]


def test_refine_timestamps_with_ctc_returns_none_when_aligner_missing(tmp_path, monkeypatch):
    """ctc_aligner 模块缺失 → 返回 None, 调用方回退原始 segments (不抛 500)."""
    from app.core.cloud_engine import _refine_timestamps_with_ctc
    import app.core.audio_processor as ap

    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path)

    # 模拟模型目录: 存在 model.onnx 与 bpe 模型 (函数内仅做 exists / 文件名检查)
    (tmp_path / "model.onnx").write_bytes(b"\x00" * 100)
    (tmp_path / "fake.bpe.model").write_bytes(b"x")

    # SentencePiece / onnxruntime 用假实现, 避免真实模型依赖
    class _FakeSPM:
        def Load(self, path):
            pass

    monkeypatch.setitem(sys.modules, "sentencepiece",
                        SimpleNamespace(SentencePieceProcessor=_FakeSPM))
    monkeypatch.setitem(sys.modules, "onnxruntime",
                        SimpleNamespace(InferenceSession=lambda *a, **k: _FakeSession()))
    monkeypatch.setattr(ap, "compute_fbank_kaldi",
                        lambda *a, **k: np.zeros((100, 80), dtype=np.float32))
    monkeypatch.setattr(ap, "apply_lfr", lambda fbank, m, n: fbank)
    monkeypatch.setattr(ap, "apply_cmvn", lambda features, cmvn: features)

    # 屏蔽 app.core.ctc_aligner 导入, 模拟模块已删除的环境
    real_import = builtins.__import__

    def _block_ctc_aligner(name, *args, **kwargs):
        if name == "app.core.ctc_aligner":
            raise ModuleNotFoundError("No module named 'app.core.ctc_aligner'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_ctc_aligner)
    # 注意: cloud_engine 内 `from app.core.config_loader import get_model_dir`
    # 本身是另一处残留 (函数级 import), 测试注入属性以便走到 ctc_aligner 分支
    import app.core.config_loader as config_loader
    monkeypatch.setattr(config_loader, "get_model_dir", lambda: str(tmp_path), raising=False)

    original_segments = [{"start": 0.0, "end": 1.0, "text": "测试文本"}]
    result = _refine_timestamps_with_ctc(str(wav_path), "测试文本", 16000, original_segments)
    assert result is None
