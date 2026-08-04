"""
Phase 2 v2.0 refactor — Zipformer-CTC-zh engine tests.

覆盖:
  1. 模块可 import (sherpa-onnx 存在 / 不存在都优雅)
  2. is_available() 在缺文件时返回 False + 明确 reason
  3. is_available() 在完整模型时返回 True
  4. from_config() 正确解析 model_dir / num_threads
  5. _result_to_segments() 边界 (空文本 / 空 timestamps / 单 token /
                              VAD 切分 / 整段输出)
  6. (integration) 端到端: 加载真实模型 + 转录 test_wavs/0.wav

默认只跑 mock + 单元测试; 集成测试用 `pytest -m integration` 单独跑。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 默认 marker 注册 (pytest 9 不注册会 warn, 不阻断)
pytestmark = []


# ─────────────────────────────────────────────────────────────────────
# 1. Import safety
# ─────────────────────────────────────────────────────────────────────

class TestZipformerImportSafety:
    """ZipformerEngine 在 sherpa-onnx 装 / 没装时都能 import."""

    def test_module_imports_without_sherpa_onnx(self):
        """当 sherpa_onnx 不可用, 模块 import 仍应成功 (运行时才报错)."""
        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            if "app.core.zipformer_engine" in sys.modules:
                del sys.modules["app.core.zipformer_engine"]
            from app.core.zipformer_engine import ZipformerEngine
            assert ZipformerEngine is not None
            assert ZipformerEngine.name == "zipformer_local"

    def test_class_metadata(self):
        """name / display_name / is_local / requires_network 注册名/前端 dropdown."""
        from app.core.zipformer_engine import ZipformerEngine
        assert ZipformerEngine.name == "zipformer_local"
        assert "Zipformer" in ZipformerEngine.display_name
        assert ZipformerEngine.is_local is True
        assert ZipformerEngine.requires_network is False

    def test_engine_extends_base_transcriber(self):
        """必须继承 BaseTranscriber (engine_registry 通过多态调用)."""
        from app.core.engine_base import BaseTranscriber
        from app.core.zipformer_engine import ZipformerEngine
        assert issubclass(ZipformerEngine, BaseTranscriber)


# ─────────────────────────────────────────────────────────────────────
# 2. is_available() 失败路径
# ─────────────────────────────────────────────────────────────────────

class TestIsAvailableFailures:
    def test_unavailable_when_sherpa_not_installed(self, tmp_path):
        """sherpa-onnx 缺失 → not available."""
        # 先造一个虚假完整模型目录, 然后屏蔽 sherpa_onnx import
        (tmp_path / "model.int8.onnx").write_bytes(b"\x00" * 200_000_000)
        (tmp_path / "tokens.txt").write_text("dummy")

        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            if "app.core.zipformer_engine" in sys.modules:
                del sys.modules["app.core.zipformer_engine"]
            from app.core.zipformer_engine import ZipformerEngine
            engine = ZipformerEngine(model_dir=str(tmp_path))
            ok, reason = engine.is_available()
            assert ok is False
            assert "sherpa-onnx" in reason.lower()

    def test_unavailable_when_model_dir_missing(self):
        """模型目录不存在 → not available."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir="/nonexistent/path/zipformer")
        ok, reason = engine.is_available()
        assert ok is False
        assert "not found" in reason.lower()

    def test_unavailable_when_model_onnx_missing(self, tmp_path):
        """缺 model.int8.onnx → not available."""
        (tmp_path / "tokens.txt").write_text("dummy")
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir=str(tmp_path))
        ok, reason = engine.is_available()
        assert ok is False
        assert "model.int8.onnx" in reason

    def test_unavailable_when_model_too_small(self, tmp_path):
        """model.int8.onnx 太小 (< 100MB) → not available."""
        (tmp_path / "model.int8.onnx").write_bytes(b"\x00" * 1000)  # 1KB
        (tmp_path / "tokens.txt").write_text("dummy")
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir=str(tmp_path))
        ok, reason = engine.is_available()
        assert ok is False
        assert "too small" in reason or "missing" in reason

    def test_unavailable_when_tokens_missing(self, tmp_path):
        """tokens.txt 缺失 → not available."""
        (tmp_path / "model.int8.onnx").write_bytes(b"\x00" * 200_000_000)
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir=str(tmp_path))
        ok, reason = engine.is_available()
        assert ok is False
        assert "tokens" in reason


# ─────────────────────────────────────────────────────────────────────
# 3. is_available() 成功路径 (用大文件模拟)
# ─────────────────────────────────────────────────────────────────────

class TestIsAvailableSuccess:
    def test_available_when_full_model_present(self, tmp_path):
        """model.int8.onnx (>=100MB) + tokens.txt 都存在 → available."""
        (tmp_path / "model.int8.onnx").write_bytes(b"\x00" * 200_000_000)
        # tokens.txt 真实约 13KB, 至少 100 字节
        (tmp_path / "tokens.txt").write_text("<blk> 0\n<sos/eos> 1\n" * 20)
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir=str(tmp_path))
        ok, reason = engine.is_available()
        assert ok is True, f"expected available, got reason: {reason}"


# ─────────────────────────────────────────────────────────────────────
# 4. from_config() 配置解析
# ─────────────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_from_config_defaults(self):
        """不传 model_dir → 使用 ~/.shiyu/models/zipformer-ctc-zh."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine.from_config({})
        assert "zipformer-ctc-zh" in engine._model_dir
        assert engine._num_threads == 4

    def test_from_config_explicit_model_dir(self, tmp_path):
        """显式 model_dir → 解析成绝对路径."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine.from_config({"model_dir": str(tmp_path)})
        assert Path(engine._model_dir).resolve() == tmp_path.resolve()

    def test_from_config_expanduser(self):
        """~ 展开成 user home."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine.from_config({"model_dir": "~/foo/bar"})
        assert "~" not in engine._model_dir
        assert engine._model_dir.endswith("foo" + os.sep + "bar")

    def test_from_config_num_threads(self):
        """num_threads 自定义."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine.from_config({"num_threads": 8})
        assert engine._num_threads == 8


# ─────────────────────────────────────────────────────────────────────
# 5. _result_to_segments() 边界
# ─────────────────────────────────────────────────────────────────────

class TestResultToSegments:
    def _make_engine(self, tmp_path):
        """构造一个不需要 sherpa-onnx 的实例."""
        from app.core.zipformer_engine import ZipformerEngine
        return ZipformerEngine(model_dir=str(tmp_path), num_threads=1)

    def test_empty_text_returns_empty(self, tmp_path):
        """result.text 为空 → [] segments."""
        engine = self._make_engine(tmp_path)
        result = MagicMock(text="", timestamps=[0.1, 0.2, 0.3])
        assert engine._result_to_segments(result) == []

    def test_whitespace_only_text_returns_empty(self, tmp_path):
        """纯空白文本 → [] segments."""
        engine = self._make_engine(tmp_path)
        result = MagicMock(text="   \n\t  ", timestamps=[0.1])
        assert engine._result_to_segments(result) == []

    def test_no_timestamps_returns_zeroed_segment(self, tmp_path):
        """text 有但 timestamps 为空 → 1 段, 时间为 0."""
        engine = self._make_engine(tmp_path)
        result = MagicMock(text="你好世界", timestamps=[])
        segments = engine._result_to_segments(result)
        assert len(segments) == 1
        assert segments[0].text == "你好世界"
        assert segments[0].start == 0.0
        assert segments[0].end == 0.0

    def test_whole_utterance_uses_first_last_timestamp(self, tmp_path):
        """无 VAD 切分 → 1 段, [timestamps[0], timestamps[-1]]."""
        engine = self._make_engine(tmp_path)
        result = MagicMock(
            text="今天天气不错",
            timestamps=[0.16, 0.48, 0.80, 1.12, 1.44, 1.76],
            segment_texts=[],
            segment_timestamps=[],
            segment_durations=[],
        )
        segments = engine._result_to_segments(result)
        assert len(segments) == 1
        assert segments[0].start == 0.16
        assert segments[0].end == 1.76
        assert segments[0].text == "今天天气不错"

    def test_vad_segmented_result_uses_segment_texts(self, tmp_path):
        """VAD 切分结果 → 多段, 时间戳来自 segment_timestamps."""
        engine = self._make_engine(tmp_path)
        result = MagicMock(
            text="第一段第二段",  # 全文本会被忽略
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            segment_texts=["第一段", "第二段"],
            segment_timestamps=[0.0, 2.0],
            segment_durations=[1.5, 1.8],
        )
        segments = engine._result_to_segments(result)
        assert len(segments) == 2
        assert segments[0].text == "第一段"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[1].text == "第二段"
        assert segments[1].start == 2.0
        assert segments[1].end == 3.8

    def test_vad_segmented_skips_empty_segments(self, tmp_path):
        """VAD 切分中有空白段 → 跳过."""
        engine = self._make_engine(tmp_path)
        result = MagicMock(
            text="",
            timestamps=[0.0, 0.5, 1.0],
            segment_texts=["有内容", "", "  "],
            segment_timestamps=[0.0, 1.0, 2.0],
            segment_durations=[0.5, 0.5, 0.5],
        )
        segments = engine._result_to_segments(result)
        assert len(segments) == 1
        assert segments[0].text == "有内容"


# ─────────────────────────────────────────────────────────────────────
# 6. VAD 校准 (P0: interval 覆盖多段不再整体替换, 吸附后强制单调)
# ─────────────────────────────────────────────────────────────────────

class TestVadAlignment:
    """_align_with_vad 的最近边界吸附 / 容差保留 / 单调性."""

    def _align(self, segments, intervals, tolerance=1.0):
        """用 fake VADDetector 跑 _align_with_vad, 返回校准后 segments."""
        from app.core.zipformer_engine import ZipformerEngine
        audio = np.zeros(16000 * 40, dtype=np.float32)  # 40s 静音垫底
        with patch("app.core.vad_detector.VADDetector") as mock_vad:
            mock_vad.return_value.detect_speech.return_value = intervals
            return ZipformerEngine._align_with_vad(
                audio, 16000, segments, boundary_tolerance=tolerance,
            )

    def test_single_interval_covering_multiple_segments_stays_monotonic(self):
        """P0 回归: 单 interval 覆盖多段 → 各段不再整体替换为 interval 端点, 输出单调不重叠."""
        from app.core.engine_base import Segment
        segments = [
            Segment(start=1.0, end=4.0, text="第一句"),
            Segment(start=5.0, end=9.0, text="第二句"),
            Segment(start=10.0, end=14.0, text="第三句"),
            Segment(start=15.0, end=19.0, text="第四句"),
            Segment(start=20.0, end=24.0, text="第五句"),
        ]
        aligned = self._align(segments, [(0.0, 30.0)])
        assert len(aligned) == 5
        assert aligned[0].start == 0.0  # 首段 start 距 interval 起点 1s, 容差内吸附
        assert aligned[-1].end == 24.0  # 远离 interval 终点的 end 保留原值
        for prev, curr in zip(aligned, aligned[1:]):
            assert curr.start >= prev.end, (
                f"校准后段时间戳必须单调不重叠: {prev} → {curr}"
            )
            assert curr.end >= curr.start

    def test_segment_start_snaps_when_near_interval_start(self):
        """段起点距 interval 起点 < 1s → 吸附到 interval 起点."""
        from app.core.engine_base import Segment
        segments = [
            Segment(start=4.2, end=6.0, text="近起点"),   # 距起点 0.8s → 吸附 5.0
            Segment(start=7.0, end=10.7, text="近终点"),  # end 距终点 0.7s → 吸附 10.0
        ]
        aligned = self._align(segments, [(5.0, 10.0)])
        assert aligned[0].start == 5.0
        assert aligned[0].end == 6.0
        assert aligned[1].start == 7.0
        assert aligned[1].end == 10.0

    def test_segment_timestamps_kept_when_far_from_interval_boundary(self):
        """段起点/终点距 interval 边界 > 1s → 保留原值 (interval 内部不再整体替换)."""
        from app.core.engine_base import Segment
        segments = [
            Segment(start=2.0, end=3.0, text="起点前"),    # 距起点 3s → 保留
            Segment(start=7.0, end=8.0, text="interval内"),  # P0 回归: 在 interval 内但距边界 >1s → 保留
            Segment(start=10.8, end=12.5, text="终点后"),  # start 只找最近起点, end 只找最近终点 → 均保留
        ]
        aligned = self._align(segments, [(5.0, 10.0)])
        assert [(s.start, s.end) for s in aligned] == [
            (2.0, 3.0), (7.0, 8.0), (10.8, 12.5),
        ]

    def test_alignment_forces_monotonic_timestamps(self):
        """校准后后段起点早于前段终点 → 强制修正为前段终点."""
        from app.core.engine_base import Segment
        segments = [
            Segment(start=0.5, end=8.0, text="前段"),
            Segment(start=5.0, end=12.0, text="后段"),  # 吸附后 start 5.0 < 前段 end 8.0
            Segment(start=20.0, end=28.0, text="尾段"),
        ]
        aligned = self._align(segments, [(0.0, 30.0)])
        assert aligned[0].start == 0.0  # 距起点 0.5s → 吸附
        assert aligned[1].start == 8.0  # 被强制为前段终点, 不再重叠
        assert aligned[1].end == 12.0
        for prev, curr in zip(aligned, aligned[1:]):
            assert curr.start >= prev.end


# ─────────────────────────────────────────────────────────────────────
# 7. 分块重叠去重 (P2: 块边界 2s 上下文重叠 + covered_end 去重)
# ─────────────────────────────────────────────────────────────────────

class _FakeStream:
    """记录输入波形的假 sherpa-onnx stream."""

    def __init__(self):
        self.waveform = None

    def accept_waveform(self, sample_rate, waveform):
        self.waveform = waveform


class _ChunkedFakeRecognizer:
    """按调用次序返回每个 chunk 预设结果的假 recognizer, 并记录输入波形.

    chunk_specs: [(texts, timestamps, durations), ...], 时间戳相对各自
    chunk 音频起点.
    """

    def __init__(self, chunk_specs):
        self._chunk_specs = chunk_specs
        self.captured = []  # 每个 chunk 的实际输入波形 (断言重叠回退用)

    def create_stream(self):
        return _FakeStream()

    def decode_stream(self, stream):
        idx = len(self.captured)
        texts, ts, dur = self._chunk_specs[idx]
        self.captured.append(stream.waveform)
        stream.result = MagicMock(
            text="",
            timestamps=[],
            segment_texts=texts,
            segment_timestamps=ts,
            segment_durations=dur,
        )


class TestChunkOverlapDedup:
    """长音频分块: 第二块起回退 2s 上下文, 与已覆盖区间重叠的重复段被丢弃."""

    def test_chunk_overlap_and_dedup(self, tmp_path):
        """65s 音频分 3 块: 重叠区重复段被丢弃, 时间偏移按块起点计算."""
        import soundfile as sf
        from app.core.zipformer_engine import ZipformerEngine

        sr = 16000
        wav = tmp_path / "long.wav"
        sf.write(str(wav), np.zeros(65 * sr, dtype=np.float32), sr)

        # 块 1 [0s, 30s): 末段 CTC 尾巴溢出到 30.5s (覆盖了块 2 起点)
        # 块 2 输入回退到 [28s, 60s): 重叠区 [28s, 30s) 的转录绝对时间
        #   落在 [28s, 30s), 与 covered_end=30.5 重叠 → 应被丢弃
        chunk_specs = [
            (["说话内容", "更多内容"], [0.0, 25.0], [5.0, 5.5]),
            (["重叠内容", "继续说话"], [0.0, 4.0], [2.0, 6.0]),
            (["收尾内容"], [0.0], [4.0]),
        ]
        recognizer = _ChunkedFakeRecognizer(chunk_specs)
        engine = ZipformerEngine(model_dir=str(tmp_path), num_threads=1)
        engine._recognizer = recognizer

        # VAD 校准走 fake (空 intervals), 避免依赖真实 VAD 模型
        with patch("app.core.vad_detector.VADDetector") as mock_vad:
            mock_vad.return_value.detect_speech.return_value = []
            result = engine.transcribe(str(wav))

        # 修正偏移后: 块 2 音频起点 28s, 重叠重复段 (28, 30) 被丢弃,
        # 新内容 (32, 38); 块 3 音频起点 58s, 内容 (58, 62)
        assert [(s.text, round(s.start, 3), round(s.end, 3)) for s in result.segments] == [
            ("说话内容", 0.0, 5.0),
            ("更多内容", 25.0, 30.5),
            ("继续说话", 32.0, 38.0),   # 块 2 的重叠重复段 (28, 30) 被丢弃
            ("收尾内容", 58.0, 62.0),
        ]
        # 块输入回退: 块 1 [0, 30s), 块 2 [28s, 60s), 块 3 [58s, 65s)
        assert len(recognizer.captured[0]) == 30 * sr
        assert len(recognizer.captured[1]) == (60 - 28) * sr
        assert len(recognizer.captured[2]) == (65 - 58) * sr


# ─────────────────────────────────────────────────────────────────────
# 8. E2E 集成测试 (真实模型)
# ─────────────────────────────────────────────────────────────────────

ZIPFORMER_MODEL_DIR = os.path.expanduser("~/.shiyu/models/zipformer-ctc-zh")


@pytest.mark.integration
class TestZipformerE2E:
    """真实模型跑 test_wavs/0.wav (5.6 秒中文). 需要先下载模型."""

    @pytest.fixture(autouse=True)
    def _require_model(self):
        if not (Path(ZIPFORMER_MODEL_DIR) / "model.int8.onnx").exists():
            pytest.skip(f"模型未下载, 跳过: {ZIPFORMER_MODEL_DIR}")

    def test_e2e_transcribe_returns_chinese(self):
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine.from_config({"model_dir": ZIPFORMER_MODEL_DIR})
        ok, reason = engine.is_available()
        assert ok is True, f"is_available failed: {reason}"

        wav = Path(ZIPFORMER_MODEL_DIR) / "test_wavs" / "0.wav"
        assert wav.exists(), f"sample wav missing: {wav}"

        result = engine.transcribe(str(wav))
        # 中文音频应返回至少 1 段非空文本
        assert result.language == "zh"
        assert len(result.segments) >= 1
        assert result.full_text.strip() != ""
        # 时间戳 sanity: start < end
        for seg in result.segments:
            assert seg.start >= 0
            assert seg.end >= seg.start

    def test_e2e_segment_timestamps_match_duration(self):
        """端点时间戳应大致覆盖 wav 时长."""
        from app.core.zipformer_engine import ZipformerEngine
        import soundfile as sf

        engine = ZipformerEngine.from_config({"model_dir": ZIPFORMER_MODEL_DIR})
        wav = Path(ZIPFORMER_MODEL_DIR) / "test_wavs" / "0.wav"
        audio, sr = sf.read(str(wav))
        audio_duration = len(audio) / sr

        result = engine.transcribe(str(wav))
        assert len(result.segments) >= 1
        seg = result.segments[0]
        # 端点不能超过 wav 时长太多 (>2s 误差算异常)
        assert seg.end <= audio_duration + 2.0, (
            f"end={seg.end} > duration={audio_duration}"
        )