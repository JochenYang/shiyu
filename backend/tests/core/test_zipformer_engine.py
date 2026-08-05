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

    def test_from_config_ctc_delay_sec(self):
        """ctc_delay_sec 自定义; 缺省时用默认 0.15."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine.from_config({"ctc_delay_sec": 0.2})
        assert engine._ctc_delay_sec == 0.2
        engine_default = ZipformerEngine.from_config({})
        assert engine_default._ctc_delay_sec == 0.15

    def test_init_kwargs_from_config_ctc_delay_sec(self):
        """生产注册路径 (init_kwargs_from_config) 透传 ctc_delay_sec.

        回归: reviewer P1-2 — main.py 用 init_kwargs_from_config 注册,
        缺失时 config.yaml engines.zipformer.ctc_delay_sec 不生效.
        """
        from app.core.zipformer_engine import ZipformerEngine
        kwargs = ZipformerEngine.init_kwargs_from_config({"ctc_delay_sec": 0.0})
        assert kwargs["ctc_delay_sec"] == 0.0
        kwargs_default = ZipformerEngine.init_kwargs_from_config({})
        assert kwargs_default["ctc_delay_sec"] == 0.15

    def test_transcriber_from_config_ctc_delay_sec(self):
        """ZipformerTranscriber.from_config (注册即用) 透传 ctc_delay_sec."""
        from app.core.zipformer_engine import ZipformerTranscriber
        t = ZipformerTranscriber.from_config({"ctc_delay_sec": 0.2})
        assert t._engine._ctc_delay_sec == 0.2


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
        """无 VAD 切分 → 1 段, [timestamps[0], timestamps[-1]], 减去 CTC 滞后 0.15s."""
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
        assert segments[0].start == 0.01
        assert segments[0].end == 1.61
        assert segments[0].text == "今天天气不错"

    def test_ctc_delay_disabled_keeps_raw_timestamps(self, tmp_path):
        """ctc_delay_sec=0 → 不补偿, 保留 sherpa-onnx 原始时间戳."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir=str(tmp_path), num_threads=1, ctc_delay_sec=0.0)
        result = MagicMock(
            text="今天天气不错",
            timestamps=[0.16, 0.48, 0.80, 1.12, 1.44, 1.76],
            segment_texts=[],
            segment_timestamps=[],
            segment_durations=[],
        )
        segments = engine._result_to_segments(result)
        assert segments[0].start == 0.16
        assert segments[0].end == 1.76

    def test_ctc_delay_custom_value(self, tmp_path):
        """自定义补偿值生效 (0.2s)."""
        from app.core.zipformer_engine import ZipformerEngine
        engine = ZipformerEngine(model_dir=str(tmp_path), num_threads=1, ctc_delay_sec=0.2)
        result = MagicMock(
            text="今天天气不错",
            timestamps=[0.16, 0.48, 0.80, 1.12, 1.44, 1.76],
            segment_texts=[],
            segment_timestamps=[],
            segment_durations=[],
        )
        segments = engine._result_to_segments(result)
        assert segments[0].start == 0.0  # 0.16 - 0.2 → clamp 0
        assert segments[0].end == 1.56

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
        assert segments[0].start == 0.0    # 0 - 0.15 → clamp 0
        assert segments[0].end == 1.5
        assert segments[1].text == "第二段"
        assert segments[1].start == 1.85   # 2.0 - 0.15
        assert segments[1].end == 3.65     # 1.85 + 1.8

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

        # 块 1 [0s, 30s) 内容区 [0, 28): 末段 CTC 尾巴溢出到 30.5s,
        #   但只保留到内容区末尾 28s (块尾 2s 重叠区交给块 2 完整解码)
        # 块 2 输入回退到 [28s, 60s) 内容区 [30, 58): 重叠区 [28, 30) 的
        #   "重叠内容" 绝对时间落在内容区外 → 整段丢弃 (无 char_times 路径)
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

        # v2.3 语义: 每块只输出内容区 [chunk_start, 块尾-2s) 的句子,
        #   起点钳制到内容区起点; 所有时间戳减 0.15s CTC 滞后补偿
        #   (起点 < 0 时 clamp 到 0)
        assert [(s.text, round(s.start, 3), round(s.end, 3)) for s in result.segments] == [
            ("说话内容", 0.0, 5.0),
            ("更多内容", 24.85, 28.0),     # 块尾 2s 重叠区截断, 交给块 2 重新解码
            ("继续说话", 31.85, 37.85),   # 块 2 的重叠重复段 (27.85, 29.85) 整段丢弃
            ("收尾内容", 60.0, 61.85),    # 块 3 内容区 [60, 65): 起点钳制到 60
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

    def test_e2e_long_audio_chunked_path(self, tmp_path):
        """>30s 音频走分块路径 (reviewer P2-2): 无异常、时间戳不超界、无相邻重复.

        用 0.wav 拼接 6 段 + 段间静音 → 33.7s, 触发 30s 分块 + 2s 重叠.
        """
        import numpy as np
        import soundfile as sf
        from app.core.zipformer_engine import ZipformerEngine

        engine = ZipformerEngine.from_config({"model_dir": ZIPFORMER_MODEL_DIR})
        src = Path(ZIPFORMER_MODEL_DIR) / "test_wavs" / "0.wav"
        audio, sr = sf.read(str(src), dtype="float32")
        silence = np.zeros(int(0.4 * sr), dtype="float32")
        long_audio = np.concatenate([audio, silence] * 6)
        long_wav = tmp_path / "long.wav"
        sf.write(str(long_wav), long_audio, sr)
        duration = len(long_audio) / sr
        assert duration > 30.0  # 必须触发分块

        result = engine.transcribe(str(long_wav))
        assert len(result.segments) >= 1
        for i, seg in enumerate(result.segments):
            assert seg.start >= 0
            assert seg.end >= seg.start
            assert seg.end <= duration + 2.0, f"seg {i} end={seg.end} > duration={duration}"
            assert len(seg.text) > 0
        # 无相邻重复段 (块边界去重回归)
        for i in range(1, len(result.segments)):
            assert result.segments[i].text != result.segments[i - 1].text, (
                f"相邻重复段: {result.segments[i].text!r}"
            )

# ─────────────────────────────────────────────────────────────────────
# 9. v2.3 块边界处理: _trim_segment / _dedup_overlap / _build_char_times
# ─────────────────────────────────────────────────────────────────────

class TestTrimSegment:
    """_trim_segment: 块内容区裁剪 + 停顿点对齐 (整句保留, 残句丢弃)."""

    def _mk(self, text, times, start=0.0, end=None):
        from app.core.engine_base import Segment
        if end is None:
            end = times[-1] + 1.0
        return Segment(start=start, end=end, text=text, char_times=times)

    def test_no_char_times_clamps_bounds(self):
        """无 char_times (VAD 切分路径): 直接钳制 start/end."""
        from app.core.engine_base import Segment
        from app.core.zipformer_engine import ZipformerEngine
        seg = Segment(start=1.0, end=9.0, text="测试文本")
        out = ZipformerEngine._trim_segment(seg, 2.0, 8.0)
        assert out.start == 2.0
        assert out.end == 8.0
        assert out.text == "测试文本"

    def test_empty_interval_returns_none(self):
        """内容区内没有内容 (整段落在已覆盖区) → None."""
        from app.core.engine_base import Segment
        from app.core.zipformer_engine import ZipformerEngine
        seg = Segment(start=3.0, end=4.0, text="重叠", char_times=[3.0, 3.5])
        assert ZipformerEngine._trim_segment(seg, 5.0, 8.0) is None

    def test_segment_after_hi_returns_none(self):
        """整段在 hi 之后 (块尾解码内容) → None (由下一块负责)."""
        from app.core.zipformer_engine import ZipformerEngine
        seg = self._mk("下一块内容", [9.0, 9.1, 9.2, 9.3, 9.4], start=9.0)
        assert ZipformerEngine._trim_segment(seg, 5.0, 8.0) is None

    def test_trim_keeps_complete_sentence_across_hi(self):
        """句尾跨 hi 时整句保留 (end 扩展到句尾真实时间, 而不是截断)."""
        from app.core.zipformer_engine import ZipformerEngine
        # 第一句 [0.0, 0.3], 第二句 [1.4, 1.7], 句间停顿 1.1s
        # 内容区 [1.0, 1.5): 落在第二句中间. 句首回退 → "五" (1.4),
        # 句尾扩展 → "八" (1.7), 整句跨 hi 仍完整保留, end = 句尾真实时间
        text = "一二三四" + "五六七八"
        times = [0.0, 0.1, 0.2, 0.3, 1.4, 1.5, 1.6, 1.7]
        seg = self._mk(text, times)
        out = ZipformerEngine._trim_segment(seg, 1.0, 1.5)
        assert out.text == "五六七八"
        assert out.char_times == [1.4, 1.5, 1.6, 1.7]
        # end 覆盖到句尾真实时间 (1.7), 不截断到 hi (1.5)
        assert out.end == 1.7
        # 句首 1.4 > lo=1.0, start 钳到 lo (内容区起点)
        assert out.start == 1.0

    def test_trim_with_pause_alignment(self):
        """裁剪点对齐停顿点: 起点回退到句首, 终点扩展到句尾."""
        from app.core.zipformer_engine import ZipformerEngine
        # 4 字一句, 句间停顿 1s. 内容区 [1.0, 3.2) 落在第二句内
        text = "一二三四" + "五六七八" + "九十甲乙"
        times = [0.0, 0.1, 0.2, 0.3, 1.4, 1.5, 1.6, 1.7, 2.8, 2.9, 3.0, 3.1]
        seg = self._mk(text, times)
        out = ZipformerEngine._trim_segment(seg, 1.0, 3.2)
        # i0: 第一个 t>=1.0 = "五"(1.4) → 回退到句首 "五" (1.4 前 0.3 停顿 1.1s)
        # i1: 最后一个 t<3.2 = "甲"(2.8)... 扩展链到句尾 "乙"(3.1)
        # 文本 = "五六七八九十甲乙"
        assert out.text == "五六七八九十甲乙"
        assert out.char_times == [1.4, 1.5, 1.6, 1.7, 2.8, 2.9, 3.0, 3.1]
        assert out.start == 1.0  # 句首 1.4 > lo=1.0, 钳到内容区起点
        assert out.end == 3.2    # 钳制到 hi

    def test_start_clamped_to_lo_when_sentence_starts_inside(self):
        """句首在内容区内时, start 钳到 lo."""
        from app.core.zipformer_engine import ZipformerEngine
        text = "一二三四" + "五六七八"
        times = [0.0, 0.1, 0.2, 0.3, 1.6, 1.7, 1.8, 1.9]
        seg = self._mk(text, times)
        out = ZipformerEngine._trim_segment(seg, 1.0, 2.0)
        # i0 = "五"(1.6) >= 1.0, 回退: 前有停顿 (0.3→1.6 = 1.3s) → 停在 "五"
        # start = max(seg.start=0, min(lo=1.0, ct[i0]=1.6)) = 1.0
        assert out.text == "五六七八"
        assert out.start == 1.0


class TestDedupOverlap:
    """_dedup_overlap: 相邻段时间重叠 + 文本包含 → 去重."""

    def _mk(self, text, start, end):
        from app.core.engine_base import Segment
        return Segment(start=start, end=end, text=text)

    def test_drop_tail_duplicate(self):
        """后段是前段子串且时间重叠 → 丢弃后段."""
        from app.core.zipformer_engine import ZipformerEngine
        segs = [
            self._mk("然后当我们的计时器超时时", 27.5, 30.0),
            self._mk("当我们的计时器超时时", 28.0, 30.1),
        ]
        out = ZipformerEngine._dedup_overlap(segs)
        assert [(s.text, s.start) for s in out] == [("然后当我们的计时器超时时", 27.5)]

    def test_replace_prev_with_full_sentence(self):
        """前段是后段子串 (后段更完整) → 用后段替换前段."""
        from app.core.zipformer_engine import ZipformerEngine
        segs = [
            self._mk("如果我们移除玩家的", 58.0, 59.4),
            self._mk("如果我们移除玩家的碰撞体", 58.3, 62.0),
        ]
        out = ZipformerEngine._dedup_overlap(segs)
        assert [(s.text) for s in out] == ["如果我们移除玩家的碰撞体"]

    def test_identical_text_drops_second(self):
        """完全相同文本 + 时间重叠 → 保留前段."""
        from app.core.zipformer_engine import ZipformerEngine
        segs = [
            self._mk("重复的内容", 10.0, 12.0),
            self._mk("重复的内容", 10.2, 12.1),
        ]
        out = ZipformerEngine._dedup_overlap(segs)
        assert len(out) == 1
        assert out[0].start == 10.0

    def test_no_overlap_kept(self):
        """时间不重叠 → 即使子串也保留 (不同时段内容是独立的)."""
        from app.core.zipformer_engine import ZipformerEngine
        segs = [
            self._mk("第一句内容", 0.0, 2.0),
            self._mk("第一句", 5.0, 6.0),
        ]
        out = ZipformerEngine._dedup_overlap(segs)
        assert len(out) == 2

    def test_adjacent_zero_gap_duplicate_dropped(self):
        """块尾整句越界时, 前后段 end/start 完全邻接 (0 间隔) → 仍去重."""
        from app.core.zipformer_engine import ZipformerEngine
        segs = [
            self._mk("然后当我们的计时器超时时", 27.5, 29.45),
            self._mk("当我们的计时器超时时", 29.45, 30.1),
        ]
        out = ZipformerEngine._dedup_overlap(segs)
        assert len(out) == 1
        assert out[0].text == "然后当我们的计时器超时时"

    def test_chained_replace_then_dedup(self):
        """替换后与下一段继续链式去重 (后段替换前段后, 再与再后段比较)."""
        from app.core.zipformer_engine import ZipformerEngine
        segs = [
            self._mk("如果我们移除玩家的", 58.0, 59.4),
            self._mk("如果我们移除玩家的碰撞体", 58.3, 62.0),
            self._mk("如果我们移除玩家的碰撞体让它直接从地图上掉落", 58.5, 64.0),
        ]
        out = ZipformerEngine._dedup_overlap(segs)
        assert [(s.text) for s in out] == [
            "如果我们移除玩家的碰撞体让它直接从地图上掉落"
        ]


class TestBuildCharTimes:
    """_build_char_times: token/timestamps → 逐字符时间, 失败回退 None."""

    def test_valid_mapping(self):
        """合法 token 流 → char_times (与文本等长, 时间取起始字节所属 token)."""
        from app.core.bbpe_map import BYTE_TO_BCHAR
        from app.core.zipformer_engine import ZipformerEngine
        raw = "你好".encode("utf-8")
        tokens = [BYTE_TO_BCHAR[b] for b in raw]
        timestamps = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        ct = ZipformerEngine._build_char_times("你好", tokens, timestamps)
        assert ct == [1.0, 1.3]

    def test_decoded_text_mismatch_returns_none(self):
        """解码文本与传入 text 不一致 → None (时间戳不可信)."""
        from app.core.bbpe_map import BYTE_TO_BCHAR
        from app.core.zipformer_engine import ZipformerEngine
        raw = "你好".encode("utf-8")
        tokens = [BYTE_TO_BCHAR[b] for b in raw]
        ct = ZipformerEngine._build_char_times("你好吗", tokens, [1.0] * 6)
        assert ct is None

    def test_none_or_mismatched_inputs_return_none(self):
        """tokens/timestamps 缺失或长度不一致 → None."""
        from app.core.zipformer_engine import ZipformerEngine
        assert ZipformerEngine._build_char_times("abc", None, None) is None
        assert ZipformerEngine._build_char_times("abc", ["a", "b"], [1.0]) is None
        assert ZipformerEngine._build_char_times("", [], []) is None

    def test_magicmock_defaults_to_none(self):
        """MagicMock 段 (旧测试兼容) 的 tokens/timestamps 属性 → None, 不抛异常."""
        from app.core.zipformer_engine import ZipformerEngine
        from unittest.mock import MagicMock
        r = MagicMock()
        assert ZipformerEngine._build_char_times("abc", r.tokens, r.timestamps) is None
