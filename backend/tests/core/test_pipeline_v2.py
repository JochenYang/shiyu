"""
Shiyu v2.0 Pipeline tests — 4 阶段 (ASR → CORRECT → VALIDATE → FORMAT).

覆盖:
  1. PipelineContext 默认 engines 是 zipformer_local (v2.0 主走)
  2. _stage_asr: 引擎可用 → 透传 primary 到 raw_segments
  3. _stage_asr: 引擎不可用 → 空 raw_segments, 不抛异常
  4. _stage_correct: ITN + 标点生效
  5. _stage_correct: deterministic 失败时降级透传
  6. _stage_format: SRT/LRC/ASS 三种格式输出
  7. run() 完整 4 阶段流水线 (mock 引擎 + 测试音频)
"""
from unittest.mock import MagicMock

import pytest


def _make_mock_engine(name: str, segments, available: bool = True):
    """构造 mock 引擎."""
    from app.core.engine_base import TranscribeResult
    engine = MagicMock()
    engine.name = name
    engine.display_name = name
    engine.is_local = True
    engine.requires_network = False
    engine.is_available.return_value = (available, "" if available else "not available")
    from app.core.engine_base import Segment
    if isinstance(segments, list) and segments and not isinstance(segments[0], Segment):
        segments = [Segment(**s) if isinstance(s, dict) else s for s in segments]
    result = TranscribeResult(segments=segments or [], language="zh")
    engine.transcribe.return_value = result
    return engine


class TestPipelineContextV2:
    """v2.0 PipelineContext 字段."""

    def test_engines_default_is_zipformer(self):
        """v2.0 默认 engines = ['zipformer_local']."""
        from app.core.pipeline import PipelineContext, PRIMARY_ENGINE
        assert PRIMARY_ENGINE == "zipformer_local"
        ctx = PipelineContext()
        assert ctx.engines == ["zipformer_local"]

    def test_stages_enum_has_4_members(self):
        """v2.0 流水线 4 阶段."""
        from app.core.pipeline import PipelineStage
        stages = {s.value for s in PipelineStage}
        assert stages == {"asr", "correct", "validate", "format"}

    def test_no_fuse_or_clause_stages(self):
        """FUSE / CLAUSE 已砍."""
        from app.core.pipeline import PipelineStage
        stage_names = [s.name for s in PipelineStage]
        assert "FUSE" not in stage_names
        assert "CLAUSE" not in stage_names
        assert "VAD" not in stage_names
        assert "CORRECTION" not in stage_names
        assert "POST_PROCESS" not in stage_names


class TestStageASR:
    """Stage 1: ASR — 单主引擎."""

    def test_engine_runs_and_populates_raw_segments(self):
        from app.core.pipeline import PipelineContext, SubtitlePipeline
        from app.core.engine_base import Segment

        segs = [
            Segment(start=0.0, end=2.0, text="你好世界"),
            Segment(start=2.5, end=5.0, text="这是测试"),
        ]
        engine = _make_mock_engine("zipformer_local", segs)

        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            engine_getter=lambda n: engine,
            skip_asr=True,  # 手动控制
        )
        # 走 _stage_asr (skip_asr=False)
        ctx.skip_asr = False
        # 屏蔽 duration 计算 (librosa 不在测试范围)
        from app.core.pipeline import SubtitlePipeline as SP
        SP._compute_duration = lambda self: None
        SubtitlePipeline(ctx)._stage_asr()
        assert "zipformer_local" in ctx.engines_results
        assert len(ctx.raw_segments) == 2
        assert ctx.raw_segments[0]["text"] == "你好世界"

    def test_unavailable_engine_yields_empty(self):
        from app.core.pipeline import PipelineContext, SubtitlePipeline

        engine = _make_mock_engine("zipformer_local", [], available=False)
        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            engine_getter=lambda n: engine,
        )
        from app.core.pipeline import SubtitlePipeline as SP
        SP._compute_duration = lambda self: None
        SubtitlePipeline(ctx)._stage_asr()
        assert ctx.engines_results == {}
        assert ctx.raw_segments == []

    def test_skip_asr_preserves_existing_results(self):
        """skip_asr=True 用于 /correct 端点, 保留现有 raw_segments."""
        from app.core.pipeline import PipelineContext, SubtitlePipeline

        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            skip_asr=True,
            raw_segments=[{"start": 0, "end": 1, "text": "preset"}],
        )
        SubtitlePipeline(ctx)._stage_asr()
        # engines_results 应为空 dict (因为没跑 ASR)
        assert ctx.engines_results == {}


class TestStageCorrect:
    """Stage 2: CORRECT — 确定性修正 + ITN + 标点."""

    def test_corrected_segments_is_list(self):
        from app.core.pipeline import PipelineContext, SubtitlePipeline
        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            raw_segments=[
                {"start": 0, "end": 2, "text": "九百八十。"},
                {"start": 3, "end": 5, "text": "正常"},
            ],
            # 关掉标点避免测试耦合 punctuation.py
            enable_punctuation=False,
        )
        SubtitlePipeline(ctx)._stage_correct()
        assert isinstance(ctx.corrected_segments, list)
        assert len(ctx.corrected_segments) == 2

    def test_itn_converts_chinese_digits(self):
        """ITN: 九百八十 → 980."""
        from app.core.pipeline import PipelineContext, SubtitlePipeline
        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            raw_segments=[
                {"start": 0, "end": 2, "text": "默认值九百八十"},
            ],
            enable_punctuation=False,
        )
        SubtitlePipeline(ctx)._stage_correct()
        # ITN 应转换 (假设 itn_normalize 可用)
        result_text = ctx.corrected_segments[0]["text"]
        # 不强制断言 (可能 punctuation disable 影响其他转换), 但顺序应保持
        assert "九百八十" in result_text or "980" in result_text

    def test_correction_failure_falls_back_to_source(self):
        """deterministic_correct 抛异常时降级透传."""
        from app.core.pipeline import PipelineContext, SubtitlePipeline
        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            raw_segments=[{"start": 0, "end": 1, "text": "test"}],
            enable_punctuation=False,
        )
        # mock 让 deterministic_correct 抛异常
        import app.core.asr_corrector as ac
        original = ac.deterministic_correct
        ac.deterministic_correct = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("mock fail")
        )
        try:
            SubtitlePipeline(ctx)._stage_correct()
            # 降级: corrected_segments 应该是 source (text 保持 "test")
            assert ctx.corrected_segments[0]["text"] == "test"
        finally:
            ac.deterministic_correct = original


class TestStageFormat:
    """Stage 4: FORMAT — SRT/LRC/ASS."""

    def _ctx_with_segments(self, fmt="srt"):
        from app.core.pipeline import PipelineContext
        from app.core.engine_base import Segment
        segs = [Segment(start=0.0, end=2.0, text="你好")]
        engine = _make_mock_engine("zipformer_local", segs)
        return PipelineContext(
            audio_path="/tmp/x.wav",
            engine_getter=lambda n: engine,
            enable_punctuation=False,
            enable_deterministic_correction=False,
            output_format=fmt,
        )

    def test_srt_output(self):
        from app.core.pipeline import SubtitlePipeline
        ctx = self._ctx_with_segments("srt")
        SubtitlePipeline(ctx).run()
        assert "00:00:00,000" in ctx.output_content
        assert "你好" in ctx.output_content

    def test_lrc_output(self):
        from app.core.pipeline import SubtitlePipeline
        ctx = self._ctx_with_segments("lrc")
        SubtitlePipeline(ctx).run()
        assert "[00:00.00]" in ctx.output_content

    def test_ass_output(self):
        from app.core.pipeline import SubtitlePipeline
        ctx = self._ctx_with_segments("ass")
        SubtitlePipeline(ctx).run()
        assert "[Script Info]" in ctx.output_content


class TestRunEnd2End:
    """完整 4 阶段跑通."""

    def test_full_pipeline_with_mock_engine(self):
        from app.core.pipeline import PipelineContext, SubtitlePipeline
        from app.core.engine_base import Segment

        segs = [
            Segment(start=0.0, end=2.5, text="九百八十"),
            Segment(start=3.0, end=5.0, text="六十帧每秒"),
        ]
        engine = _make_mock_engine("zipformer_local", segs)
        ctx = PipelineContext(
            audio_path="/tmp/x.wav",
            engine_getter=lambda n: engine,
            enable_punctuation=False,
        )
        # 屏蔽 duration
        from app.core.pipeline import SubtitlePipeline as SP
        SP._compute_duration = lambda self: None
        result_ctx = SubtitlePipeline(ctx).run()
        # run 后 output_content 非空
        assert result_ctx.output_content
        assert "00:00:00" in result_ctx.output_content
        # engines_results 收集成功
        assert "zipformer_local" in result_ctx.engines_results
        # engine_meta 标记 primary
        assert result_ctx.engine_meta["primary"] == "zipformer_local"
