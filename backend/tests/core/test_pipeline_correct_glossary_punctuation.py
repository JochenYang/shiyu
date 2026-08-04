"""
Shiyu Pipeline CORRECT 阶段回归测试 — glossary 统一应用 + 双标点修复.

覆盖:
  1. glossary (前端自定义词典) 非空时, 对所有段文本统一做 str.replace (引擎无关)
  2. deterministic_correct 成功后, pipeline 层不再跑独立 punctuation (防双标点)
  3. deterministic_correct 关闭时, pipeline 层仍跑独立 punctuation
  4. deterministic_correct 抛异常降级时, pipeline 层仍跑独立 punctuation
"""
from unittest.mock import MagicMock

import pytest


def _build_ctx(raw_segments, **kwargs):
    """构造 PipelineContext (默认参数 + 覆盖)."""
    from app.core.pipeline import PipelineContext
    defaults = dict(audio_path="/tmp/x.wav", raw_segments=raw_segments)
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class TestStageCorrectGlossary:
    """glossary (前端自定义词典) 对所有引擎结果统一生效."""

    def test_glossary_replaces_text_in_every_segment(self):
        """glossary 非空时, 每段文本都被 str.replace 替换."""
        from app.core.pipeline import SubtitlePipeline
        ctx = _build_ctx(
            [
                {"start": 0.0, "end": 2.0, "text": "巴蒂回调。"},
                {"start": 3.0, "end": 5.0, "text": "默认值巴蒂"},
            ],
            glossary={"巴蒂": "body"},
            # 关闭 deterministic 与 punctuation, 只观察 glossary 替换
            enable_deterministic_correction=False,
            enable_punctuation=False,
        )
        SubtitlePipeline(ctx)._stage_correct()
        assert ctx.corrected_segments[0]["text"] == "body回调。"
        assert ctx.corrected_segments[1]["text"] == "默认值body"

    def test_empty_glossary_leaves_text_untouched(self):
        """glossary 为空时, 文本保持原样."""
        from app.core.pipeline import SubtitlePipeline
        ctx = _build_ctx(
            [{"start": 0.0, "end": 2.0, "text": "巴蒂回调"}],
            glossary={},
            enable_deterministic_correction=False,
            enable_punctuation=False,
        )
        SubtitlePipeline(ctx)._stage_correct()
        assert ctx.corrected_segments[0]["text"] == "巴蒂回调"


class TestStageCorrectPunctuationOnce:
    """双标点回归: deterministic 成功则跳过 pipeline 层独立标点."""

    def _run_stage(self, monkeypatch, raw=None, **kwargs):
        # 先加载 asr_corrector, 使其模块级绑定的 get_punctuation_engine 保持真实引用
        # (deterministic 内部的 Phase 0 标点恢复走真实引擎)
        import app.core.asr_corrector  # noqa: F401
        from app.core.pipeline import SubtitlePipeline
        import app.core.punctuation as punct_module

        mock_engine = MagicMock()
        mock_engine.punctuate_segments.side_effect = lambda segs: segs
        monkeypatch.setattr(punct_module, "get_punctuation_engine", lambda: mock_engine)

        ctx = _build_ctx(
            raw or [{"start": 0.0, "end": 1.0, "text": "默认值九百八十"}],
            **kwargs,
        )
        SubtitlePipeline(ctx)._stage_correct()
        return ctx, mock_engine

    def test_deterministic_success_skips_pipeline_punctuation(self, monkeypatch):
        """deterministic 成功后, pipeline 层不再调用独立 punctuation (防双标点)."""
        ctx, mock_engine = self._run_stage(monkeypatch)
        assert mock_engine.punctuate_segments.call_count == 0
        assert ctx.corrected_segments[0]["text"]

    def test_deterministic_disabled_keeps_pipeline_punctuation(self, monkeypatch):
        """deterministic 关闭时, pipeline 层仍跑独立 punctuation."""
        ctx, mock_engine = self._run_stage(
            monkeypatch, enable_deterministic_correction=False,
        )
        assert mock_engine.punctuate_segments.call_count == 1
        assert ctx.corrected_segments[0]["text"] == "默认值九百八十"

    def test_deterministic_failure_falls_back_to_pipeline_punctuation(self, monkeypatch):
        """deterministic 抛异常降级时, pipeline 层仍跑独立 punctuation."""
        import app.core.asr_corrector as ac
        original = ac.deterministic_correct
        ac.deterministic_correct = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("mock fail")
        )
        try:
            ctx, mock_engine = self._run_stage(monkeypatch)
            assert mock_engine.punctuate_segments.call_count == 1
            assert ctx.corrected_segments[0]["text"] == "默认值九百八十"
        finally:
            ac.deterministic_correct = original
