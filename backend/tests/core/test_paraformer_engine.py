"""
Paraformer 分块重叠去重逻辑测试 (纯函数, 不加载真实模型).

背景: 长音频 30s 硬分块, 块间加 2s 上下文重叠避免句子在块边界截断;
Paraformer 每块只输出 1 个长段 (CIF 无 VAD 切分), 重叠区会被转录两次,
拼接时必须按绝对时间戳去重 (covered_end 过滤).

覆盖:
  1. _merge_chunks_segments 纯函数:
     - 完全落在重叠区的段被丢弃
     - 跨边界段 (起点在重叠区、终点超出) 裁剪起点保留句尾, 不丢内容
     - 容差窗口 (50ms) 内的跨边界段保留
     - 整块落在重叠区 → 跳过该块, covered_end 不推进
     - 一块内多段时只保留不重叠的部分
     - 无重叠时全部保留, 空输入返回空
  2. 模块 import 安全 (sherpa-onnx 缺失也能 import, 纯函数可测的前提)
"""
from __future__ import annotations

from app.core.engine_base import Segment
from app.core.paraformer_engine import _merge_chunks_segments


class TestMergeChunksSegments:
    """分块拼接重叠去重 (构造有重叠时间戳的段列表验证)."""

    def test_cross_boundary_segment_is_clipped_not_dropped(self):
        """第二块段起点落在重叠区但终点超出 → 裁剪起点, 句尾内容不丢失."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [Segment(start=28.5, end=58.0, text="第二块(跨边界)")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "第二块(跨边界)"]
        assert merged[1].start == 30.0
        assert merged[1].end == 58.0

    def test_segment_fully_in_overlap_region_is_dropped(self):
        """第二块段完全落在重叠区 (end <= covered_end) → 整段丢弃."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [Segment(start=28.0, end=29.5, text="整段重叠")],
            [Segment(start=31.0, end=58.0, text="第三块")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "第三块"]

    def test_non_overlap_segment_is_kept(self):
        """段起点在 covered_end 之后 → 保留, 拼接顺序不变."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [Segment(start=30.2, end=58.0, text="第二块")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "第二块"]
        assert merged[1].start == 30.2

    def test_segment_within_tolerance_window_is_kept(self):
        """起点落在 covered_end-0.05 容差窗口内 → 保留 (吸收 CIF 边界噪声)."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [Segment(start=29.96, end=58.0, text="第二块(跨边界)")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "第二块(跨边界)"]

    def test_skipped_chunk_must_not_advance_covered_end(self):
        """被丢弃的整段重叠块若推进 covered_end, 后续跨边界段会被误杀."""
        # 第二块 [29, 29.5] 完全重叠 → 丢弃且不推进;
        # 第三块 [29.90, 58] 跨边界 → 裁剪为 [30, 58] 保留
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [Segment(start=29.0, end=29.5, text="重叠块")],
            [Segment(start=29.90, end=58.0, text="第三块(跨边界)")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "第三块(跨边界)"]
        assert merged[1].start == 30.0

    def test_clipped_segment_updates_covered_end(self):
        """裁剪保留的跨边界段会推进 covered_end, 防止后续块重复保留."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [Segment(start=28.0, end=58.0, text="第二块(裁剪)")],
            [Segment(start=57.5, end=90.0, text="第三块(再裁剪)")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "第二块(裁剪)", "第三块(再裁剪)"]
        assert merged[1].start == 30.0
        assert merged[2].start == 58.0

    def test_multi_segment_chunk_keeps_only_non_overlap(self):
        """一块内多段时, 只保留不落在重叠区的段."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="第一块")],
            [
                Segment(start=28.0, end=29.0, text="重叠段"),
                Segment(start=31.0, end=45.0, text="新内容"),
            ],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["第一块", "新内容"]

    def test_no_overlap_all_kept(self):
        """块间无重叠 → 全部保留且保持顺序."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="a")],
            [Segment(start=30.0, end=60.0, text="b")],
            [Segment(start=60.0, end=90.0, text="c")],
        ]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["a", "b", "c"]

    def test_empty_chunk_and_empty_input(self):
        """空输入 / 空块 (解码失败) 都返回空或跳过, 不报错."""
        assert _merge_chunks_segments([]) == []
        chunks = [[], [Segment(start=5.0, end=30.0, text="a")], []]
        merged = _merge_chunks_segments(chunks)
        assert [s.text for s in merged] == ["a"]

    def test_custom_tolerance_changes_keep_decision(self):
        """容差可调: tolerance=0 时起点略早于 covered_end 且终点未超出的段被丢弃."""
        chunks = [
            [Segment(start=0.0, end=30.0, text="a")],
            [Segment(start=29.98, end=30.0, text="b")],
        ]
        assert len(_merge_chunks_segments(chunks, overlap_tolerance=0.0)) == 1
        assert len(_merge_chunks_segments(chunks, overlap_tolerance=0.05)) == 2


class TestParaformerModuleImportSafety:
    """模块在 sherpa-onnx 缺失时仍可 import (纯函数单测的前提)."""

    def test_top_level_import_does_not_require_sherpa_onnx(self):
        """顶层 import 不触发 sherpa_onnx 加载.

        sherpa-onnx 只在 _ensure_loaded / transcribe 内按需导入,
        纯函数与注册逻辑不依赖它, 保证无模型环境可 import 并跑测试.
        """
        import app.core.paraformer_engine as pf
        assert "sherpa_onnx" not in dir(pf)
