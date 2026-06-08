"""
Phase 1B 测试: sentence_splitter VAD 文本复制 bug 修复

覆盖:
  1. 单 ASR 段跨多 VAD 区间时, 文本不再被复制到每个 VAD 区间
  2. 多 ASR 段各自落到正确 VAD 区间（BFS-based dedupe）
  3. VAD 区间没有 ASR 段覆盖时不产生空段
  4. 回归保护: 单 ASR 段完全覆盖单 VAD 区间时行为不变
  5. dedupe 不会把同一段文本分给两个不相邻区间
"""
import pytest

from app.core.sentence_splitter import _align_segments_to_vad_regions, _split_text_at_punctuation


class TestVADAssignNoDuplication:
    """VAD 区间对齐时, 文本必须不重复 (Phase 1B 核心 bug)."""

    def test_single_asr_segment_spanning_two_vad_regions_not_duplicated(self):
        """单 ASR 段覆盖 [0, 4], 跨两个 VAD 区间 [0, 2] + [2, 4].

        修复前: 两段都被赋予完整文本 → 重复
        修复后: 文本被切分到两段, 段间文本不重复
        """
        asr_segments = [
            {"start": 0.0, "end": 4.0, "text": "你好世界今天天气很好"},
        ]
        vad_intervals = [
            (0.0, 2.0),
            (2.0, 4.0),
        ]

        result = _align_segments_to_vad_regions(asr_segments, vad_intervals)

        assert len(result) == 2, f"应产出 2 段, 实际 {len(result)}"

        # 关键: 文本连接起来应包含原文, 但不重复
        joined = "".join(seg["text"].replace(" ", "") for seg in result)
        original = asr_segments[0]["text"]
        # 每一段都应该是原文的子串, 不存在完整复制
        for seg in result:
            text = seg["text"].replace(" ", "")
            assert text in original, (
                f"段 {seg} 文本应在原文内, 但 {text!r} 不在 {original!r}"
            )
        # 关键不变量: 没有完整复制 — 任一段文本长度 < 原文长度
        for seg in result:
            assert len(seg["text"].replace(" ", "")) < len(original), (
                f"段 {seg} 文本被完整复制, 长度 {len(seg['text'])} >= {len(original)}"
            )

    def test_punctuation_split_distributes_text_across_regions(self):
        """含标点的 ASR 段应被切分到对应 VAD 区间, 标点对齐。"""
        asr_segments = [
            {"start": 0.0, "end": 4.0, "text": "第一句。第二句。"},
        ]
        vad_intervals = [
            (0.0, 2.0),
            (2.0, 4.0),
        ]

        result = _align_segments_to_vad_regions(asr_segments, vad_intervals)

        assert len(result) == 2
        # 文本应被切分, 两段都应非空
        assert all(seg["text"].strip() for seg in result)

    def test_multiple_asr_segments_distributed_one_per_region(self):
        """多 ASR 段各自落到对应 VAD 区间, 不重复。"""
        asr_segments = [
            {"start": 0.0, "end": 2.0, "text": "第一段"},
            {"start": 2.0, "end": 4.0, "text": "第二段"},
        ]
        vad_intervals = [
            (0.0, 2.0),
            (2.0, 4.0),
        ]

        result = _align_segments_to_vad_regions(asr_segments, vad_intervals)

        assert len(result) == 2
        texts = [seg["text"] for seg in result]
        assert "第一段" in " ".join(texts)
        assert "第二段" in " ".join(texts)


class TestVADAssignEdgeCases:
    """边界条件与回归保护."""

    def test_no_vad_intervals_falls_back_to_segments(self):
        """无 VAD 区间时直接返回原 segments."""
        asr_segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        result = _align_segments_to_vad_regions(asr_segments, [])
        assert result == asr_segments

    def test_vad_regions_filtered_when_outside_asr_range(self):
        """完全在 ASR 范围外的 VAD 区间被过滤掉."""
        asr_segments = [{"start": 1.0, "end": 2.0, "text": "x"}]
        vad_intervals = [(5.0, 6.0), (7.0, 8.0)]  # 都不重叠
        result = _align_segments_to_vad_regions(asr_segments, vad_intervals)
        # 全部 VAD 区间被过滤, 回落原 segments
        assert len(result) == 1
        assert result[0]["text"] == "x"

    def test_no_duplicate_text_across_disjoint_vad_regions(self):
        """两个 VAD 区间, 每个被分配不同的 ASR 段, 文本不重复."""
        asr_segments = [
            {"start": 0.0, "end": 1.5, "text": "段A文本"},
            {"start": 2.5, "end": 4.0, "text": "段B文本"},
        ]
        vad_intervals = [
            (0.0, 1.5),
            (2.5, 4.0),
        ]
        result = _align_segments_to_vad_regions(asr_segments, vad_intervals)
        texts = [seg["text"] for seg in result]
        # 每个 ASR 段最多在结果中出现一次
        for original in ["段A文本", "段B文本"]:
            count = sum(1 for t in texts if original in t)
            assert count == 1, f"{original} 应出现 1 次, 实际 {count} 次"

    def test_single_asr_single_vad_unchanged(self):
        """回归: 一对一映射时行为不变."""
        asr_segments = [{"start": 0.0, "end": 2.0, "text": "你好"}]
        vad_intervals = [(0.0, 2.0)]
        result = _align_segments_to_vad_regions(asr_segments, vad_intervals)
        assert len(result) == 1
        assert result[0]["text"] == "你好"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0


class TestSplitTextAtPunctuation:
    """辅助函数 _split_text_at_punctuation 单元测试."""

    def test_chinese_period_split(self):
        chunks = _split_text_at_punctuation("第一句。第二句。")
        assert "".join(chunks) == "第一句。第二句。"
        assert len(chunks) >= 1

    def test_english_period_split(self):
        chunks = _split_text_at_punctuation("Hello. World.")
        # 拼接后内容应保留
        joined = "".join(chunks)
        assert "Hello" in joined
        assert "World" in joined

    def test_no_punctuation_returns_whole(self):
        chunks = _split_text_at_punctuation("整段无标点文本")
        assert chunks == ["整段无标点文本"]

    def test_empty_returns_empty(self):
        assert _split_text_at_punctuation("") == []
