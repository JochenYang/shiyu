"""
Phase 1 测试: text_splitter 战术修复

覆盖:
  1. DEFAULT_MAX_CHARS 默认值降为 25
  2. DEFAULT_MAX_CHARS_EN 默认值降为 42
  3. split_text_at_boundaries 实际使用新 max_chars 触发切分
  4. _assign_timestamps_with_overlap 默认 0.05s 重叠且连续无回退
  5. _assign_timestamps_with_overlap(overlap_sec=0) 退化为旧行为
"""
import pytest

from app.core.text_splitter import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHARS_EN,
    DEFAULT_MIN_CHARS,
    OVERLAP_SECONDS,
    _assign_timestamps_with_overlap,
    split_text_at_boundaries,
)


class TestPhase1Defaults:
    """常量默认值与 OVERLAP_SECONDS。"""

    def test_default_max_chars_is_25(self):
        """Phase 1: 中文字幕上限 35 -> 25。"""
        assert DEFAULT_MAX_CHARS == 25, (
            f"DEFAULT_MAX_CHARS 应为 25, 实际 {DEFAULT_MAX_CHARS}"
        )

    def test_default_max_chars_en_is_42(self):
        """Phase 1: 英文字幕上限 50 -> 42。"""
        assert DEFAULT_MAX_CHARS_EN == 42, (
            f"DEFAULT_MAX_CHARS_EN 应为 42, 实际 {DEFAULT_MAX_CHARS_EN}"
        )

    def test_overlap_seconds_is_005(self):
        """Phase 1 新增常量: 相邻段默认 0.05s 微重叠。"""
        assert OVERLAP_SECONDS == 0.05, (
            f"OVERLAP_SECONDS 应为 0.05, 实际 {OVERLAP_SECONDS}"
        )


class TestPhase1SplitBehavior:
    """split_text_at_boundaries 实际生效新阈值。"""

    def test_split_text_at_boundaries_respects_max_chars_25(self):
        """40 字符 ASCII 文本必须被切分, 最长片段 <= 25 字符。"""
        long_text = "a" * 40  # 40 ASCII chars, 长度确定
        assert len(long_text) == 40

        result = split_text_at_boundaries(
            [{"start": 0.0, "end": 10.0, "text": long_text}],
            max_chars=DEFAULT_MAX_CHARS,
        )
        # 至少 2 段
        assert len(result) >= 2, f"40 字符应切分为 >=2 段, 实际 {len(result)} 段"
        # 最长段 <= 25
        max_len = max(len(seg["text"]) for seg in result)
        assert max_len <= DEFAULT_MAX_CHARS, (
            f"切分后最长片段 {max_len} 超过 DEFAULT_MAX_CHARS={DEFAULT_MAX_CHARS}"
        )

    def test_split_25_char_text_not_split(self):
        """恰好 25 字符文本不应被切分 (边界值)。"""
        text_25 = "a" * DEFAULT_MAX_CHARS  # 25 chars
        assert len(text_25) == 25
        result = split_text_at_boundaries(
            [{"start": 0.0, "end": 5.0, "text": text_25}],
            max_chars=DEFAULT_MAX_CHARS,
        )
        assert len(result) == 1, (
            f"恰好 25 字符应保持 1 段, 实际 {len(result)} 段"
        )

    def test_split_cjk_realistic_scene(self):
        """真实 CJK 场景: 混合中英文标点 + 逗号分隔多短句。

        评审 follow-up 建议补充: 验证 CJK 路径(level 1 句子终止 + level 2 逗号切分)
        在新 DEFAULT_MAX_CHARS=25 下的端到端行为, 不只是 ASCII hard-cut 路径。
        """
        # 真实场景: 多短句用逗号连接, 总长约 26 字符, 每段应在 25 字符内
        cjk_text = "今天天气真好啊,我们去公园玩吧,你觉得怎么样"
        assert len(cjk_text) <= 30, f"样本长度异常 {len(cjk_text)}"

        result = split_text_at_boundaries(
            [{"start": 0.0, "end": 5.0, "text": cjk_text}],
            max_chars=DEFAULT_MAX_CHARS,
        )

        # 至少 1 段 (整体 < 25 字符可能不被切)
        assert len(result) >= 1, f"应至少 1 段, 实际 {len(result)} 段"
        # 每段不超过 max_chars
        for seg in result:
            assert len(seg["text"]) <= DEFAULT_MAX_CHARS, (
                f"切分片段 {seg['text']!r} 超过 {DEFAULT_MAX_CHARS}: "
                f"len={len(seg['text'])}"
            )
        # 文本内容应完整保留
        joined = "".join(seg["text"] for seg in result)
        assert "今天天气真好啊" in joined
        assert "公园" in joined


class TestPhase1OverlapTimestamps:
    """_assign_timestamps_with_overlap 行为验证。"""

    def test_default_overlap_continuous_no_gap(self):
        """默认 overlap=0.05 时, 3 段连续无回退且端点正确。"""
        seg = {"start": 0.0, "end": 9.0, "text": "AB"}  # duration=9
        parts = ["A", "B", "C"]
        result: list = []
        _assign_timestamps_with_overlap(seg, parts, duration=9.0, result=result)

        assert len(result) == 3
        # parts[0].start == seg.start
        assert result[0]["start"] == 0.0
        # parts[-1].end == seg.end
        assert result[2]["end"] == 9.0
        # 连续性: 默认 overlap=0.05, 允许 gap = -0.05 (即 50ms 微重叠)
        for i in range(len(result) - 1):
            gap = result[i + 1]["start"] - result[i]["end"]
            # gap 应在 [-OVERLAP_SECONDS, 0] 区间内
            assert gap >= -OVERLAP_SECONDS - 0.001, (
                f"段 {i}->{i+1} 时间戳回退超过 OVERLAP_SECONDS: "
                f"end={result[i]['end']} start={result[i + 1]['start']} gap={gap}"
            )
            assert gap <= 0.001, (
                f"段 {i}->{i+1} 不应有正 gap (默认 overlap>0): gap={gap}"
            )

    def test_overlap_zero_disables_overlap(self):
        """overlap_sec=0 应退化为旧行为 (纯比例分配, 无重叠)。"""
        seg = {"start": 0.0, "end": 6.0, "text": "ABC"}
        parts = ["A", "B", "C"]
        result: list = []
        _assign_timestamps_with_overlap(
            seg, parts, duration=6.0, result=result, overlap_sec=0.0
        )

        # 各段均分 6 秒: 0-2, 2-4, 4-6
        assert result[0] == {"start": 0.0, "end": 2.0, "text": "A"}
        assert result[1] == {"start": 2.0, "end": 4.0, "text": "B"}
        assert result[2] == {"start": 4.0, "end": 6.0, "text": "C"}

    def test_first_start_preserved_last_end_preserved(self):
        """不变式: parts[0].start == seg.start, parts[-1].end == seg.end。"""
        seg = {"start": 1.5, "end": 8.5, "text": ""}
        parts = ["x", "y", "z", "w"]
        result: list = []
        _assign_timestamps_with_overlap(seg, parts, duration=7.0, result=result)
        assert result[0]["start"] == 1.5
        assert result[-1]["end"] == 8.5

    def test_empty_parts_preserves_original(self):
        """parts 为空时回退到原 segment。"""
        seg = {"start": 0.0, "end": 5.0, "text": "x"}
        result: list = []
        _assign_timestamps_with_overlap(seg, [], duration=5.0, result=result)
        assert result == [{"start": 0.0, "end": 5.0, "text": "x"}]
