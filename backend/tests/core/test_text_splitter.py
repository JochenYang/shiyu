"""
text_splitter 测试 — _find_sentence_splits 句界识别。

覆盖:
  1. 中文省略号 '……' (连续 2+ 个 U+2026) 作为句界, split 点在最后一个省略号后
  2. 英文句点切分 ('Hello world. Next.')
  3. 小数点不切分 ('3.5 版本')
"""
from app.core.text_splitter import _find_sentence_splits


class TestFindSentenceSplits:
    """_find_sentence_splits 句界识别."""

    def test_chinese_ellipsis_is_sentence_boundary(self):
        """中文省略号 '……' 作为句界, split_after 位于最后一个省略号之后."""
        assert _find_sentence_splits("第一句……第二句") == [5]

    def test_english_period_splits_sentences(self):
        """英文句点 (后跟空格或行尾) 切分句子."""
        assert _find_sentence_splits("Hello world. Next.") == [12, 18]

    def test_decimal_point_not_split(self):
        """小数点两侧是数字时不切分."""
        assert _find_sentence_splits("3.5 版本") == []


# ─────────────────────────────────────────────────────────────────────
# _split_by_pauses 停顿切分 (zipformer 无标点场景)
# ─────────────────────────────────────────────────────────────────────

from app.core.text_splitter import (
    PAUSE_THRESHOLD,
    _assign_timestamps_with_overlap,
    _split_by_pauses,
)
import pytest


def _seg_with_char_times(text: str, times: list):
    """构造带 char_times 的段 (字符数与文本等长)."""
    assert len(text) == len(times)
    return {"start": times[0], "end": times[-1] + 1.0, "text": text, "char_times": times}


def _times_for(text: str, gap_at: list, step: float = 0.1):
    """构造字符时间戳: 默认步长 0.1s, 指定位置 (两字符之间) 加大停顿."""
    times = [0.0]
    for i in range(1, len(text)):
        gap = PAUSE_THRESHOLD + 0.5 if (i - 1) in gap_at else step
        times.append(times[-1] + gap)
    return times


class TestSplitByPauses:
    """停顿切分: 逐字符时间戳的停顿点优先切分, 过短段按 MIN_PAUSE_FRAGMENT 合并."""

    def test_no_split_when_short_enough(self):
        """文本 ≤ max_chars 时不做停顿切分 (返回 None, 由调用方保持原样)."""
        text = "这是一段不超过二十五字的普通台词"
        seg = _seg_with_char_times(text, _times_for(text, []))
        assert _split_by_pauses(seg, 25) is None

    def test_split_at_pauses(self):
        """长文本在停顿点 (gap > 阈值) 处切分, 无停顿的超长段硬切到 max_chars."""
        text = "第一句话在这里说完" + "第二句话在这里说完" + "第三句话在这里说完"
        times = _times_for(text, [8, 17])  # 第 9、18 字后各有一个停顿
        seg = _seg_with_char_times(text, times)
        parts = _split_by_pauses(seg, 25)
        assert parts == ["第一句话在这里说完", "第二句话在这里说完", "第三句话在这里说完"]

    def test_short_pause_fragment_merged_into_prev(self):
        """停顿段过短 (< MIN_PAUSE_FRAGMENT) 时并入前段, 不产生碎字幕."""
        text = "这里是一段比较长的第一句" + "短句" + "这里是一段比较长的第二句"
        times = _times_for(text, [11, 13])  # 第一句后、短句后各一个停顿
        seg = _seg_with_char_times(text, times)
        parts = _split_by_pauses(seg, 25)
        # "短句" 2 字 < 7 → 并入前段
        assert parts == ["这里是一段比较长的第一句短句", "这里是一段比较长的第二句"]

    def test_merged_back_to_single_returns_none(self):
        """全部停顿段都过短被合并回一段 → 返回 None, 调用方回退硬切."""
        text = "甲乙丙丁戊己庚辛"
        times = _times_for(text, [0, 1, 2, 3, 4, 5, 6])  # 每字后停顿, 8 个 1 字段
        seg = _seg_with_char_times(text, times)
        assert _split_by_pauses(seg, 5) is None

    def test_overlong_pause_part_hard_cut(self):
        """停顿段自身超 max_chars (内部无停顿的长口语) → 仍硬切到 max_chars."""
        text = "一二三四五六七八九十" + "这是一段内部没有停顿所以非常长的话"
        times = _times_for(text, [9])  # 第 10 字后停顿, 后段 17 字无停顿
        seg = _seg_with_char_times(text, times)
        parts = _split_by_pauses(seg, 10)
        # 前段 10 字整, 后段 17 字 > 10 → 硬切为 10 + 7
        assert parts == [
            "一二三四五六七八九十",
            "这是一段内部没有停顿所以非常长的话"[:10],
            "这是一段内部没有停顿所以非常长的话"[10:],
        ]

    def test_invalid_char_times_returns_none(self):
        """char_times 缺失或长度不匹配 → None."""
        text = "这是一个超过十个字符的长文本内容"
        assert _split_by_pauses({"start": 0, "end": 1, "text": text}, 10) is None
        seg = {"start": 0, "end": 1, "text": text, "char_times": [0.0, 0.1]}
        assert _split_by_pauses(seg, 10) is None

    def test_assign_timestamps_uses_char_times(self):
        """精确分配: 段首/段尾取 char_times 真实时间, 非比例插值."""
        text = "第一句比较长的台词" + "第二句比较长的内容"
        times = _times_for(text, [8])  # 第 9 字后停顿 (两句边界)
        seg = _seg_with_char_times(text, times)
        parts = _split_by_pauses(seg, 12)
        assert parts == ["第一句比较长的台词", "第二句比较长的内容"]
        result = []
        _assign_timestamps_with_overlap(seg, parts, seg["end"] - seg["start"], result)
        # 首段: start = 首字时间, end = 末字时间; 末段: end = 段尾
        # (实现 round(.., 3), 浮点累加误差用 approx 容忍)
        assert result[0]["start"] == pytest.approx(times[0])
        assert result[0]["end"] == pytest.approx(times[8])
        assert result[1]["start"] == pytest.approx(times[9])
        assert result[1]["end"] == pytest.approx(seg["end"])
