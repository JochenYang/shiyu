"""
文本切分工具测试 (text_splitter)。

这里只覆盖云端引擎仍在使用的切分能力: CJK 判定与三级文本切分。
本地引擎不再走按字符数比例插值 —— 字幕行由 ``caption_builder`` 在已测量的
词边界上做最小代价 DP 切分, 因此旧的 ``split_text_at_boundaries`` 及其内部
辅助函数已从模块中移除, 相关测试也一并移除。
"""
from app.core.text_splitter import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHARS_EN,
    PAUSE_THRESHOLD,
    is_mostly_cjk,
    split_text_for_subtitles,
)


class TestIsMostlyCjk:
    def test_pure_chinese(self):
        assert is_mostly_cjk("这是一句中文") is True

    def test_pure_english(self):
        assert is_mostly_cjk("this is english") is False

    def test_empty_is_treated_as_cjk(self):
        assert is_mostly_cjk("") is True

    def test_mixed_above_threshold(self):
        # 4 个汉字 + 3 个 ASCII 字符 → 汉字占比 > 30%
        assert is_mostly_cjk("中文测试abc") is True


class TestSplitTextForSubtitles:
    def test_empty_input(self):
        assert split_text_for_subtitles("") == [""]

    def test_short_text_is_returned_as_one_piece(self):
        assert split_text_for_subtitles("短句", max_len=25) == ["短句"]

    def test_splits_at_sentence_punctuation(self):
        text = "第一句话。第二句话。第三句话。"
        parts = split_text_for_subtitles(text, max_len=DEFAULT_MAX_CHARS)
        assert len(parts) >= 3
        assert "".join(parts) == text

    def test_long_sentence_splits_at_commas(self):
        text = "这是一个很长的句子，里面有很多逗号，需要被切成多行，方便阅读，不能太长。"
        parts = split_text_for_subtitles(text, max_len=DEFAULT_MAX_CHARS)
        assert len(parts) > 1
        assert "".join(parts) == text

    def test_no_punctuation_falls_back_to_hard_cut(self):
        text = "没有任何标点的超长文本" * 5
        parts = split_text_for_subtitles(text, max_len=DEFAULT_MAX_CHARS)
        assert len(parts) > 1
        assert "".join(parts) == text
        assert all(len(p) <= DEFAULT_MAX_CHARS for p in parts)

    def test_english_splits_on_words(self):
        text = "the quick brown fox jumps over the lazy dog and keeps running"
        parts = split_text_for_subtitles(
            text, max_len=DEFAULT_MAX_CHARS_EN, is_cjk_flag=False
        )
        assert len(parts) > 1
        assert " ".join(parts) == text

    def test_text_is_never_lost(self):
        """切分不得丢字 (拼接回原文必须一致)。"""
        samples = [
            "中文没有标点的一长串内容需要被切分成多行显示",
            "Mixed 中英 混排 text with punctuation，需要正确处理。",
            "a" * 120,
        ]
        for text in samples:
            parts = split_text_for_subtitles(text, max_len=DEFAULT_MAX_CHARS)
            assert "".join(parts) == text, text


class TestSharedConstants:
    """共享常量的取值约定。"""

    def test_pause_threshold_is_shared_with_caption_builder(self):
        from app.core import caption_builder

        assert caption_builder.PAUSE_THRESHOLD == PAUSE_THRESHOLD

    def test_english_limit_is_wider_than_chinese(self):
        assert DEFAULT_MAX_CHARS_EN > DEFAULT_MAX_CHARS

    def test_removed_proportional_path_is_gone(self):
        """比例插值切分已移除, 不得再被重新引入。"""
        from app.core import text_splitter

        for name in (
            "split_text_at_boundaries",
            "_assign_timestamps",
            "_assign_timestamps_with_overlap",
            "_split_by_pauses",
        ):
            assert not hasattr(text_splitter, name), name
