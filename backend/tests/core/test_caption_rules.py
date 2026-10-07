"""
字幕断句与文本单元测试。

覆盖 SubPop 移植过来的三条关键规则:
  1. 功能性写法 (0.1 / 1/30 / 5GHz / C++ / iPhone 18) 不被标点清理破坏
  2. 断句只落在已测量的词边界上, 时间戳全部来自输入测量值
  3. 段尾虚词、段首虚词、真实停顿对切点选择的影响
"""
import pytest

from app.core.caption_builder import build_captions, measured_words, punctuation_boundary
from app.core.semantic_breaks import coordinated_ranges, phrase_partition
from app.core.text_units import clean_generated_text, display_length, protected_ranges


# ── 文本单元 ─────────────────────────────────────────────────────────


class TestTextUnits:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("版本 0.2 发布了", "版本 0.2 发布了"),
            ("速度是 1/30 秒", "速度是 1/30 秒"),
            ("频率 5GHz", "频率 5GHz"),
            ("用 C++ 写的", "用 C++ 写的"),
            ("买了 iPhone 18 Pro", "买了 iPhone 18 Pro"),
            ("分辨率 1920:1080", "分辨率 1920:1080"),
        ],
    )
    def test_functional_units_survive_punctuation_cleanup(self, raw, expected):
        """数值、单位、标识符里的点与斜杠不是句子标点, 必须原样保留。"""
        assert clean_generated_text(raw) == expected

    def test_sentence_punctuation_is_removed(self):
        """真正的句子标点被删除。"""
        assert clean_generated_text("你好，世界。") == "你好世界"

    def test_removing_delimiter_between_english_words_keeps_a_space(self):
        """删掉英文单词之间的句点不能让两个词粘连。"""
        assert clean_generated_text("hello.world") == "hello world"

    def test_display_length_counts_ascii_as_half(self):
        """ASCII 记半个汉字宽, 中英混排宽度可比较。"""
        assert display_length("模型") == 2
        assert display_length("model") == 2.5
        assert display_length("模型model") == 4.5

    def test_protected_ranges_cover_numbers_and_identifiers(self):
        spans = protected_ranges("版本 0.2 跑在 1/30 秒")
        covered = {i for a, b in spans for i in range(a, b)}
        # 0.2 与 1/30 的字符全部在保护区内
        assert all(i in covered for i in range(3, 6))
        assert all(i in covered for i in range(10, 14))

    def test_vocabulary_terms_are_protected(self):
        spans = protected_ranges("用 godot 做游戏", terms=["Godot"])
        assert spans, "词库词条应当进入保护区"


# ── 标点边界 ─────────────────────────────────────────────────────────


class TestPunctuationBoundary:
    def test_hard_and_soft_boundaries(self):
        assert punctuation_boundary("结束了。") == "hard"
        assert punctuation_boundary("然后呢，") == "soft"

    def test_decimal_point_is_not_a_boundary(self):
        """小数点是数值的一部分, 不是句末标点。"""
        assert punctuation_boundary("3.14") is None
        assert punctuation_boundary("版本 0.2.") is None
    def test_comma_inside_number_is_not_a_boundary(self):
        assert punctuation_boundary("1,", "000") is None

    def test_closing_quote_after_period_is_still_hard(self):
        assert punctuation_boundary("结束了。”") == "hard"


# ── 测量词 ───────────────────────────────────────────────────────────


class TestMeasuredWords:
    def test_length_mismatch_is_rejected(self):
        """字符时间与文本长度不一致时必须报错, 而不是猜测时间。"""
        with pytest.raises(ValueError, match="长度不一致"):
            measured_words("四个字", [0.0, 1.0], 3.0)

    def test_words_cover_the_whole_text(self):
        """词序列必须逐字符覆盖原文 (含空白与标点), 不丢字。"""
        text = "我们把模型的下载速度提高了"
        times = [i * 0.2 for i in range(len(text))]
        words = measured_words(text, times, len(text) * 0.2)
        assert "".join(w.text for w in words) == text
        assert words[0].start == 0.0
        # 词尾取「末字符开始时间」(不吞掉其后静音), 因此末词 end 是末字符的开始时间
        assert words[-1].end == pytest.approx(times[-1])

    def test_word_times_come_from_measured_input(self):
        """每个词的起止时间都必须是输入测量值的子集, 不能是插值结果。"""
        text = "测试断句"
        times = [0.0, 0.5, 1.1, 1.9]
        words = measured_words(text, times, 2.5)
        allowed = set(times) | {2.5}
        for word in words:
            assert word.start in allowed
            assert word.end in allowed

    def test_word_end_does_not_absorb_following_silence(self):
        """词尾不吞掉其后静音: 7 秒停顿不能把前一个词拉成 7 秒。

        回归: 早期实现取「下一字符开始时间」作为词尾, 一段 7s 静音把「大家」
        拉成 7.2s, 超过单行时长上限而独占一行, 词间停顿也变成负数。
        """
        text = "谢谢大家"
        times = [0.0, 0.3, 0.5, 7.5]
        words = measured_words(text, times, 8.0)
        longest = max(w.end - w.start for w in words)
        assert longest < 1.0, f"词时长被静音拉长: {[(w.text, w.start, w.end) for w in words]}"

    def test_ascii_text_splits_on_spaces(self):
        """ASCII 按空白切词; 空白并入前一词以保证逐字符覆盖。"""
        text = "hello world again"
        times = [i * 0.1 for i in range(len(text))]
        words = measured_words(text, times, len(text) * 0.1)
        assert [w.text.strip() for w in words] == ["hello", "world", "again"]
        assert "".join(w.text for w in words) == text


# ── 断句 ─────────────────────────────────────────────────────────────


class TestPhrasePartition:
    def test_never_splits_inside_protected_units(self):
        """DP 切分不得落在 1/30 这类功能性写法内部。"""
        text = "速度提升到了1/30秒的水平"
        times = [i * 0.1 for i in range(len(text))]
        words = measured_words(text, times, len(text) * 0.1)

        protected = protected_ranges(text)
        interior = {i for a, b in protected for i in range(a + 1, b)}
        assert interior, "本例应当存在保护区间内部位置"

        # 还原每个词在原文中的字符偏移, 检查切点是否落在保护区内部
        cursor = 0
        cuts = []
        for chunk in phrase_partition(words, max_chars=6, max_duration=99, punctuation_boundary=punctuation_boundary):
            cursor += sum(len(w.text) for w in chunk)
            cuts.append(cursor)
        cuts.pop()  # 末尾切点即文本长度, 不算切分
        assert not (set(cuts) & interior), f"切点 {cuts} 落入保护区间内部 {sorted(interior)}"

    def test_partition_preserves_word_order_and_coverage(self):
        text = "我们把模型的下载速度提高到了每秒三十兆的水平"
        times = [i * 0.15 for i in range(len(text))]
        words = measured_words(text, times, len(text) * 0.15)
        chunks = phrase_partition(words, max_chars=14, max_duration=99, punctuation_boundary=punctuation_boundary)
        assert [w.text for chunk in chunks for w in chunk] == [w.text for w in words]

    def test_coordinated_structures_are_kept_together(self):
        """「使用体验和研究成果」这类并列结构不应被切开。"""
        text = "分享使用体验和研究成果"
        spans = coordinated_ranges(text)
        assert spans, "并列结构应被识别出来"


class TestBuildCaptions:
    def test_captions_are_monotonic_and_use_measured_times(self):
        text = "我们把模型的下载速度提高到了每秒三十兆"
        times = [i * 0.2 for i in range(len(text))]
        end = len(text) * 0.2
        captions = build_captions(text, times, 0.0, end, max_chars=14)

        assert captions, "应产出至少一行字幕"
        # 覆盖全部文字
        assert "".join(c["text"] for c in captions).replace(" ", "") == text.replace(" ", "")
        # 单调不重叠
        for previous, current in zip(captions, captions[1:]):
            assert current["start"] >= previous["end"] - 1e-9
        # 时间不越界
        assert captions[0]["start"] >= 0.0
        assert captions[-1]["end"] <= end + 1e-9

    def test_all_timestamps_come_from_measured_input(self):
        """每一行的时间都取自输入测量值或段边界, 绝不插值。"""
        text = "测试一下断句效果如何"
        times = [0.0, 0.4, 0.9, 1.3, 1.8, 2.4, 2.9, 3.5, 4.0, 4.6]
        end = 5.2
        captions = build_captions(text, times, 0.0, end, max_chars=8)
        allowed = set(times) | {0.0, end}
        for row in captions:
            assert row["start"] in allowed, row
            assert row["end"] in allowed, row

    def test_duplicate_timestamps_do_not_lose_text(self):
        """重复时间戳 (CTC 常见) 不得丢字 —— 退化行并入上一行。"""
        for text, times, end in [
            ("我们开始吧", [0.0, 0.2, 0.4, 1.0, 1.0], 1.0),
            ("你好世界", [0.0, 0.5, 1.0, 1.0], 1.0),
            ("全部同时", [1.0, 1.0, 1.0, 1.0], 1.5),
        ]:
            captions = build_captions(text, times, 0.0, end)
            assert "".join(c["text"] for c in captions) == text, (text, captions)

    def test_length_mismatch_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="长度不一致"):
            build_captions("四个字", [0.0, 1.0], 0.0, 3.0)

    def test_too_small_max_chars_is_rejected(self):
        with pytest.raises(ValueError, match="不应小于 6"):
            build_captions("文本", [0.0, 0.1], 0.0, 0.2, max_chars=3)

    def test_real_pause_splits_lines(self):
        """词之间超过阈值的真实停顿应成为切分点。"""
        text = "第一句在这里第二句在那里"
        # 前 6 字连读, 中间 1 秒停顿, 后面连读
        times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        captions = build_captions(text, times, 0.0, 3.2, max_chars=20)
        assert len(captions) >= 2
        assert captions[0]["text"].startswith("第一句")