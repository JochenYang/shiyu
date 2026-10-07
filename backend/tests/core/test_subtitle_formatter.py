"""
Subtitle formatter tests — 中文/英文换行与段合并的格式化行为 (P1 修复).

覆盖:
  1. _split_long_lines: 中文长文本按字符切分, 行内不插空格, 行间 \\N 连接
  2. _split_long_lines: 英文按单词切分, 词间保留空格
  3. segments_from_transcription: 中文段合并直接拼接, 不加空格
  4. segments_from_transcription: 英文段合并保留空格分隔
  5. generate_srt: 基础输出 (BOM / 序号 / 时间轴 / 空行)
"""
from app.core.subtitle_formatter import (
    SubtitleSegment,
    _split_long_lines,
    generate_srt,
    segments_from_transcription,
)


class TestSplitLongLines:
    """_split_long_lines 中文/英文换行行为."""

    def test_cjk_long_text_breaks_by_char_without_spaces(self):
        """中文长文本按字符切分, 行内不插空格, 内容不丢失."""
        text = "这是一段用于测试字幕自动换行的中文长文本内容"
        result = _split_long_lines(text, max_length=10, max_lines=2)
        lines = result.split("\\N")
        assert len(lines) == 2
        assert all(" " not in line for line in lines)
        # 换行仅改变展示, 不丢失/不插入任何字符
        assert "".join(lines) == text

    def test_cjk_short_text_untouched(self):
        """中文短文本不超过 max_length 时原样返回."""
        text = "短文本"
        assert _split_long_lines(text) == text

    def test_english_long_text_breaks_by_word_keeping_spaces(self):
        """英文长文本按单词切分, 词间保留单个空格, 拼接还原原文."""
        text = "This is a fairly long English sentence that should be split"
        result = _split_long_lines(text, max_length=20, max_lines=2)
        lines = result.split("\\N")
        assert len(lines) == 2
        for line in lines:
            assert line == line.strip()  # 行首行尾无多余空格
        # 单词级切分: 按空格重新拼接与原文一致
        assert " ".join(lines) == text


class TestSegmentsFromTranscription:
    """segments_from_transcription 不再合并相邻段。

    字幕行已由引擎在测量词边界上断好; 格式化层再按间隔合并会把断句结果
    重新打回一整行 (回归: 3 行曾被拼成 4.65s 一条)。
    """

    def test_adjacent_cjk_segments_are_not_merged(self):
        segments_raw = [
            {"start": 0.0, "end": 0.8, "text": "第一段"},
            {"start": 0.9, "end": 2.0, "text": "第二段"},
        ]
        result = segments_from_transcription(segments_raw)
        assert len(result) == 2
        assert [s.text for s in result] == ["第一段", "第二段"]

    def test_adjacent_english_segments_are_not_merged(self):
        segments_raw = [
            {"start": 0.0, "end": 0.8, "text": "First part"},
            {"start": 0.9, "end": 2.0, "text": "second part"},
        ]
        result = segments_from_transcription(segments_raw)
        assert len(result) == 2
        assert [s.text for s in result] == ["First part", "second part"]

    def test_short_segments_are_not_merged(self):
        """极短段也保留 —— 不再因「读不完」而被吞进邻段。"""
        segments_raw = [
            {"start": 0.0, "end": 0.5, "text": "嗯"},
            {"start": 0.5, "end": 2.0, "text": "我想说的是"},
        ]
        result = segments_from_transcription(segments_raw)
        assert len(result) == 2

    def test_times_are_preserved_exactly(self):
        segments_raw = [{"start": 0.01, "end": 1.11, "text": "对我做了介绍"}]
        result = segments_from_transcription(segments_raw)
        assert result[0].start == 0.01
        assert result[0].end == 1.11

    def test_empty_text_is_dropped(self):
        segments_raw = [
            {"start": 0.0, "end": 1.0, "text": "  "},
            {"start": 1.0, "end": 2.0, "text": "有内容"},
        ]
        result = segments_from_transcription(segments_raw)
        assert [s.text for s in result] == ["有内容"]


class TestGenerateSrt:
    """generate_srt 基础输出."""

    def test_basic_output_structure(self):
        """SRT 输出包含 BOM、序号、时间轴与空行分隔."""
        segments = [
            SubtitleSegment(start=0.0, end=2.5, text="你好世界"),
            SubtitleSegment(start=3.0, end=5.25, text="Hello world"),
        ]
        srt = generate_srt(segments)
        assert srt.startswith("\ufeff")
        lines = srt.splitlines()
        assert lines[0] == "\ufeff1"
        assert lines[1] == "00:00:00,000 --> 00:00:02,500"
        assert lines[2] == "你好世界"
        assert lines[3] == ""
        assert lines[4] == "2"
        assert lines[5] == "00:00:03,000 --> 00:00:05,250"
        assert lines[6] == "Hello world"
