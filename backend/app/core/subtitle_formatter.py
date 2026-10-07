"""
Subtitle format generators: SRT, LRC, ASS.
"""
from typing import List, Dict
from dataclasses import dataclass

from .text_splitter import is_mostly_cjk


@dataclass
class SubtitleSegment:
    start: float  # seconds
    end: float    # seconds
    text: str


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_lrc_time(seconds: float) -> str:
    """Convert seconds to LRC time format: [mm:ss.xx]"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    hundredths = int((seconds - int(seconds)) * 100)
    return f"[{minutes:02d}:{secs:02d}.{hundredths:02d}]"


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format: H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _strip_leading_punct(text: str) -> str:
    """去掉行首的句读标点。

    逐行字幕接续上一行时, CORRECT 阶段的段首连词规则可能在行首留
    逗号 ("…，" + "，然后…"), 观感差且不影响语义 —— 标点只在行尾
    有意义。已生成的旧 SRT 重新导出时也走这里, 无需重新转录。
    """
    return text.lstrip("，、；：,;:.。！？!？… ")


def _split_long_lines(text: str, max_length: int = 40, max_lines: int = 2) -> str:
    """将长文本切分为多行字幕（行间用 \\N 连接）。

    中文（CJK 占比 >30%）按字符切分、行内不插空格；英文按单词切分、词间保留空格。
    When text exceeds max_lines, remaining content is appended to the last line
    rather than discarded, ensuring no content is lost.
    """
    if len(text) <= max_length:
        return text

    is_cjk = is_mostly_cjk(text)
    # 中文按字符切分；英文优先按单词切分，无空格时退化为字符级
    words = text.split(" ") if (not is_cjk and " " in text) else list(text)
    sep = "" if is_cjk else " "
    lines = []
    current_line = ""
    overflow = False

    for word in words:
        if overflow:
            # Append remaining words to the last line
            lines[-1] = lines[-1] + sep + word
            continue
        if len(current_line) + len(word) + (len(sep) if current_line else 0) <= max_length:
            current_line += (sep + word if current_line else word)
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
            if len(lines) >= max_lines:
                # Exceeded max lines: start appending to last line
                lines[-1] = lines[-1] + sep + current_line
                overflow = True

    if not overflow and current_line and len(lines) < max_lines:
        lines.append(current_line)

    return "\\N".join(lines)


def generate_srt(segments: List[SubtitleSegment],
                 max_line_length: int = 40,
                 max_line_count: int = 2) -> str:
    """Generate SRT format subtitle.

    Splits long lines for readability when a segment exceeds max_line_length.

    Args:
        segments: List of subtitle segments
        max_line_length: Max chars per line (40 default)
        max_line_count: Max lines per subtitle event (2 default)

    Returns:
        SRT formatted string
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        text = _strip_leading_punct(seg.text)
        if len(text) > max_line_length:
            text = _split_long_lines(text, max_line_length, max_line_count)
        lines.append(str(i))
        lines.append(f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}")
        lines.append(text)
        lines.append("")
    return "\ufeff" + "\n".join(lines)


def generate_lrc(segments: List[SubtitleSegment]) -> str:
    """Generate LRC format subtitle (lyrics style).
    
    LRC uses start time only, no end time.
    """
    lines = []
    for seg in segments:
        time_tag = _format_lrc_time(seg.start)
        lines.append(f"{time_tag}{_strip_leading_punct(seg.text)}")
    return "\n".join(lines)

def generate_ass(segments: List[SubtitleSegment],
                 title: str = "Subtitle",
                 resolution: tuple = (1920, 1080),
                 style_name: str = "Default") -> str:
    """Generate ASS (Advanced SubStation Alpha) format subtitle.
    
    Includes basic styling header.
    """
    width, height = resolution
    
    header = f"""[Script Info]
Title: {title}
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    event_lines = []
    for seg in segments:
        start = _format_ass_time(seg.start)
        end = _format_ass_time(seg.end)
        # Escape ASS special chars (行首句读标点先清理, 与 SRT/LRC 一致)
        text = _strip_leading_punct(seg.text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        event_lines.append(
            f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{text}"
        )
    
    return header + "\n".join(event_lines)


def _join_text(prev_text: str, curr_text: str) -> str:
    """合并相邻段文本：prev 以中文为主时直接拼接，英文/混合场景保留空格分隔。"""
    sep = "" if is_mostly_cjk(prev_text) else " "
    return prev_text + sep + curr_text


def segments_from_transcription(segments_raw: List[Dict]) -> List[SubtitleSegment]:
    """把流水线输出的原始段转成 SubtitleSegment。

    这里**不再合并相邻段**。字幕行已由引擎在已测量的词边界上断好 (见
    ``caption_builder``), 格式化层再按「间隔小就拼起来」合并会把断句结果
    重新打回一整行 —— 实测 3 行 (0.01-1.11 / 1.45-2.87 / 2.89-4.65) 被拼成
    一个 4.65 秒的单条字幕。

    Args:
        segments_raw: ``[{"start": float, "end": float, "text": str}, ...]``

    Returns:
        ``SubtitleSegment`` 列表, 顺序与时间均与输入一致 (仅剔除空文本)。
    """
    result = []
    for seg in segments_raw:
        text = (seg.get("text") or "").strip()
        if text:
            result.append(
                SubtitleSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=text,
                )
            )
    return result