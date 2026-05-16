"""
Subtitle format generators: SRT, LRC, ASS.
"""
import re
import math
from typing import List, Dict
from dataclasses import dataclass


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


def _split_long_lines(text: str, max_length: int = 40, max_lines: int = 2) -> str:
    """Split long text into multiple lines for subtitle display.
    
    When text exceeds max_lines, remaining content is appended to the last line
    rather than discarded, ensuring no content is lost.
    """
    if len(text) <= max_length:
        return text

    words = text.split(" ") if " " in text else list(text)
    lines = []
    current_line = ""
    overflow = False

    for word in words:
        if overflow:
            # Append remaining words to the last line
            lines[-1] = lines[-1] + " " + word
            continue
        if len(current_line) + len(word) + 1 <= max_length:
            current_line += (" " + word if current_line else word)
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
            if len(lines) >= max_lines:
                # Exceeded max lines: start appending to last line
                lines[-1] = lines[-1] + " " + current_line
                overflow = True

    if not overflow and current_line and len(lines) < max_lines:
        lines.append(current_line)

    return "\\N".join(lines)


def generate_srt(segments: List[SubtitleSegment], 
                 max_line_length: int = 40,
                 max_line_count: int = 2) -> str:
    """Generate SRT format subtitle.
    
    Args:
        segments: List of subtitle segments
        max_line_length: Max chars per line
        max_line_count: Max lines per subtitle event
    
    Returns:
        SRT formatted string
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        text = _split_long_lines(seg.text, max_line_length, max_line_count)
        lines.append(str(i))
        lines.append(f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def generate_lrc(segments: List[SubtitleSegment]) -> str:
    """Generate LRC format subtitle (lyrics style).
    
    LRC uses start time only, no end time.
    """
    lines = []
    for seg in segments:
        time_tag = _format_lrc_time(seg.start)
        lines.append(f"{time_tag}{seg.text}")
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
        # Escape ASS special chars
        text = seg.text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        event_lines.append(
            f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{text}"
        )
    
    return header + "\n".join(event_lines)


def segments_from_transcription(segments_raw: List[Dict], 
                                 merge_gap: float = 0.5) -> List[SubtitleSegment]:
    """Convert raw transcription segments to SubtitleSegment objects.
    
    Optionally merge segments with small gaps.
    """
    result = []
    for seg in segments_raw:
        s = SubtitleSegment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", "").strip()
        )
        if s.text:
            result.append(s)
    
    # Merge segments with small gaps
    if len(result) > 1 and merge_gap > 0:
        merged = [result[0]]
        for curr in result[1:]:
            prev = merged[-1]
            gap = curr.start - prev.end
            if gap <= merge_gap and len(prev.text) + len(curr.text) < 80:
                # Merge
                prev.end = curr.end
                prev.text += " " + curr.text
            else:
                merged.append(curr)
        result = merged
    
    return result