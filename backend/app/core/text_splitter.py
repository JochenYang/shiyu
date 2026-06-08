"""
通用文本切分工具 — 合并 ASR 碎片 segment 为完整的字幕句子。

提供两套策略：
  1. ``split_text_for_subtitles()`` — 三级分段（句子→逗号→硬切），用于 ASR
     引擎的原始文本切分（替换 sensevoice_engine 和 cloud_engine 的重复实现）。
  2. ``split_text_at_boundaries()`` — 按标点在长句处切分 + 比例分配时间戳，
     用于修正后超长 segment 的二次拆解（替换 segment_merger.split_long_segments
     和 sentence_splitter._split_long_text）。

同时提供 ``is_mostly_cjk()`` 公共工具函数。
"""
import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────

# 句子终止标点（遇到这些标点即可认为一句话结束）
TERMINAL_PUNCT = frozenset("。？！.!?）\n")

# 默认阈值
DEFAULT_MAX_CHARS = 25         # 中文字幕推荐上限（Phase 1: 35→25, 与 sensevoice/cloud 引擎 max_len 对齐）
DEFAULT_MAX_CHARS_EN = 42      # 英文字幕推荐上限（Phase 1: 50→42, 英文单词级切分更紧凑）
DEFAULT_MIN_CHARS = 15         # 切分后最小片段字符数
DEFAULT_MAX_DURATION = 7.0     # 切分后最大时长（秒）

# Phase 1 新增:相邻段间微重叠（秒），用于掩盖时间戳抖动/比例分配误差
# 行业惯例（YouTube auto-caption 近似值）；overlap_sec=0 可禁用
OVERLAP_SECONDS = 0.05


# ── CJK 检测 ──────────────────────────────────────────────────────────────


def is_mostly_cjk(text: str) -> bool:
    """Check if text is primarily CJK (Chinese/Japanese/Korean)."""
    if not text:
        return True
    cjk = sum(1 for c in text if (
        '\u4e00' <= c <= '\u9fff' or   # CJK Unified
        '\u3040' <= c <= '\u30ff' or   # Hiragana + Katakana
        '\uac00' <= c <= '\ud7af' or   # Hangul
        '\u3000' <= c <= '\u303f'      # CJK Symbols
    ))
    return cjk / len(text) > 0.3


# ── 三级文本切分（ASR 引擎原生字幕用）───────────────────────────────


def split_text_for_subtitles(
    text: str,
    max_len: int = 25,
    is_cjk_flag: bool = True,
) -> List[str]:
    """Split text into subtitle-friendly segments using 3-level strategy.

    Level 1: Split by sentence boundaries (。！？；!?;  .!? for English)
    Level 2: Split long sentences at commas/colons
    Level 3: Hard-cut fallback (word-aware for English, char-level for CJK)

    Args:
        text: Raw text to split.
        max_len: Maximum characters per segment (25 recommended for CJK,
            42 for English).
        is_cjk_flag: True for CJK text (character-level), False for
            English (word-level).

    Returns:
        List of text fragments, each <= max_len if possible.
    """
    if not text:
        return [""] if text == "" else []

    # Level 1: Sentence boundary split
    if is_cjk_flag:
        parts = re.split(r"([。！？；!?;.])", text)
    else:
        parts = re.split(r"([.!?;](?:\s|$))", text)

    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        s = s.strip()
        if s:
            sentences.append(s)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    # Level 2: Comma/colon split for long segments
    refined = []
    for sent in sentences:
        if len(sent) <= max_len:
            refined.append(sent)
            continue
        if is_cjk_flag:
            sub_parts = re.split(r"([，、,：:、])", sent)
        else:
            sub_parts = re.split(r"(,\s*)", sent)
        current = ""
        for j in range(0, len(sub_parts) - 1, 2):
            clause = sub_parts[j] + (sub_parts[j + 1] if j + 1 < len(sub_parts) else "")
            if not current:
                current = clause
            elif len(current) + len(clause) <= max_len:
                current += clause
            else:
                if current.strip():
                    refined.append(current.strip())
                current = clause
        if len(sub_parts) % 2 == 1 and sub_parts[-1]:
            if current and len(current) + len(sub_parts[-1]) <= max_len:
                current += sub_parts[-1]
            else:
                if current.strip():
                    refined.append(current.strip())
                current = sub_parts[-1]
        if current.strip():
            refined.append(current.strip())

    # Level 3: Hard-cut fallback
    final = []
    for seg in refined:
        if len(seg) <= max_len:
            final.append(seg)
            continue
        if is_cjk_flag:
            i = 0
            while i < len(seg):
                end = min(i + max_len, len(seg))
                chunk = seg[i:end]
                if end < len(seg) and seg[end] in "，、,：:。！？；!?;":
                    chunk += seg[end]
                    end += 1
                final.append(chunk.strip())
                i = end
        else:
            words = seg.split(" ")
            current = ""
            for word in words:
                if not word:
                    continue
                if not current:
                    current = word
                elif len(current) + 1 + len(word) <= max_len:
                    current += " " + word
                else:
                    if current.strip():
                        final.append(current.strip())
                    current = word
            if current.strip():
                final.append(current.strip())

    if not final:
        final = [text.strip()] if text.strip() else []

    return final


# ── 长句切分（修正后超长 segment 拆解用）─────────────────────────


def split_text_at_boundaries(
    segments: List[Dict],
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> List[Dict]:
    """Split segments that exceed length/duration at sentence boundaries.

    Works on each segment individually: if text exceeds ``max_chars`` or
    duration exceeds ``max_duration``, it is split at terminal punctuation.
    Short fragments (< ``min_chars``) are merged into adjacent segments.
    Timestamps are assigned proportionally by character count.

    Args:
        segments: List of dicts with keys ``start``, ``end``, ``text``.
        max_chars: Maximum chars per segment (triggers split).
        min_chars: Minimum chars for a split fragment; shorter ones
            are merged into neighbours.
        max_duration: Maximum seconds per segment.

    Returns:
        New list with long segments split; original segments unchanged.
    """
    if not segments:
        return []

    result: List[Dict] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        duration = seg["end"] - seg["start"]

        # Keep as-is if short enough
        if len(text) <= max_chars and duration <= max_duration:
            result.append(dict(seg))
            continue

        # Find valid sentence-end split points
        split_after = _find_sentence_splits(text)

        if not split_after:
            # No punctuation found — hard-cut at max_chars if text is too long
            if len(text) > max_chars:
                parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
                _assign_timestamps(seg, parts, duration, result)
                continue
            result.append(dict(seg))
            continue

        # Build raw parts from split points
        parts = _build_parts(text, split_after, min_chars)

        # Merge tiny parts with neighbours
        merged_parts = _merge_tiny_parts(parts, min_chars)

        if len(merged_parts) <= 1:
            # All parts merged back (e.g. all very short) — hard split without merge
            parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            if len(parts) > 1:
                _assign_timestamps(seg, parts, duration, result)
                continue
            result.append(dict(seg))
            continue

        _assign_timestamps(seg, merged_parts, duration, result)

    return result


def _assign_timestamps_with_overlap(
    seg: Dict,
    parts: List[str],
    duration: float,
    result: List[Dict],
    overlap_sec: float = OVERLAP_SECONDS,
) -> None:
    """按字符数比例分配时间戳，相邻段间保留 ``overlap_sec`` 微重叠。

    Phase 1 新增：替代纯比例分配，用微小重叠掩盖时间戳抖动。
    默认 ``overlap_sec = OVERLAP_SECONDS (0.05s)``，传 0 即可禁用回到旧行为。

    不变式:
      - parts[0].start == seg.start
      - parts[-1].end  == seg.end
      - parts[i].end + overlap_sec >= parts[i+1].start  (连续无回退，允许微小重叠)
    """
    total_chars = sum(len(p) for p in parts)
    if total_chars == 0:
        result.append(dict(seg))
        return

    cursor = seg["start"]
    n = len(parts)
    for idx, part_text in enumerate(parts):
        is_last = idx == n - 1
        part_ratio = len(part_text) / total_chars
        if is_last:
            # 最后一段固定为 seg.end，避免累加误差
            part_end = seg["end"]
        else:
            part_end = cursor + duration * part_ratio
        result.append({
            "start": round(cursor, 3),
            "end": round(part_end, 3),
            "text": part_text,
        })
        # 非最后段：下一段 start = 当前 end - overlap_sec（用 max 防止倒退到 seg.start 之前）
        if not is_last:
            cursor = max(seg["start"], part_end - overlap_sec)


def _assign_timestamps(
    seg: Dict,
    merged_parts: List[str],
    duration: float,
    result: List[Dict],
) -> None:
    """Assign proportional timestamps to split parts and append to result.

    Phase 1 改动:委托给 ``_assign_timestamps_with_overlap``，默认 0.05s 重叠。
    签名保持不变，向后兼容。
    """
    _assign_timestamps_with_overlap(seg, merged_parts, duration, result)


# ── 内部辅助 ──────────────────────────────────────────────────────────


def _find_sentence_splits(text: str) -> List[int]:
    """Find positions after sentence-ending punctuation.

    Returns indices *after* the punctuation char where a split
    is valid (e.g. ``[3, 7, 12]``).
    """
    split_after: List[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "。！？!?\n":
            split_after.append(i + 1)
        elif ch == ".":
            if i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1  # decimal point, skip
                continue
            if i + 1 >= len(text) or text[i + 1] == " ":
                split_after.append(i + 1)
        i += 1
    return split_after


def _build_parts(text: str, split_after: List[int], min_chars: int) -> List[str]:
    """Build text parts from split point indices."""
    parts: List[str] = []
    prev = 0
    for sp in split_after:
        part = text[prev:sp].strip()
        if part:
            parts.append(part)
        prev = sp
    remaining = text[prev:].strip()
    if remaining:
        if parts and len(remaining) < min_chars:
            parts[-1] += remaining
        else:
            parts.append(remaining)
    return parts


def _merge_tiny_parts(parts: List[str], min_chars: int) -> List[str]:
    """Merge very short parts into neighbouring segments."""
    merged: List[str] = []
    buf = ""
    for part in parts:
        if buf and len(part) < min_chars:
            buf += part
        else:
            if buf:
                merged.append(buf)
            buf = part
    if buf:
        merged.append(buf)
    return merged
