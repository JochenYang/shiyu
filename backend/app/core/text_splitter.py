"""
文本切分工具 — 供云端引擎把整段识别文本切成字幕行。

提供：
  * ``split_text_for_subtitles()`` — 三级分段（句子→逗号→硬切），云端引擎用它
    切分整段识别文本。

本地引擎不再走比例插值切分: 字幕行由 ``caption_builder`` 在已测量的词边界上做
最小代价 DP 切分。这里保留的常量 (如 ``PAUSE_THRESHOLD``) 供两边共用。

同时提供 ``is_mostly_cjk()`` 公共工具函数。
"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────

# 句子终止标点（遇到这些标点即可认为一句话结束）

# 默认阈值
DEFAULT_MAX_CHARS = 25         # 中文字幕推荐上限（云端引擎切分用）
DEFAULT_MAX_CHARS_EN = 42      # 英文字幕推荐上限（云端引擎切分用）


# 停顿切分阈值（秒）: 逐字符时间戳中相邻字符间隔超过该值视为句间停顿,
# 作为优先切分点 (zipformer 无标点输出场景, 比 25 字符硬切更贴近真实句边界)
PAUSE_THRESHOLD = 0.4

# 停顿切分的独立片段下限: 低于该字数的停顿段并入相邻段


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


# ── 三级文本切分（云端引擎用）─────────────────────────────────────


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
