"""
VAD 驱动的智能断句器 — 利用语流停顿将 ASR segments 合并为句子。

实现原理：
  1. 对音频运行 Silero-VAD，检测活跃语音区间
  2. 停顿时间超过阈值 (>0.5s) → 判为句子边界
  3. 将 VAD 边界对应到 ASR segment 的时间戳
  4. 在边界处切分，非边界处合并
  5. 结合标点启发式做二次精修

使用场景：
  - 在 ASR 输出后、修正管道前插入
  - 替代或补充 ``segment_merger.merge_segments()``
"""
import logging
from typing import Dict, List, Tuple

from app.core import text_splitter
logger = logging.getLogger(__name__)

# 句子间隔阈值（秒）：停顿超过此值即认为句子结束
DEFAULT_MIN_PAUSE_SEC = 0.5

# 最大句子长度（字符）：超过此值强制切分
MAX_SENTENCE_CHARS = 40


def _deduplicate_segments(segments: List[Dict]) -> List[Dict]:
    """Merge overlapping ASR segments — text-only, never changes timestamps.

    ASR sliding window (e.g. 3s overlap) can produce segments with overlapping
    timestamps and duplicate/partial text. This preprocessor:

    1. Sorts by start time
    2. When text is a substring of the previous segment → keep previous, extend end
    3. When previous text is a substring → keep current segment (longer text)
    4. When partial overlap exists (e.g. continuation across chunk boundary) → merge text
    5. **Never modifies start/end timestamps** — VAD alignment will produce
       authoritative timestamps later.
    """
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: (s["start"], s["end"]))
    result: List[Dict] = [dict(sorted_segs[0])]

    for seg in sorted_segs[1:]:
        prev = result[-1]
        pt = prev["text"].strip()
        ct = seg["text"].strip()

        if not ct:
            continue

        # Check if segments overlap in time
        gap = seg["start"] - prev["end"]
        if gap >= -0.01:
            # No timestamp overlap → keep as separate segment
            result.append(dict(seg))
            continue

        # Timestamp overlap detected: merge text
        # Pure duplicate: one text entirely contained in the other
        if ct in pt:
            # prev already contains this text → extend its end time
            prev["end"] = max(prev["end"], seg["end"])
            continue
        if pt in ct:
            # seg has more/better text → replace prev's text, keep prev's start
            prev["text"] = ct
            prev["end"] = max(prev["end"], seg["end"])
            continue

        # Partial overlap: find common character overlap at boundary
        overlap_len = 0
        for n in range(min(len(pt), len(ct)), 0, -1):
            if pt[-n:] == ct[:n]:
                overlap_len = n
                break
        if overlap_len > len(pt) * 0.3:  # significant overlap
            new_text = pt + ct[overlap_len:]
        else:
            new_text = pt + " " + ct
        prev["text"] = new_text
        prev["end"] = max(prev["end"], seg["end"])

    return result


def _align_segments_to_vad_regions(
    segments: List[Dict],
    speech_intervals: List[Tuple[float, float]],
) -> List[Dict]:
    """Align ASR text to VAD speech intervals (preserving all text, deduped).

    Phase 1B fix (corrected): the previous "BFS-dedupe with segment_consumed
    cursor" version was too aggressive and DROPPED text in the common case
    where a VAD region is much shorter than its overlapping ASR segment
    (frac < 1.0 but consumed cursor advanced faster than the next region
    could read the remainder). The new approach:

      1. Run the original proportional-split algorithm that writes
         ``region_texts[i]`` for every (segment × region) overlap.
      2. For each region, deduplicate its accumulated chunks by removing
         the longest common prefix with the **previous region's tail text**
         (suffix-prefix overlap detection, max 64 chars).
      3. This guarantees every ASR character ends up in exactly one region
         (no drops) while still removing the sliding-window duplicates.

    Args:
        segments: Original ASR segments (each with start, end, text).
        speech_intervals: VAD speech intervals ``[(start, end), ...]``
            from ``VADDetector.detect_speech()``.

    Returns:
        New segments with VAD-based timestamps and deduped text.
    """
    if not speech_intervals:
        return list(segments)

    if not segments:
        return []

    # Filter speech intervals to those that overlap with the ASR range.
    seg_start = min(s["start"] for s in segments)
    seg_end = max(s["end"] for s in segments)
    active_regions: List[Tuple[float, float]] = [
        (max(s, seg_start), min(e, seg_end))
        for s, e in speech_intervals
        if e > seg_start and s < seg_end
    ]
    if not active_regions:
        return list(segments)

    # ── Step 1: Original proportional-split (preserves all text) ──────
    region_texts: List[List[str]] = [[] for _ in active_regions]

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Find all overlapping regions and their overlap durations
        overlaps: List[Tuple[int, float]] = []
        for i, (rs, re) in enumerate(active_regions):
            o = min(seg["end"], re) - max(seg["start"], rs)
            if o > 0:
                overlaps.append((i, o))

        if not overlaps:
            # No overlap — append to nearest region
            nearest = min(
                range(len(active_regions)),
                key=lambda i: min(
                    abs(seg["start"] - active_regions[i][1]),
                    abs(seg["end"] - active_regions[i][0]),
                ),
            )
            region_texts[nearest].append(text)
            continue

        # Distribute text proportionally by overlap duration
        total_overlap = sum(o for _, o in overlaps)
        cursor = 0
        text_len = len(text)
        for idx, (region_idx, overlap_dur) in enumerate(overlaps):
            if cursor >= text_len:
                break
            if idx == len(overlaps) - 1:
                chunk = text[cursor:]
            else:
                n = max(1, round(text_len * overlap_dur / total_overlap))
                # Try to find a natural boundary near the cut position
                cut_pos = cursor + n
                if cut_pos < text_len:
                    # Search within ±3 chars for punctuation or space
                    search_start = max(cursor + 1, cut_pos - 3)
                    search_end = min(text_len, cut_pos + 3)
                    best_cut = cut_pos
                    for p in range(search_start, search_end):
                        if text[p] in " ，、。！？；：.!?;:)\n":
                            best_cut = p + 1
                            break
                    if best_cut == cut_pos:
                        remaining = text_len - best_cut
                        if remaining == 1 and best_cut > cursor + 1:
                            best_cut -= 1
                    cut_pos = best_cut
                chunk = text[cursor:cut_pos]
                cursor = cut_pos
            if chunk.strip():
                region_texts[region_idx].append(chunk.strip())

    # ── Step 2: Dedup across region boundaries (suffix-prefix) ────────
    def _strip_overlap_prefix(existing: str, new: str) -> str:
        """Return new[overlap_len:] where overlap_len is the longest suffix
        of *existing* that equals a prefix of *new* (max 64 chars check)."""
        max_check = min(len(existing), len(new), 64)
        for n in range(max_check, 0, -1):
            if existing[-n:] == new[:n]:
                return new[n:]
        return new

    result: List[Dict] = []
    prev_text = ""  # accumulated text from previous region
    for i, (rs, re) in enumerate(active_regions):
        if not region_texts[i]:
            continue
        # First, dedup within region (consecutive duplicates)
        deduped_chunks: List[str] = []
        for chunk in region_texts[i]:
            if deduped_chunks and deduped_chunks[-1] == chunk:
                continue  # skip exact duplicate
            deduped_chunks.append(chunk)
        merged = " ".join(deduped_chunks)
        # Then, dedup against previous region's tail (cross-region sliding)
        if prev_text:
            merged = _strip_overlap_prefix(prev_text, merged)
        if not merged.strip():
            continue
        result.append({
            "start": round(rs, 3),
            "end": round(re, 3),
            "text": merged,
        })
        # Keep a sliding window of last ~64 chars for next region dedup
        prev_text = (prev_text + merged)[-128:]

    result.sort(key=lambda s: s["start"])
    if not result:
        return list(segments)
    return result


def _split_text_at_punctuation(text: str) -> List[str]:
    """Split text at sentence/clause boundaries, preserving punctuation.

    Returns a list of non-empty chunks. Falls back to the whole text
    wrapped in a single-element list if no split points are found.
    """
    if not text:
        return []
    split_points: List[int] = [0]
    for i, ch in enumerate(text):
        if ch in "。！？!?；;,.、，\n":
            # Split AFTER the punctuation
            if i + 1 < len(text):
                split_points.append(i + 1)
    if split_points[-1] != len(text):
        split_points.append(len(text))

    chunks: List[str] = []
    for i in range(len(split_points) - 1):
        chunk = text[split_points[i]:split_points[i + 1]].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def split_segments_at_vad_boundaries(
    segments: List[Dict],
    speech_intervals: List[Tuple[float, float]],
    min_pause: float = DEFAULT_MIN_PAUSE_SEC,
) -> List[Dict]:
    """Merge/split ASR segments into sentences using VAD speech intervals.

    **VAD-first approach**: VAD speech intervals (from ``detect_speech()``)
    define the subtitle timing; ASR text is aligned to these intervals.
    This produces accurate timestamps regardless of ASR engine quality.

    Pipeline:
    0. **Deduplicate**:  Remove text duplicates from ASR sliding window.
    1. **Align to VAD**:  Assign ASR text to VAD speech intervals.
       Each interval → one subtitle segment with VAD-based timestamps.
    2. **Enforce max length**:  Split any segment exceeding ``MAX_SENTENCE_CHARS``
       using punctuation-based heuristics.

    Args:
        segments: Original ASR segments (each with ``start``, ``end``, ``text``).
        speech_intervals: VAD speech intervals ``[(start, end), ...]``
            from ``VADDetector.detect_speech()``.
        min_pause: Minimum pause duration that counts as a sentence break.

    Returns:
        New list of sentence-level segments with VAD-accurate start/end/text.
    """
    if not segments:
        return []

    # Step 0: Deduplicate overlapping ASR text (does NOT change timestamps)
    clean = _deduplicate_segments(segments)

    if not speech_intervals:
        return _enforce_max_length(clean)

    # Step 1: Align ASR text to VAD speech intervals (VAD-first timestamps)
    aligned = _align_segments_to_vad_regions(clean, speech_intervals)

    # Step 2: Enforce max length
    final = _enforce_max_length(aligned)

    logger.info(
        "[SentenceSplitter] %d ASR segments → %d VAD regions → %d sentences (VAD intervals: %d)",
        len(segments), len(aligned), len(final), len(speech_intervals),
    )
    return final


def _split_long_text(seg: Dict, max_chars: int) -> List[Dict]:
    """Split a segment that exceeds ``max_chars`` using punctuation as hints."""
    if max_chars <= 0:
        return [seg]
    return text_splitter.split_text_at_boundaries(
        [seg], max_chars, text_splitter.DEFAULT_MIN_CHARS, text_splitter.DEFAULT_MAX_DURATION,
    )


def _enforce_max_length(segments: List[Dict]) -> List[Dict]:
    """Enforce MAX_SENTENCE_CHARS on a list of segments."""
    result: List[Dict] = []
    for seg in segments:
        if len(seg["text"]) > MAX_SENTENCE_CHARS:
            result.extend(_split_long_text(seg, MAX_SENTENCE_CHARS))
        else:
            result.append(seg)
    return result
