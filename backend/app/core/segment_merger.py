"""
Segment Merger — Heuristic merging of adjacent broken segments.

ASR fixed-window chunking often cuts sentences into fragments like:
  Seg 1: "今天天气真"
  Seg 2: "好我们去公"
  Seg 3: "园散步。"

This module detects and merges such fragments BEFORE LLM correction,
so the LLM receives complete sentences instead of broken pieces.

Three-red-lines principle: merged segments preserve timestamps of their
constituent parts (start=first.start, end=last.end).
"""

# Terminal punctuation marks that indicate a sentence is likely complete
_TERMINAL_PUNCT = frozenset("。？！.!?）\n")

# Chinese sentence-initial particles/pronouns that indicate continuation
_CONTINUATION_STARTS = frozenset(
    "我你他她它我们你们他们她们这那哪谁什么怎"
    "但而并且然虽因所为如若于与及以把被将"
    "还又再才就便已也都在不没正"
)


def _ends_with_terminal(text: str) -> bool:
    """Check if text ends with a sentence-ending punctuation."""
    text = text.rstrip()
    return bool(text) and text[-1] in _TERMINAL_PUNCT


def _starts_with_continuation(text: str) -> bool:
    """Check if text starts with a continuation pattern (lowercase or Chinese particle)."""
    text = text.lstrip()
    if not text:
        return True
    first = text[0]
    # English lowercase letter → likely continuation
    if first.isascii() and first.islower():
        return True
    # Chinese continuation starter
    if first in _CONTINUATION_STARTS:
        return True
    return False


def _is_short_fragment(text: str, max_len: int = 5) -> bool:
    """Check if text is a very short fragment (likely broken)."""
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= max_len:
        # Short AND doesn't end with terminal punctuation → fragment
        return not _ends_with_terminal(stripped)
    return False


def merge_segments(segments: list, max_short_len: int = 5) -> list:
    """Merge adjacent segments that are clearly broken fragments.

    Args:
        segments: List of dicts with keys [start, end, text]
        max_short_len: Max character count for a "short fragment"

    Returns:
        New list of segments with merged fragments. Returns a new list;
        does not mutate the input.

    Merge rules (applied in order):
    1. If a segment ends WITHOUT terminal punctuation AND next starts
       with continuation → merge them.
    2. If a segment is very short AND ends without terminal punctuation
       → merge with next segment.
    """
    if not segments:
        return []

    merged = []
    buf = dict(segments[0])  # start merging buffer

    for i in range(1, len(segments)):
        cur = segments[i]

        # Decide whether to merge cur into buf
        should_merge = False

        # Rule 1: No terminal + continuation start
        if not _ends_with_terminal(buf["text"]) and _starts_with_continuation(cur["text"]):
            should_merge = True

        # Rule 2: buf is a short fragment (broken)
        if not should_merge and _is_short_fragment(buf["text"], max_short_len):
            should_merge = True

        if should_merge:
            # Merge: extend text, extend end time
            sep = ""  # no separator; fragments may overlap at boundary
            buf["text"] = (buf["text"].rstrip() + " " + cur["text"].lstrip()).strip()
            buf["end"] = cur["end"]
        else:
            # Flush buffer
            merged.append(dict(buf))
            buf = dict(cur)

    # Flush last buffer
    merged.append(dict(buf))

    return merged


# ---------------------------------------------------------------------------
# Long-segment splitter (Phase 3: split corrected segments that are too long)
# ---------------------------------------------------------------------------


def split_long_segments(
    segments: list,
    max_chars: int = 50,
    min_chars: int = 15,
    max_duration: float = 7.0,
) -> list:
    """Split segments that are too long at sentence boundaries.

    After LLM correction (**Phase 2**), some merged segments may still be
    too long for subtitle display.  This function splits them at
    sentence-ending punctuation and assigns proportional timestamps.

    Args:
        segments: list of dicts with keys ``start``, ``end``, ``text``.
        max_chars: maximum chars per segment (triggers split).
        min_chars: minimum chars for a split fragment; shorter ones are
            merged into the next fragment.
        max_duration: maximum seconds per segment.

    Returns:
        New list with long segments split; original segments are NOT
        mutated.
    """
    if not segments:
        return []

    result: list[dict] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        duration = seg["end"] - seg["start"]

        # ----- keep as-is if short enough -----
        if len(text) <= max_chars and duration <= max_duration:
            result.append(dict(seg))
            continue

        # ----- find valid sentence-end split points -----
        # Manual iteration to handle decimal points correctly
        split_after: list[int] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in "。！？!?\n":
                split_after.append(i + 1)  # split after this char
            elif ch == ".":
                # decimal point (e.g. "0.5") — NOT a split
                if i > 0 and i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
                    i += 1
                    continue
                # period followed by space or end → sentence end
                if i + 1 >= len(text) or text[i + 1] == " ":
                    split_after.append(i + 1)
                # else: part of URL, abbreviation, etc. — keep
            i += 1

        if not split_after:
            result.append(dict(seg))
            continue

        # ----- build raw parts from split points -----
        parts: list[str] = []
        prev = 0
        for sp in split_after:
            part = text[prev:sp].strip()
            if part:
                parts.append(part)
            prev = sp
        remaining = text[prev:].strip()
        if remaining:
            if parts:
                # If remaining is very short, append to last part
                if len(remaining) < min_chars:
                    parts[-1] += remaining
                else:
                    parts.append(remaining)
            else:
                parts.append(remaining)

        # ----- merge tiny parts with neighbours -----
        merged_parts: list[str] = []
        buf = ""
        for part in parts:
            if buf and len(part) < min_chars:
                buf += part  # attach short fragment to previous
            else:
                if buf:
                    merged_parts.append(buf)
                buf = part
        if buf:
            merged_parts.append(buf)

        if len(merged_parts) <= 1:
            result.append(dict(seg))
            continue

        # ----- assign proportional timestamps -----
        total_chars = sum(len(p) for p in merged_parts)
        if total_chars == 0:
            result.append(dict(seg))
            continue

        cursor = seg["start"]
        for idx, part_text in enumerate(merged_parts):
            is_last = idx == len(merged_parts) - 1
            part_ratio = len(part_text) / total_chars
            part_end = seg["end"] if is_last else cursor + duration * part_ratio
            result.append({
                "start": round(cursor, 3),
                "end": round(part_end, 3),
                "text": part_text,
            })
            cursor = part_end

    return result
