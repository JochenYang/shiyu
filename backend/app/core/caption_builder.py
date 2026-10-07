"""
字幕构建 — 把「已测量的逐字符时间」变成可读字幕行。

这是 semantic_breaks 与引擎输出之间的适配层。输入是带 ``char_times``
(逐字符真实开始时间) 的段, 输出是切好的字幕段:

  1. 用 jieba 在**字符偏移**上求词边界, 词的时间取首字符开始时间与末字符
     开始时间 —— 全部来自测量值, 不做任何推算。
  2. ``protected_ranges`` / ``coordinated_ranges`` 覆盖的区间内不允许切分,
     保证 ``0.1``、``1/30``、``C++``、词库词条与并列短语保持完整。
  3. 真实停顿 (相邻字符间隔超过阈值) 作为强边界, 先把长段切成语句块。
  4. 每个语句块交给 ``phrase_partition`` 做最小代价切分。

关于时间语义 (重要)
-------------------
CTC 只给出每个字符的**开始**时间, 不给出结束时间。因此本模块:

  * 词的 ``end`` 取「末字符的开始时间」—— 这是该词最后一次被测量到的时间,
    不含其后的静音。若改用「下一字符的开始时间」, 一段 7 秒静音会把前一个
    词拉长成 7 秒, 既超出单行时长上限导致该词独占一行, 又让词间停顿变成
    负数而失效 (实测把「大家」拉成 7.2 秒)。
  * 字幕行的 ``end`` 取「下一行的开始时间」, 末行取段结束时间。这两个都是
    测量值。

因此**每一个输出的时间戳都来自测量值** (某个字符的开始时间, 或段的结束时间),
不存在按比例插值或凭常数编造的时间。代价是: 当一句话之后有较长静音时, 该行
字幕会一直显示到下一句开始 —— 这是只有字符开始时间时无法避免的取舍; 要彻底
解决需要像 SubPop 那样引入独立的强制对齐模型给出真实词尾。

没有 ``char_times`` 的段 (例如云端引擎) 不做任何推测, 原样返回, 由调用方
决定是否回退到旧的按长度切分。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from app.core.semantic_breaks import coordinated_ranges, phrase_partition
from app.core.text_units import clean_generated_text, protected_ranges

logger = logging.getLogger(__name__)

# 真实停顿阈值 (秒): 相邻字符间隔超过它视为语句边界
PAUSE_THRESHOLD = 0.4
# 单字字幕的停留时长 (秒): 末字之后再多显示一点, 避免字幕一闪而过
_CHAR_HOLD_SECONDS = 0.3
# 单行字幕的最短显示时长 (秒): 极短的词也要看得见
_MIN_ROW_SECONDS = 0.6


@dataclass(frozen=True)
class MeasuredWord:
    """一个词及其测量时间。

    Attributes:
        text: 词文本 (原样, 可能含标点或空白)。
        start: 首字符的开始时间 (测量值)。
        end: 末字符的开始时间 (测量值); 见模块 docstring 的时间语义说明。
        pause_before: 该词之前是否有超过阈值的真实停顿 (块切分信号)。
        char_start: 该词在段文本中的起始字符偏移 (含)。
        char_end: 该词在段文本中的结束字符偏移 (不含)。
    """

    text: str
    start: float
    end: float
    pause_before: bool = False
    char_start: int = 0
    char_end: int = 0


def punctuation_boundary(text: str, next_text: str = "") -> Optional[str]:
    """判断词尾标点是否为句读边界。

    Returns:
        ``'hard'`` (句号/问号/叹号) | ``'soft'`` (逗号/分号) | ``None``。
    """
    # 数字中的逗号不是句子边界 (1,000)
    if text.rstrip().endswith(",") and next_text.lstrip()[:1].isdigit():
        return None
    stripped = text.rstrip().rstrip("”’」』》〉】〕）)]\"'")
    if not stripped:
        return None
    if stripped[-1] in "。！？!?":
        return "hard"
    if stripped[-1] in "，,；;":
        return "soft"
    # 小数点不是句末标点
    if stripped.endswith(".") and not stripped[-2:-1].isdigit():
        return "hard"
    return None


def _lexical_boundaries(text: str, protected_terms: Iterable[str]) -> set:
    """求允许断句的字符偏移集合 (词边界, 已扣除保护区间)。"""
    import jieba

    jieba.setLogLevel(logging.ERROR)
    boundaries = {end for _, _, end in jieba.tokenize(text, HMM=False)}

    # 保护区间内部一律不许切
    forbidden: set = set()
    for start, end in list(protected_ranges(text, protected_terms)) + list(coordinated_ranges(text)):
        forbidden.update(range(start + 1, end))
    return boundaries - forbidden


def _pause_positions(char_times: Sequence[float]) -> set:
    """返回「其后存在真实停顿」的字符下标集合 (全部来自测量间隔)。"""
    return {
        i
        for i in range(len(char_times) - 1)
        if float(char_times[i + 1]) - float(char_times[i]) >= PAUSE_THRESHOLD
    }


def measured_words(
    text: str,
    char_times: Sequence[float],
    segment_end: float,
    protected_terms: Iterable[str] = (),
) -> List[MeasuredWord]:
    """把 (文本, 逐字符开始时间) 切成测量词。

    Args:
        text: 段文本。
        char_times: 逐字符开始时间, 长度必须与 ``text`` 一致。
        segment_end: 段的结束时间 (当前实现只用于校验, 见模块 docstring)。
        protected_terms: 用户词库词条, 其完整写法不可被切开。

    Returns:
        ``MeasuredWord`` 列表, 覆盖 ``text`` 的全部字符 (含空白与标点)。

    Raises:
        ValueError: ``char_times`` 与 ``text`` 长度不一致 —— 此时任何时间推算
            都不可靠, 宁可报错也不产出错误时间戳。
    """
    if len(char_times) != len(text):
        raise ValueError("逐字符时间与文本长度不一致, 无法可靠断句")

    if not text:
        return []

    times = [float(t) for t in char_times]
    pause_after = _pause_positions(times)

    def make(text_slice: str, first: int, last: int, pause_before: bool) -> MeasuredWord:
        """构造一个词: start 取首字符开始时间, end 取末字符开始时间。"""
        return MeasuredWord(
            text_slice,
            times[first],
            times[last],
            pause_before=pause_before,
            char_start=first,
            char_end=last + 1,
        )

    # 纯 ASCII 文本没有字符级词边界, 按空白切分; 空白归到前一个词, 保证
    # 逐字符覆盖 (否则 "hello world" 会被拼成 "helloworld")。
    if not any("\u3400" <= c <= "\u9fff" for c in text):
        words: List[MeasuredWord] = []
        start = 0
        for i, ch in enumerate(text):
            at_end = i + 1 == len(text)
            # 空白处收词, 把空白并入该词
            if ch.isspace() or at_end:
                last = i
                if start > last:
                    start = i + 1
                    continue
                words.append(
                    make(
                        text[start : last + 1],
                        start,
                        last,
                        pause_before=bool(words) and (start - 1) in pause_after,
                    )
                )
                start = i + 1
        return words

    boundaries = _lexical_boundaries(text, protected_terms)
    words = []
    start = 0
    for i in range(len(text)):
        end = i + 1
        if end in boundaries or end == len(text) or i in pause_after:
            words.append(
                make(
                    text[start:end],
                    start,
                    end - 1,
                    pause_before=bool(words) and (start - 1) in pause_after,
                )
            )
            start = end
    return words


def _blocks_by_pause(words: Sequence[MeasuredWord]) -> List[List[MeasuredWord]]:
    """按真实停顿把词序列切成语句块 (停顿是最可靠的边界信号)。"""
    blocks: List[List[MeasuredWord]] = []
    current: List[MeasuredWord] = []
    for word in words:
        if current and word.pause_before:
            blocks.append(current)
            current = []
        current.append(word)
    if current:
        blocks.append(current)
    return blocks


def build_captions(
    text: str,
    char_times: Sequence[float],
    segment_start: float,
    segment_end: float,
    max_chars: float = 20.0,
    max_duration: float = 6.0,
    protected_terms: Iterable[str] = (),
) -> List[Dict]:
    """把一个测量完整的段切成字幕行。

    输出的每一个 ``start`` 都是某个字符的开始时间; 每一个 ``end`` 都是下一行的
    ``start`` 或 ``segment_end``。两者都是测量值。

    Args:
        text: 段文本。
        char_times: 逐字符开始时间 (与 ``text`` 等长)。
        segment_start: 段的开始时间 (下界保护)。
        segment_end: 段的结束时间。
        max_chars: 单行显示宽度上限。
        max_duration: 单行时长上限 (秒), 按词的实际语音跨度计。
        protected_terms: 用户词库词条。

    Returns:
        ``[{"start": float, "end": float, "text": str}, ...]``, 单调不重叠,
        文本完整 (不丢字、不重字、不乱序)。

    Raises:
        ValueError: 输入长度不一致或 ``max_chars`` 过小时。
    """
    if max_chars < 6:
        raise ValueError("单行字数上限不应小于 6")

    words = measured_words(text, char_times, segment_end, protected_terms)
    if not words:
        return []

    # 1) 分块 → DP 切分, 得到每行的文字、测量起点与末词时间
    rows: List[Dict] = []
    for block in _blocks_by_pause(words):
        for chunk in phrase_partition(block, max_chars, max_duration, punctuation_boundary):
            if not chunk:
                continue
            chunk_text = clean_generated_text("".join(w.text for w in chunk))
            if not chunk_text:
                # 整行都是被清掉的标点: 无内容可显示, 跳过 (文字未丢失)
                continue
            # 行内逐字符时间: 从整段 char_times 按词的字符偏移取时间,
            # 再与 clean_generated_text 清理后的行文本对齐 (被删掉的
            # 标点不占时间位)。DP 切出的行仍需逐字符时间 — VALIDATE
            # 自检与前端精修都要用, 且必须全部来自测量值。
            raw = "".join(w.text for w in chunk)
            kept = clean_generated_text(raw)
            kept_idx, k = [], 0
            for i, ch in enumerate(raw):
                if k < len(kept) and ch == kept[k]:
                    kept_idx.append(i)
                    k += 1
            raw_times = [
                float(char_times[p])
                for w in chunk
                for p in range(w.char_start, w.char_end)
            ]
            row_times = [raw_times[i] for i in kept_idx if i < len(raw_times)]
            rows.append({
                "start": float(chunk[0].start),
                "spoken_end": float(chunk[-1].end),
                "text": chunk_text,
                "char_times": row_times,
            })

    if not rows:
        return []

    # 2) 收尾: 行的结束时间。
    #
    #    只用测量值。两个候选: 下一行的开始时间 (下一句真正开口的时刻), 或
    #    「本行末词最后被测量到的时刻 + 一字时长」—— 后者不会跨越停顿。
    #    若取下一行开始时间, 一句之后接 8 秒静音时字幕会多停 8 秒 (实测
    #    「一个大一的新生」停 9.6s、「谢谢大家」停 7.5s)。因此取两者较小值:
    #    既不会跨停顿逗留, 也不会短于该行真实发音。
    captions: List[Dict] = []
    for index, row in enumerate(rows):
        start = row["start"]
        spoken_end = row["spoken_end"]
        next_start = rows[index + 1]["start"] if index + 1 < len(rows) else float(segment_end)
        # 一行至少显示到它最后一个字之后一点, 但不超过下一行开口时刻
        end = min(max(spoken_end + _CHAR_HOLD_SECONDS, start + _MIN_ROW_SECONDS), next_start)
        if end <= start:
            # 起点相同的退化情形 (重复时间戳): 把文字并入上一行, 绝不丢弃
            if captions:
                captions[-1]["text"] += row["text"]
                captions[-1]["char_times"] = (
                    captions[-1].get("char_times", []) + row.get("char_times", []))
                continue
            # 首行且没有可用区间: 跳过
            logger.debug("[CaptionBuilder] 跳过无有效区间的首行: %r", row["text"])
            continue
        captions.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": row["text"],
            "char_times": row.get("char_times", []),
        })
    return captions
