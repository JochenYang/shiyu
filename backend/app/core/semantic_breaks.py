"""
语义断句 — 在已测量的词边界上做最小代价切分。

移植自 SubPop ``probes/semantic_breaks.py``。核心思想: 字幕断句的落点必须是
**真实测量到的**词边界, 而不是按字符数比例推算的位置。比例插值会让时间戳
漂移 (本项目的 tasks/lessons.md 记录过一次同类事故), 因此这里只在给定的词
序列上选择切点, 从不发明时间戳。

代价项 (越小越适合作为切点):
  * 基础代价随段长上升, 过短也加罚 (避免切出「的」这种碎片)
  * 段尾落在「的/把/被/和/与/及/在/将/这/那」等虚词 → 加罚 (读不通)
  * 段首落在「和/与/及/的/了/着/过」 → 加罚
  * 段间有真实停顿 → 减罚 (最可靠的边界信号)
  * 前后是句读标点 → 减罚
  * 后接「其实/但是/所以」等引导词 → 减罚
"""
from __future__ import annotations

import re
from typing import Callable, List, Sequence

from app.core.text_units import clean_generated_text, display_length

# 句首引导词: 出现在下一段开头时, 说明这里是一个自然的语义起头
LEAD_INS = (
    "其实", "但是", "不过", "所以", "因此", "而且", "并且", "另外", "然后", "比如", "例如",
)

# 段尾/段首虚词: 落点在这些字后面或前面都会把短语劈开
_TRAILING_PARTICLES = ("的", "把", "被", "和", "与", "及", "在", "将", "这", "那", "对", "给", "从")
_LEADING_PARTICLES = ("和", "与", "及", "的", "了", "着", "过")


def coordinated_ranges(text: str) -> List[tuple]:
    """返回应整体保留的并列结构区间 (如「使用体验和研究成果」)。

    用 jieba 的词性标注找 ``A 和 B`` 结构; 短小的名词/动词并列不应被切开。
    纯 ASCII 文本直接返回空 —— 中英混排的并列结构判定依赖中文词性, 对
    英文误判的代价大于收益。

    Args:
        text: 待分析文本。

    Returns:
        ``[(start, end), ...]`` 字符区间, 与 ``protected_ranges`` 同格式。
    """
    if not re.search(r"[\u3400-\u9fff]", text):
        return []

    import jieba.posseg

    tokens = []
    cursor = 0
    for token in jieba.posseg.cut(text, HMM=False):
        tokens.append((token.word, token.flag, cursor, cursor + len(token.word)))
        cursor += len(token.word)

    spans: List[tuple] = []

    # 「这/那/一/几 + 量词 + 名词」构成的时间短语整体保留
    for i in range(len(tokens) - 2):
        if tokens[i][0] in ("这", "那", "一", "几") and tokens[i + 1][1].startswith("q") and tokens[i + 2][1].startswith("n"):
            left = i - 1 if i and tokens[i - 1][1].startswith("t") else i
            if tokens[i + 2][3] - tokens[left][2] <= 10:
                spans.append((tokens[left][2], tokens[i + 2][3]))

    # 「名词/动词 + 和/与/及 + 名词/动词」并列结构
    for i, (word, _, _, _) in enumerate(tokens):
        if word not in ("和", "与", "及") or i == 0 or i + 1 == len(tokens):
            continue
        left, right = i - 1, i + 1
        if not all(tokens[n][1].startswith(("n", "v")) for n in (left, right)):
            continue
        # 左侧再扩一个动宾单元 (如「使用 体验 和 研究 成果」)
        if (
            left > 0
            and tokens[left][1].startswith("n")
            and tokens[left - 1][1].startswith("v")
            and len(tokens[left - 1][0] + tokens[left][0]) <= 6
        ):
            left -= 1
        if (
            right + 1 < len(tokens)
            and tokens[right][1].startswith("v")
            and tokens[right + 1][1].startswith("n")
            and len(tokens[right][0] + tokens[right + 1][0]) <= 6
        ):
            right += 1
        start, end = tokens[left][2], tokens[right][3]
        if end - start <= 12:
            spans.append((start, end))

    return spans


def phrase_partition(
    words: Sequence,
    max_chars: float,
    max_duration: float,
    punctuation_boundary: Callable[[str, str], str | None],
) -> List[List]:
    """把一段连续语音按最小代价切成字幕行, 返回词的分组。

    只选择切点, 不产生新词、不修改时间戳。整段无法在不超限的情况下切开时,
    仍保证返回至少一个分组 (由调用方决定是否接受超限)。

    Args:
        words: 已测量的词序列, 每项需有 ``text`` / ``start`` / ``end``。
        max_chars: 单行显示宽度上限 (按 display_length 计)。
        max_duration: 单行时长上限 (秒)。
        punctuation_boundary: ``(本词文本, 下一词文本) -> 'hard'|'soft'|None``。

    Returns:
        分组后的词列表 ``[[word, ...], ...]``, 顺序与原序列一致。
    """
    count = len(words)
    if not count:
        return []

    clean = [clean_generated_text(w.text) for w in words]

    costs = [float("inf")] * (count + 1)
    paths: List[int | None] = [None] * (count + 1)
    costs[0] = 0.0

    for end in range(1, count + 1):
        # 从 end 往前扩展起点, 一旦超限即可停止 (再往前只会更长)
        for start in range(end - 1, -1, -1):
            text = "".join(clean[start:end])
            length = display_length(text)
            duration = words[end - 1].end - words[start].start
            if end - start > 1 and (length > max_chars or duration > max_duration):
                break

            cost = 3.5 + max(0.0, length - 14) ** 2 * 0.22 + max(0.0, 5 - length) ** 2 * 0.65
            if text.endswith(LEAD_INS):
                cost += 15
            if text.endswith(_TRAILING_PARTICLES):
                cost += 7
            if text.startswith(_LEADING_PARTICLES):
                cost += 8

            if end < count:
                left = text
                right = "".join(clean[end:])
                pause = words[end].start - words[end - 1].end
                cost -= min(1.5, max(0.0, pause) * 5)
                if punctuation_boundary(words[end - 1].text, words[end].text):
                    cost -= 2
                if right.startswith(LEAD_INS) and length >= 5:
                    cost -= 3
                # 「……你知道吗 / 告诉你为什么」这类引导性设问适合在疑问处切开
                if re.search(r"(?:不知道|知道|想知道|解释|告诉你)为什么$", left) and display_length(right) >= 5:
                    cost -= 3

            candidate = costs[start] + cost
            if candidate < costs[end]:
                costs[end] = candidate
                paths[end] = start

    chunks: List[List] = []
    end = count
    while end:
        start = paths[end]
        if start is None:
            raise ValueError("无法在已测量的词边界断句")
        chunks.append(list(words[start:end]))
        end = start
    return list(reversed(chunks))
