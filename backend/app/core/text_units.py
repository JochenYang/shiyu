"""
文本单元 — 必须在显示清理与断句中被保护的功能性写法。

移植自 SubPop ``probes/text_units.py``。要点:
  * ``0.1`` 去掉点会变成 ``01``, ``1/30`` 去掉斜杠会变成 ``130``,
    ``GHz`` 被拆开就失去单位含义 —— 这些不是句子标点, 必须原样保留。
  * 显示长度按全角字符计: ASCII 记 0.5, 汉字记 1, 这样 25 个汉字的
    字幕行与 ``50`` 个 ASCII 字符占用相同的视觉宽度。
  * ``clean_generated_text`` 只删句子标点, 不碰保护区间; 删标点导致
    英文单词粘连时补一个空格。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Tuple

# 数值与单位: 0.1 / 1/30 / 3:2 / 12% / 30fps / 5GHz / 100元 / 2.5倍
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])[+\-−]?\d+(?:[.,]\d+)*"
    r"(?:[:/\-–]\d+(?:[.,]\d+)*)*"
    r"(?:\s*(?:[%‰]|(?:kWh|kHz|MHz|GHz|Hz|fps|Mbps|Gbps|GB|MB|TB|KB|mm|cm|km|kg|ms|s|m|g|K|V|W|h)"
    r"(?![A-Za-z])|毫米|厘米|公里|毫秒|分钟|英寸|千克|公斤|像素|赫兹|秒|米|度|档|元|倍))?"
)

# 标识符: C++ / iPhone 18 / model.int8.onnx / Godot#4
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:(?:[-'’][A-Za-z0-9]+)+|(?:\.\d+)+|\+\+|#)"
)


def protected_ranges(text: str, terms: Iterable[str] = ()) -> List[Tuple[int, int]]:
    """返回 ``text`` 中不可被标点清理或断句切开的字符区间。

    Args:
        text: 待检查文本。
        terms: 额外词表 (用户词库), 其完整写法同样受保护。

    Returns:
        ``[(start, end), ...]`` 半开区间列表, 可能重叠, 调用方自行合并。
    """
    spans: List[Tuple[int, int]] = [
        (m.start(), m.end())
        for pattern in (_NUMBER, _IDENTIFIER)
        for m in pattern.finditer(text)
    ]
    for term in terms:
        term = term.strip()
        if not term:
            continue
        # 词条内部空格允许匹配任意空白, 便于命中识别结果里的等价写法
        pattern = r"\s+".join(re.escape(part) for part in term.split())
        spans.extend(
            (m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)
        )
    return spans


def display_length(text: str) -> float:
    """显示宽度: ASCII 记 0.5 个汉字宽, 组合字符不计。"""
    return sum(0.5 if c.isascii() else 1 for c in text if not unicodedata.combining(c))


def clean_generated_text(text: str) -> str:
    """删除句中标点, 保留保护区间与功能性写法。

    识别模型常输出整段带标点的文本; 字幕按语义重排后这些标点多数是多余的
    (逗号尤其如此), 但数值、标识符和词库词条里的点、斜杠、加号必须保留。
    """
    protected = {i for start, end in protected_ranges(text) for i in range(start, end)}
    result: List[str] = []
    for i, char in enumerate(text):
        if not unicodedata.category(char).startswith("P") or i in protected:
            result.append(char)
        elif (
            i
            and i + 1 < len(text)
            and text[i - 1].isascii()
            and (text[i - 1].isalnum() or text[i - 1] in "+#")
            and text[i + 1].isascii()
            and text[i + 1].isalnum()
        ):
            # 删掉英文单词之间的句点不能把两个词粘成一个
            result.append(" ")
    return "".join(result).strip()
