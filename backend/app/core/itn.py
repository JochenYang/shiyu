"""
Inverse Text Normalization (ITN) — 剪映 GrowVoiceITN 简化版。

核心思想:
  - 中文数字 → 阿拉伯数字
  - 但有"年份模式"例外:"二零二六" 是 4 个独立数字 2026,不是数学计算

支持的转换:
  一二三      → 123
  二零二六    → 2026 (年份模式)
  百分之二十  → 20%
  三点一四    → 3.14
  五十六岁    → 56岁
  三百二十    → 320
  一万        → 10000
"""
import logging
import re

logger = logging.getLogger(__name__)

# ── 基础映射 ──────────────────────────────────────────────────────

_CN_DIGITS = "零一二三四五六七八九"
_CN_DIGIT_VAL = {c: i for i, c in enumerate(_CN_DIGITS)}

_CN_UNITS = {
    "十": 10, "百": 100, "千": 1000,
    "万": 10000, "亿": 100000000,
}


def _is_year_pattern(text: str, start: int) -> int:
    """检测以 start 开头的"年份"模式,返回匹配长度,否则 0。

    年份特征:连续 4 个 (或 2-3 个) 数字+零的组合,后面跟 "年" / "级" / "届" 等量词
    例如: 二零二六年 → 4,  二零二五 → 4
    """
    n = len(text)
    if start >= n or text[start] not in _CN_DIGITS or text[start] == "零":
        return 0

    i = start
    seq_len = 0
    has_zero = False
    while i < n and text[i] in _CN_DIGITS:
        if text[i] == "零":
            has_zero = True
        seq_len += 1
        i += 1
        # 限制年份长度 (2-4 位)
        if seq_len > 4:
            break

    # 后续是"年/级/届/度"才算年份
    if i < n and text[i] in "年度届级":
        return seq_len
    # 或有零 + 长度≥3 的也视作年份(容错)
    if has_zero and seq_len >= 3:
        return seq_len
    return 0


def _year_pattern_to_str(text: str, start: int, length: int) -> str:
    """将年份模式直接转字符串。"""
    return "".join(str(_CN_DIGIT_VAL[text[i]]) for i in range(start, start + length))


def _parse_unit_number(text: str, start: int) -> tuple:
    """解析带单位的中文数字 (十/百/千/万/亿)。

    Returns:
        (value, consumed_chars) 或 (None, 0)
    """
    n = len(text)
    if start >= n:
        return None, 0

    section = 0
    digit = 0
    total = 0
    had_digit = False
    i = start
    last_was_zero = False

    while i < n:
        ch = text[i]
        if ch in _CN_DIGIT_VAL or ch == "两":
            # "两" 口语中表示 2 (如 "两千万"), 在单位数内部按 2 处理
            d = _CN_DIGIT_VAL[ch] if ch in _CN_DIGIT_VAL else 2
            if d == 0:
                last_was_zero = True
            else:
                digit = d
                had_digit = True
                last_was_zero = False
            i += 1
            continue

        if ch in _CN_UNITS:
            unit_val = _CN_UNITS[ch]
            if unit_val >= 10000:
                # 万/亿: 段收尾。
                # 亿是最高级单位, 前面累计的 total 需整体并入亿级 ("三千万亿" = 3千万 × 1亿);
                # 万是低级单位, 只把当前段累加到 total ("五亿三千万" = 5亿 + 3千万)
                if digit:
                    section += digit
                    digit = 0
                if unit_val >= 100000000:
                    total = (total + section) * unit_val
                else:
                    total += section * unit_val
                section = 0
                had_digit = False
                last_was_zero = False
                i += 1
                continue
            else:
                # 十/百/千
                if ch == "十" and not had_digit and not digit:
                    # 单独的"十"或段开头的"十" → 10
                    section += 10
                    had_digit = True
                else:
                    if digit:
                        section += digit * unit_val
                        digit = 0
                    elif not last_was_zero:
                        # "百" "千" 缺前数字 → 视为 1
                        section += unit_val
                    had_digit = True
                last_was_zero = False
                i += 1
                continue

        # 非数字字符,停止
        break

    # 收尾
    if digit:
        section += digit

    if total or section or had_digit:
        return total + section, i - start

    return None, 0


def _parse_serial_digits(text: str, start: int) -> tuple:
    """解析连续数字串 (像 "1234" 一样)。

    用于: "二零二六" → 2026, "一二三" → 123
    Returns:
        (value_str, consumed_chars) 或 (None, 0)
    """
    n = len(text)
    i = start
    chars = []
    while i < n and text[i] in _CN_DIGIT_VAL:
        chars.append(str(_CN_DIGIT_VAL[text[i]]))
        i += 1
    if len(chars) >= 1:
        return "".join(chars), i - start
    return None, 0


# ── 公开 API ──────────────────────────────────────────────────────


def itn_normalize(text: str) -> str:
    """对一段 ASR 文本做 ITN (Inverse Text Normalization)。

    Examples:
        >>> itn_normalize("二零二六年")
        '2026年'
        >>> itn_normalize("我今年五十六岁")
        '我今年56岁'
        >>> itn_normalize("百分之二十")
        '20%'
    """
    if not text:
        return text

    # 快速跳过:无任何中文数字字符、单位或"百分"就返回原文本
    _skip_chars = _CN_DIGITS + "十百千万亿" + "百分之"
    if not any(c in text for c in _skip_chars):
        return text

    result = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # ── "百分之 X" ──
        if text[i:i + 3] == "百分之":
            j = i + 3
            num_val, consumed = _parse_unit_number(text, j)
            if num_val is not None and consumed > 0:
                result.append(f"{num_val}%")
                i = j + consumed
                continue
            # 解析失败,原样输出
            result.append(ch)
            i += 1
            continue

        # ── "十" 开头 (是单位但不在 _CN_DIGIT_VAL) ──
        if ch == "十":
            num_val, consumed = _parse_unit_number(text, i)
            if num_val is not None and consumed > 0:
                result.append(str(num_val))
                i += consumed
                continue

        # ── "两" 开头: 后跟计数单位时按数字解析 (如 "两千万"),
        #    否则原样输出 (避免 "两个人" → "2个人")
        if ch == "两":
            if i + 1 < n and text[i + 1] in "十百千万亿":
                num_val, consumed = _parse_unit_number(text, i)
                if num_val is not None and consumed > 0:
                    result.append(str(num_val))
                    i += consumed
                    continue
            result.append(ch)
            i += 1
            continue

        # ── 非数字字符 ──
        if ch not in _CN_DIGIT_VAL:
            result.append(ch)
            i += 1
            continue

        # ── 数字字符开头 ──
        # 优先检查年份模式
        year_len = _is_year_pattern(text, i)
        if year_len > 0:
            result.append(_year_pattern_to_str(text, i, year_len))
            i += year_len
            continue

        # 检查后续是否带 "点" 小数
        # 找 "X点Y" 模式
        int_end = i
        while int_end < n and text[int_end] in _CN_DIGIT_VAL:
            int_end += 1

        if int_end < n and text[int_end] == "点" and int_end + 1 < n and text[int_end + 1] in _CN_DIGIT_VAL:
            # 整数+小数
            int_part = "".join(str(_CN_DIGIT_VAL[text[k]]) for k in range(i, int_end))
            after_dot = int_end + 1
            frac_chars = []
            while after_dot < n and text[after_dot] in _CN_DIGIT_VAL:
                frac_chars.append(str(_CN_DIGIT_VAL[text[after_dot]]))
                after_dot += 1
            result.append(f"{int_part}.{''.join(frac_chars)}")
            i = after_dot
            continue

        # 检查是否带单位 (十/百/千/万/亿),包含 "十" 开头的情况
        if int_end < n and text[int_end] in "十百千万亿":
            num_val, consumed = _parse_unit_number(text, i)
            if num_val is not None and consumed > 0:
                result.append(str(num_val))
                i += consumed
                continue

        # 纯连续数字串 (像 "1234")
        # 但要避免把孤立的"一"也转 — 看后面是否是其他数字字符
        # "一二三" → 123, 但单个"一" 后面紧跟 "点"/"年" 也能转
        next_ch = text[int_end] if int_end < n else ""
        if next_ch in "年度届级月日点" or (int_end - i) >= 2:
            # 多位或带量词 → 当作数字串
            val_str, consumed = _parse_serial_digits(text, i)
            if val_str is not None and consumed >= 2:
                result.append(val_str)
                i += consumed
                continue

        # 兜底:单数字字符也转 (因为前面已经确认有数字)
        result.append(str(_CN_DIGIT_VAL[ch]))
        i += 1

    normalized = "".join(result)
    if normalized != text:
        logger.debug("[ITN] %r → %r", text, normalized)
    return normalized
