"""
Byte-level BPE 映射表与解码工具 — sherpa-onnx zipformer-ctc 专用.

sherpa-onnx 的 zipformer-ctc 中文模型 (vocab 1000) 使用 byte-level BPE:
文本先 UTF-8 编码为字节流, 每个字节映射为一个 Unicode 字符符号 (字节 0x00-0xFF
映射到 PRINTABLE_BASE_CHARS[byte]), 词边界以 U+2581 (▁) 标记, 再由
sentencepiece 风格的 BPE 合并成子词 token (如 ``▁ƍĩĴ``).

识别时 sherpa-onnx 输出 ``result.tokens`` (符号列表) 和 ``result.text``
(已解码文本). 本模块复刻 C++ ``SymbolTable::DecodeByteBpe`` 的逐字节解码,
并额外记录「字节 → token 索引」和「字符 → 起始字节」两层映射, 从而把
``result.timestamps`` (逐 token 开始时间) 精确对到每个字符上 —— 这是
zipformer 整段输出场景下消除长段比例插值累积误差的关键.

映射表来源: icefall (k2-fsa) 沿用的 fairseq byte-level BPE 标准映射
(PRINTABLE_BASE_CHARS, 256 项). 该表是事实性数据, 各字节到符号的对应
关系固定; tokens.txt 中的单字节符号即为这些映射字符.
"""
from typing import Dict, List, Optional, Tuple

# 字节 → Unicode 码点 的标准映射表 (icefall/fairseq byte-level BPE).
# 索引 = 字节值 (0-255), 值 = 映射后的 Unicode 码点.
PRINTABLE_BASE_CHARS = [
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267,
    268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
    280, 281, 282, 283, 284, 285, 286, 287, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 288, 289, 290, 291, 292,
    293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304,
    305, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
    321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333,
    334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
    346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357,
    358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369,
    370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381,
    382, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
    395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406,
    407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
    419, 420, 421, 422,
]

# 字节 → 符号字符
BYTE_TO_BCHAR: Dict[int, str] = {
    b: chr(PRINTABLE_BASE_CHARS[b]) for b in range(256)
}
# 符号字符 → 字节 (逆映射)
BCHAR_TO_BYTE: Dict[str, int] = {bc: b for b, bc in BYTE_TO_BCHAR.items()}

# 词边界标记 (sentencepiece 风格), 解码时还原为空格
SPACE_ESCAPE = "\u2581"  # ▁

# 特殊符号: 不产生字节 (识别输出中除 <unk> 外不应出现)
SPECIAL_SYMBOLS = frozenset({"<blk>", "<sos/eos>", "<unk>", "<SIL>", "</s>"})


def decode_token_bytes(tokens: List[str]) -> Tuple[bytes, List[int]]:
    """把 token 符号列表解码为字节流, 返回 (字节流, 每个 token 覆盖的字节数).

    复刻 sherpa-onnx ``SymbolTable::DecodeByteBpe`` 的逐字节逻辑:
      - ``▁``: 词边界, 若上一字节是 ASCII 可打印且非空格则追加空格 (0x20)
      - 映射符号: 还原为原始字节
      - 其他 ASCII 可打印字符: 原样保留
      - 特殊符号 (``<unk>`` 等): 跳过, 不产生字节

    Args:
        tokens: sherpa-onnx ``result.tokens`` (符号字符串列表).

    Returns:
        (bytes, token_sizes): 拼接后的字节流, 及与 tokens 等长的每 token 字节数
        (被跳过的 token 计 0).
    """
    out = bytearray()
    token_sizes: List[int] = []
    for sym in tokens:
        size_before = len(out)
        if sym in SPECIAL_SYMBOLS:
            token_sizes.append(0)
            continue
        for ch in sym:
            if ch == SPACE_ESCAPE:
                # C++ 端 std::isprint 为 C locale: 仅 0x20-0x7E 可打印
                if out and out[-1] != 0x20 and 0x20 <= out[-1] <= 0x7E:
                    out.append(0x20)
            elif ch in BCHAR_TO_BYTE:
                out.append(BCHAR_TO_BYTE[ch])
            elif 0x20 <= ord(ch) <= 0x7E:
                out.append(ord(ch))
            # 其余 (应不出现): 跳过
        token_sizes.append(len(out) - size_before)
    return bytes(out), token_sizes


def tokens_to_char_times(
    tokens: List[str],
    timestamps: List[float],
) -> Optional[Tuple[List[float], str]]:
    """从 token 符号与逐 token 时间戳重建逐字符开始时间.

    Args:
        tokens: sherpa-onnx ``result.tokens``.
        timestamps: sherpa-onnx ``result.timestamps`` (与 tokens 等长, 秒).

    Returns:
        (char_times, decoded_text): 每个字符的开始时间 (与 decoded_text 等长),
        以及解码后的文本; 解码失败 (非法 UTF-8 / 长度不匹配) 返回 None.
    """
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return None
    try:
        raw, token_sizes = decode_token_bytes(tokens)
        text = raw.decode("utf-8")
    except (UnicodeDecodeError, TypeError):
        return None

    if not text:
        return None

    # 字节 offset → token 索引 (token j 覆盖 [cum[j], cum[j+1]))
    cum: List[int] = []
    acc = 0
    for size in token_sizes:
        acc += size
        cum.append(acc)

    def token_of(byte_offset: int) -> int:
        # 二分: 找最小的 j 使 cum[j] > byte_offset
        lo, hi = 0, len(cum)
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] > byte_offset:
                hi = mid
            else:
                lo = mid + 1
        return min(lo, len(cum) - 1)

    # 字符 → 起始字节 offset → token → 时间戳
    char_times: List[float] = []
    offset = 0
    for ch in text:
        char_times.append(float(timestamps[token_of(offset)]))
        offset += len(ch.encode("utf-8"))
    return char_times, text
