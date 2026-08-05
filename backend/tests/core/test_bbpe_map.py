"""
bbpe_map 测试 — byte-level BPE 解码与逐字符时间戳映射.

覆盖:
  1. decode_token_bytes: 单字节符号 → UTF-8 字节流 (中文 3 字节/字)
  2. ▁ 词边界 → 空格 (仅当上一字节是 ASCII 可打印 0x20-0x7E)
  3. 特殊符号 (<blk>/<sos/eos>/<unk>) 跳过, 不产生字节
  4. tokens_to_char_times: 字符 → 起始字节 → token 时间戳 映射
  5. 非法 UTF-8 / 长度不匹配 → None
"""
from app.core.bbpe_map import (
    BYTE_TO_BCHAR,
    SPACE_ESCAPE,
    decode_token_bytes,
    tokens_to_char_times,
)


def _byte_sym(b: int) -> str:
    """字节值 → 单字节 BPE 符号."""
    return BYTE_TO_BCHAR[b]


class TestDecodeTokenBytes:
    """逐字节解码: 符号列表 → 字节流 + 每 token 字节数."""

    def test_chinese_utf8_bytes(self):
        """中文 '你好' (6 字节) 以单字节符号给出时解码回原字节."""
        raw = "你好".encode("utf-8")
        tokens = [_byte_sym(b) for b in raw]
        out, sizes = decode_token_bytes(tokens)
        assert out == raw
        assert sizes == [1] * 6

    def test_space_escape_only_between_printable_ascii(self):
        """▁ 仅在上一字节是 ASCII 可打印 (0x20-0x7E) 时还原为空格."""
        # "hello": 字节流前加词边界 ▁ → 解码后前导空格被丢弃 (out 为空时不加)
        out, _ = decode_token_bytes([SPACE_ESCAPE + "h", "e", "l", "l", "o"])
        assert out == b"hello"
        # 中文后加词边界: 上一字节是非 ASCII 字节 → 不加空格
        zh = "你".encode("utf-8")
        out, _ = decode_token_bytes(
            [_byte_sym(b) for b in zh] + [SPACE_ESCAPE + _byte_sym(ord("a"))]
        )
        assert out == zh + b"a"

    def test_special_symbols_skipped(self):
        """<unk> 等特殊符号跳过, 不产生字节, 计 0 字节."""
        raw = "好".encode("utf-8")
        tokens = [_byte_sym(b) for b in raw[:1]] + ["<unk>"] + [_byte_sym(b) for b in raw[1:]]
        out, sizes = decode_token_bytes(tokens)
        assert out == raw
        assert sizes == [1, 0, 1, 1]


class TestTokensToCharTimes:
    """逐字符开始时间重建."""

    def test_char_start_time_from_token(self):
        """字符时间 = 其起始字节所属 token 的时间戳."""
        raw = "你好".encode("utf-8")
        tokens = [_byte_sym(b) for b in raw]
        timestamps = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        char_times, text = tokens_to_char_times(tokens, timestamps)
        assert text == "你好"
        # '你' 起始字节 = 第 0 字节 → token0 = 1.0; '好' → 第 3 字节 → token3 = 1.3
        assert char_times == [1.0, 1.3]

    def test_multi_byte_token_time(self):
        """多字节 token (BPE 合并) 内字符共享 token 时间戳."""
        raw = "你".encode("utf-8")
        tokens = [_byte_sym(raw[0]) + _byte_sym(raw[1]) + _byte_sym(raw[2])]
        char_times, text = tokens_to_char_times(tokens, [2.5])
        assert text == "你"
        assert char_times == [2.5]

    def test_unk_skipped_keeps_alignment(self):
        """<unk> 跳过 (0 字节) 不破坏后续字符的时间映射."""
        raw = "好".encode("utf-8")
        tokens = ["<unk>"] + [_byte_sym(b) for b in raw[:1]]
        tokens += [_byte_sym(b) for b in raw[1:]]
        timestamps = [0.5] + [1.0, 1.1, 1.2]
        char_times, text = tokens_to_char_times(tokens, timestamps)
        assert text == "好"
        assert char_times == [1.0]  # 起始字节属于跳过 <unk> 后的第 1 个 token

    def test_invalid_utf8_returns_none(self):
        """非法 UTF-8 字节流 → None (调用方回退比例分配)."""
        assert tokens_to_char_times([_byte_sym(0xFF)], [1.0]) is None

    def test_length_mismatch_returns_none(self):
        """tokens 与 timestamps 长度不一致 → None."""
        assert tokens_to_char_times(["a"], []) is None
        assert tokens_to_char_times([], []) is None
