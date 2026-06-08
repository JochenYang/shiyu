"""
Phase 1 测试: asr_corrector 战术修复

覆盖:
  1. PUNCT_MIN_LEN 常量存在且为 10
  2. _apply_punct_rules 在 >10 字符时补句号
  3. _apply_punct_rules 在 <=10 字符时不补句号 (回归保护)
"""
import pytest

from app.core.asr_corrector import PUNCT_MIN_LEN, ASRDeterministicCorrector


class TestPhase1PunctMinLen:
    """PUNCT_MIN_LEN 常量与行为验证。"""

    def test_punct_min_len_constant_exists(self):
        """PUNCT_MIN_LEN 应导出且等于 10 (原硬编码值)。"""
        assert PUNCT_MIN_LEN == 10, (
            f"PUNCT_MIN_LEN 应为 10, 实际 {PUNCT_MIN_LEN}"
        )

    def test_punct_rule_3_triggers_above_threshold(self):
        """> PUNCT_MIN_LEN 字符且无尾标点 -> 自动补句号。"""
        corrector = ASRDeterministicCorrector()
        # 11 字符无尾标点
        text = "这是一段超过十字的中文文本"
        assert len(text) > PUNCT_MIN_LEN
        result = corrector._apply_punct_rules(
            text, prev_text="", seg_idx=0, total_segs=2,
        )
        assert result.endswith("。"), (
            f"长文本应补句号, 实际: {result!r}"
        )

    def test_punct_rule_3_skips_below_threshold(self):
        """<= PUNCT_MIN_LEN 字符不补句号 (回归: 行为不变)。"""
        corrector = ASRDeterministicCorrector()
        # 恰好 PUNCT_MIN_LEN 字符无尾标点
        text = "a" * PUNCT_MIN_LEN  # 10 ASCII chars, 长度确定
        assert len(text) == PUNCT_MIN_LEN
        result = corrector._apply_punct_rules(
            text, prev_text="", seg_idx=0, total_segs=2,
        )
        # 短文本不补句号
        assert not result.endswith("。"), (
            f"短文本 (=={PUNCT_MIN_LEN}字符) 不应补句号, 实际: {result!r}"
        )

    def test_punct_rule_3_respects_existing_punct(self):
        """已有标点的文本不重复补。"""
        corrector = ASRDeterministicCorrector()
        text = "这是一段已经有句号的文本内容。"  # 14 字符
        result = corrector._apply_punct_rules(
            text, prev_text="", seg_idx=0, total_segs=2,
        )
        # 已有句号, 不应再加一个
        assert result.count("。") == 1, (
            f"已有句号的文本不应重复补, 实际: {result!r}"
        )
