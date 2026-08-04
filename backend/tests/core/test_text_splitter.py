"""
text_splitter 测试 — _find_sentence_splits 句界识别。

覆盖:
  1. 中文省略号 '……' (连续 2+ 个 U+2026) 作为句界, split 点在最后一个省略号后
  2. 英文句点切分 ('Hello world. Next.')
  3. 小数点不切分 ('3.5 版本')
"""
from app.core.text_splitter import _find_sentence_splits


class TestFindSentenceSplits:
    """_find_sentence_splits 句界识别."""

    def test_chinese_ellipsis_is_sentence_boundary(self):
        """中文省略号 '……' 作为句界, split_after 位于最后一个省略号之后."""
        assert _find_sentence_splits("第一句……第二句") == [5]

    def test_english_period_splits_sentences(self):
        """英文句点 (后跟空格或行尾) 切分句子."""
        assert _find_sentence_splits("Hello world. Next.") == [12, 18]

    def test_decimal_point_not_split(self):
        """小数点两侧是数字时不切分."""
        assert _find_sentence_splits("3.5 版本") == []
