"""
itn 测试 — itn_normalize 中文数字 → 阿拉伯数字转换。

覆盖:
  1. 量词搭配: 五十六岁 → 56岁, 十个人 → 10个人, 二十个人 → 20个人
  2. 百分比与小数: 百分之二十 → 20%, 三点一四 → 3.14
  3. 年份模式: 二零二六年 → 2026年
  4. 万/亿计数单位: 一百万 → 1000000, 三百二十万 → 3200000,
     一亿两千万 → 120000000, 五亿三千万 → 530000000
  5. 普通多位数字: 一百二十三 → 123
"""
from app.core.itn import itn_normalize


class TestItnNormalize:
    """itn_normalize 中文数字转换."""

    def test_age_with_unit(self):
        """带"岁"量词的数字转换 (五十六岁 → 56岁)."""
        assert itn_normalize("五十六岁") == "56岁"

    def test_percent(self):
        """"百分之" 前缀转换 (百分之二十 → 20%)."""
        assert itn_normalize("百分之二十") == "20%"

    def test_year_pattern(self):
        """年份模式按位转换 (二零二六年 → 2026年)."""
        assert itn_normalize("二零二六年") == "2026年"

    def test_hundred_million_plus_ten_million(self):
        """亿后接千万级累加 (一亿两千万 → 120000000)."""
        assert itn_normalize("一亿两千万") == "120000000"

    def test_hundred_million_plus_ten_thousand(self):
        """亿后接万级累加 (五亿三千万 → 530000000)."""
        assert itn_normalize("五亿三千万") == "530000000"

    def test_million_level(self):
        """万级转换 (三百二十万 → 3200000)."""
        assert itn_normalize("三百二十万") == "3200000"

    def test_one_million(self):
        """整百万转换 (一百万 → 1000000)."""
        assert itn_normalize("一百万") == "1000000"

    def test_plain_number(self):
        """普通多位数字 (一百二十三 → 123)."""
        assert itn_normalize("一百二十三") == "123"

    def test_decimal(self):
        """小数转换 (三点一四 → 3.14)."""
        assert itn_normalize("三点一四") == "3.14"

    def test_ten_alone_with_person(self):
        """孤立"十"后跟量词 (十个人 → 10个人)."""
        assert itn_normalize("十个人") == "10个人"

    def test_twenty_with_person(self):
        """十位数字后跟量词 (二十个人 → 20个人)."""
        assert itn_normalize("二十个人") == "20个人"
