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


class TestNonNumericCollocations:
    """固定搭配里的"数字"字不表数量, 不能被 ITN 转换。

    这是 ITN 的固有难点: "一" 既可作数词 ("一个人"), 也可作构词成分
    ("说一下")。单靠上下文无法完全判定, 因此用词表兜住高频搭配。
    曾出现 "说一下" → "说1下" 的误转。
    """

    def test_yixia(self):
        assert itn_normalize("说一下") == "说一下"

    def test_yixia_with_prefix(self):
        assert itn_normalize("我们来说一下这个") == "我们来说一下这个"

    def test_yi_series(self):
        for word in ("一直", "一起", "一样", "一定", "一般", "一些", "一边"):
            assert itn_normalize(word) == word

    def test_shifen_means_very(self):
        """"十分" 作"非常"解时不转 (十分感谢 ≠ 10分感谢)。"""
        assert itn_normalize("十分感谢") == "十分感谢"

    def test_shifen_with_time_unit_still_converts(self):
        """"十分钟" 里的"十分"是数量, 仍要转。"""
        assert itn_normalize("十分钟") == "10分钟"

    def test_decimal_not_treated_as_collocation(self):
        """"一点五" 是小数, 不能被"一点"的固定搭配规则挡住。"""
        assert itn_normalize("一点五") == "1.5"

    def test_quantity_still_converts(self):
        """真实数量仍要转换, 修复不能回退已有能力。"""
        assert itn_normalize("第一个") == "第1个"
        assert itn_normalize("一个人") == "1个人"
        assert itn_normalize("十个人") == "10个人"


class TestIdiomsNotSplitIntoDigits:
    """成语里的数字字不能被拆成阿拉伯数字。

    孤立数字字只有后接量词时才算数量。"四舍五入"的"四"后跟"舍"
    (非量词), 因此保持原样 —— 否则会输出 "4舍5入"。
    """

    def test_sishewuru(self):
        assert itn_normalize("四舍五入") == "四舍五入"

    def test_sishewuru_in_sentence(self):
        assert itn_normalize("进行一个四舍五入") == "进行1个四舍五入"

    def test_number_idioms(self):
        for word in ("三心二意", "五花八门", "七上八下", "一模一样", "一见钟情"):
            assert itn_normalize(word) == word

    def test_measure_word_still_converts(self):
        """后接量词时仍是数量, 不能因为成语规则而回退。"""
        assert itn_normalize("一个人") == "1个人"
        assert itn_normalize("第一个") == "第1个"
        assert itn_normalize("一年") == "1年"

    def test_plain_word_stays(self):
        """非量词后接时保持原样 (一共 ≠ 1共)。"""
        assert itn_normalize("一共") == "一共"
