"""
ASRDeterministicCorrector / SmartDictionary tests (asr_corrector.py).

覆盖:
  1. 跨段碎片修复: 拼接词写回前段, 后段只移除碎片词 (不重复显示完整词)
  2. 词典替换智能验算: '巴蒂' 无上下文不替换, 命中 'rigidbody' 才替换
  3. 词边界行为: 'timescal' 条目启用词边界后, 独立词 'timescal' 替换为
     'timescale', 正确拼写 'timescales'/'timescale' 不受前缀匹配影响
  4. 默认词典不含 source==target 的自替换死条目

碎片测试用词对选择依据 (见 asr_corrector.py 词表):
  - 'collis' + 'ion' → 'collision': 两碎片均不在 _SHORT_STANDALONE_EN /
    _COMMON_EN_WORDS, 拼接结果在 _COMMON_EN_WORDS
  - 'gravi' + 'ty' → 'gravity': 同上
"""
from app.core.asr_corrector import (
    ASRDeterministicCorrector,
    SmartDictionary,
    _DEFAULT_SMART_DICT,
)


class TestFragmentFix:
    """Phase 1: 跨段碎片修复."""

    def _make_corrector(self):
        """构造修正器: 关闭标点与去重, 隔离碎片修复行为."""
        return ASRDeterministicCorrector(
            enable_punctuation=False,
            enable_dedup=False,
        )

    def test_join_split_word_into_prev_and_remove_from_curr(self):
        """'collis' + 'ion' 拼接为 'collision' 写回前段, 后段不再含完整词."""
        corrector = self._make_corrector()
        segments = [
            {"start": 0.0, "end": 1.0, "text": "这是 collis"},
            {"start": 1.0, "end": 2.0, "text": "ion 的碰撞"},
        ]
        result = corrector.correct(segments)
        assert result[0]["text"] == "这是 collision"
        assert result[1]["text"] == "的碰撞"
        assert "collision" not in result[1]["text"]

    def test_fix_fragments_curr_stays_empty_when_only_fragment(self):
        """后段只有碎片词时, 修复后保持空字符串 (下游 segments_from_transcription 过滤空段)."""
        corrector = self._make_corrector()
        segments = [
            {"start": 0.0, "end": 1.0, "text": "unity 的 gravi"},
            {"start": 1.0, "end": 2.0, "text": "ty"},
        ]
        corrector._fix_fragments(segments)
        assert segments[0]["text"] == "unity 的 gravity"
        assert segments[1]["text"] == ""


class TestSmartDictionary:
    """Phase 2: 智能词典替换 (智能验算)."""

    def test_ba_di_not_replaced_without_context_hint(self):
        """'巴蒂' 无上下文关联词时不被替换 (min_hits=1 未命中)."""
        assert _DEFAULT_SMART_DICT.apply("巴蒂") == "巴蒂"

    def test_ba_di_replaced_with_body_when_rigidbody_in_context(self):
        """'巴蒂' 命中 'rigidbody' 关联词时被替换为 'body'."""
        result = _DEFAULT_SMART_DICT.apply("巴蒂的 rigidbody 属性")
        assert "body" in result
        assert "巴蒂" not in result

    def test_timescal_word_boundary_keeps_correct_spelling(self):
        """'timescal' 默认条目启用词边界:
        独立词 'timescal' → 'timescale';
        正确拼写 'timescales'/'timescale' 的后续字母触发边界拒绝, 文本不变
        (word_boundary=False 时会被破坏成 'timescalees', 已修复)."""
        assert _DEFAULT_SMART_DICT.apply("timescal") == "timescale"
        assert _DEFAULT_SMART_DICT.apply("timescales") == "timescales"
        assert _DEFAULT_SMART_DICT.apply("timescale") == "timescale"

    def test_timescal_replacement_inside_chinese_context(self):
        """'timescal' 后接中文字符 (非 ASCII 字母) 时仍正常替换, 不误伤."""
        assert _DEFAULT_SMART_DICT.apply("timescal的") == "timescale的"

    def test_word_boundary_true_rejects_prefix_inside_timescales(self):
        """词边界: word_boundary=True 的 'timescal' 条目在 'timescales' 中被
        边界检查拒绝, 文本保持不变."""
        sd = SmartDictionary()
        sd.add_simple("timescal", "timescale")
        assert sd.apply("timescales") == "timescales"

    def test_default_dict_has_no_identity_entries(self):
        """默认词典不含 source==target 的自替换死条目."""
        for entry in _DEFAULT_SMART_DICT._entries:
            assert entry.source != entry.target, (
                f"identity entry: {entry.source!r} -> {entry.target!r}"
            )


class TestSentenceEndPunctuation:
    """v2.2.2 黑名单补句号: 悬空字结尾不补, 其余中文句尾补."""

    def _punct(self, text: str) -> str:
        return ASRDeterministicCorrector()._apply_punct_rules(
            text, prev_text="", seg_idx=0, total_segs=1
        )

    def test_complete_sentence_gets_period(self):
        """名词/形容词/动词补语结尾 (打开/状态/地图/体/酷) → 补句号."""
        assert self._punct("亡区域场景点击这里打开") == "亡区域场景点击这里打开。"
        assert self._punct("时我们仍然会处于慢速状态") == "时我们仍然会处于慢速状态。"
        assert self._punct("我们被击中并掉下了地图") == "我们被击中并掉下了地图。"
        assert self._punct("我们将移除玩家的碰撞体") == "我们将移除玩家的碰撞体。"
        assert self._punct("我觉得这可能会更酷") == "我觉得这可能会更酷。"
        assert self._punct("由于唯一进入的是我们的玩家") == "由于唯一进入的是我们的玩家。"

    def test_fragment_tail_gets_no_period(self):
        """段尾是悬空字 (单字及物动词/介词/方位词) → 不补句号."""
        for text in [
            "我们从添加效果开始我们可以通过改",
            "变整个游戏的时间尺度所以我们转到死",
            "请注意游戏重新开始所",
            "需要的时间也是原来的两倍因为游",
            "我们可以在这里添",
            "添加一行代码来指",
            "如果我们不慎遇到危险砰我们被",
            "之后有一个叫做八 的变量这个变量指的是",
            "r d 然后我们可以移除它在当前节点",
            "并将其设置为零点五这意味着我们将以半速运行，然后",
            "需要我们对玩家的引用其实我们这里已",
        ]:
            assert self._punct(text) == text, f"fragment: {text}"

    def test_english_tail_gets_no_period(self):
        """英文结尾无法判断碎片, 一律不补."""
        for text in [
            "将修改时间尺度输入 engine timescale",
            "你会注意到在这段 on body enter",
            "t node 并搜索 collision ship",
            "我们在这里访问玩家获取 collision shi",
        ]:
            assert self._punct(text) == text, f"english tail: {text}"

    def test_existing_punctuation_not_duplicated(self):
        """段尾已有标点 (含逗号) → 不重复补句号."""
        assert self._punct("我觉得这可能会更酷。") == "我觉得这可能会更酷。"
        assert self._punct("玩家的碰撞体，") == "玩家的碰撞体，"

    def test_question_and_exclamation_still_work(self):
        """Rule 2 语气词逻辑不受影响: 吗→？ 啊→！."""
        assert self._punct("你准备好了吗") == "你准备好了吗？"
        assert self._punct("这真是太棒了啊") == "这真是太棒了啊！"
