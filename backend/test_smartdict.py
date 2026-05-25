# -*- coding: utf-8 -*-
"""测试智能验算引擎 - SmartDictionary"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.core.asr_corrector import (
    SmartDictionary, DictEntry,
    ASRDeterministicCorrector, load_asr_dictionary,
)

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS: {name}")
    except AssertionError as e:
        FAIL += 1
        print(f"  FAIL: {name} — {e}")

# ============================
# 1. SmartDictionary 基础
# ============================

def test_basic_replacement():
    sd = SmartDictionary()
    sd.add_simple("timescal", "timescale")
    assert sd.apply("set timescal to 2") == "set timescale to 2", "basic replace failed"

test("basic replacement", test_basic_replacement)

def test_multi_candidate():
    """同源多候选：上下文决定选哪个。"""
    sd = SmartDictionary([
        DictEntry("巴蒂", "body",
                  context_hints=["回调", "碰撞", "节点"],
                  word_boundary=True, min_hits=1),
        DictEntry("巴蒂", "Barty",
                  context_hints=["先生", "小姐", "医生"],
                  word_boundary=True, min_hits=1),
    ])
    # 游戏上下文 → 选 body
    r1 = sd.apply("巴蒂回调函数结构体处理代码")
    assert "body" in r1, f"game context should → body, got: {r1}"
    # 姓名上下文 → 选 Barty
    r2 = sd.apply("巴蒂先生今天来参加会议")
    assert "Barty" in r2, f"name context should → Barty, got: {r2}"

test("multi-candidate competition", test_multi_candidate)

def test_word_boundary():
    """词边界：不会把 bar 匹配到 barometer。"""
    sd = SmartDictionary()
    sd.add_simple("bar", "foo")
    result = sd.apply("this is a barometer reading")
    # "bar" in "barometer" should NOT match because of word boundary
    assert "foo" not in result, f"word boundary failed: {result}"

test("word boundary prevents substring match", test_word_boundary)

def test_word_boundary_english():
    """英语词边界：独立 bar 应该被替换。"""
    sd = SmartDictionary()
    sd.add_simple("bar", "foo")
    result = sd.apply("open the bar and run")
    assert "foo" in result, f"independent bar should be replaced: {result}"

test("word boundary allows independent word", test_word_boundary_english)

def test_word_boundary_chinese():
    """中文上下文中的英语词边界。"""
    sd = SmartDictionary()
    sd.add_simple("bar", "foo")
    result = sd.apply("设置 bar 参数")
    assert "foo" in result, f"bar in chinese should be replaced: {result}"

test("word boundary chinese context", test_word_boundary_chinese)

# ============================
# 2. 上下文置信度
# ============================

def test_context_enough_hits():
    """上下文关联词命中足够 → 替换。"""
    sd = SmartDictionary([
        DictEntry("八的", "body的",
                  context_hints=["回调", "碰撞"],
                  word_boundary=True, min_hits=1),
    ])
    result = sd.apply("在八的回调函数中处理碰撞")
    assert "body的" in result, f"context match should apply: {result}"

test("context hits enough → apply", test_context_enough_hits)

def test_context_no_hits():
    """上下文无关联词 → 不替换。"""
    sd = SmartDictionary([
        DictEntry("八的", "body的",
                  context_hints=["回调", "碰撞"],
                  word_boundary=True, min_hits=1),
    ])
    result = sd.apply("八的小朋友今天来玩")
    assert "body的" not in result, f"no context should NOT apply: {result}"

test("context no hits → skip", test_context_no_hits)

def test_context_min_hits_0():
    """min_hits=0 且无 context_hints → 总是替换（兼容旧行为）。"""
    sd = SmartDictionary([
        DictEntry("八的", "body的",
                  context_hints=[],
                  word_boundary=True, min_hits=0),
    ])
    result = sd.apply("八的小朋友")
    # word_boundary=True, 八的前后是"的"和"小"（中文 → 非边界）
    # "八的" — 前是空（pos=0 → 边界），后是"的" (中文, 非字母)
    # after is "的" which is Chinese → not ascii → boundary OK
    # So boundary check: before=' ' (pos=0), after='的' (Chinese) → both_boundary=True
    assert "body的" in result, f"min_hits=0 should always apply: {result}"

test("min_hits=0 always apply", test_context_min_hits_0)

# ============================
# 3. 集成测试
# ============================

def test_integration_with_corrector():
    """完整修正器 + 智能词典。"""
    sd = SmartDictionary([
        DictEntry("巴蒂", "body",
                  context_hints=["回调", "碰撞", "节点"],
                  word_boundary=True, min_hits=1),
    ])
    corrector = ASRDeterministicCorrector(dictionary=sd, enable_punctuation=False, enable_dedup=False)
    segs = [{"start": 0, "end": 1, "text": "巴蒂回调函数体处理碰撞检测代码"}]
    result = corrector.correct(segs)
    assert "body" in result[0]["text"], f"integration failed: {result}"

test("integration with ASRDeterministicCorrector", test_integration_with_corrector)

def test_integration_no_context():
    """上下文不匹配时，集成修正器也不应该替换。"""
    sd = SmartDictionary([
        DictEntry("巴蒂", "body",
                  context_hints=["回调", "碰撞", "节点"],
                  word_boundary=True, min_hits=1),
    ])
    corrector = ASRDeterministicCorrector(dictionary=sd, enable_punctuation=False, enable_dedup=False)
    segs = [{"start": 0, "end": 1, "text": "巴蒂先生是我的老师"}]
    result = corrector.correct(segs)
    assert "body" not in result[0]["text"], f"should not replace without context: {result}"
    assert "巴蒂" in result[0]["text"], f"should keep original: {result}"

test("integration no context → keep original", test_integration_no_context)

# ============================
# 4. 边界情况
# ============================

def test_source_equals_target():
    """源==目标的条目不应该改变文本。"""
    sd = SmartDictionary()
    sd.add(DictEntry("collision", "collision", word_boundary=True, min_hits=0))
    result = sd.apply("handle collision detection")
    assert result == "handle collision detection", f"no-op should not change: {result}"

test("source==target no effect", test_source_equals_target)

def test_empty_input():
    sd = SmartDictionary()
    sd.add_simple("foo", "bar")
    assert sd.apply("") == "", "empty input should stay empty"
    assert sd.apply_to_texts([]) == [], "empty list should stay empty"

test("empty input", test_empty_input)

def test_no_match():
    sd = SmartDictionary()
    sd.add_simple("xyz", "abc")
    result = sd.apply("hello world test")
    assert result == "hello world test", f"no match should keep original: {result}"

test("no match keeps original", test_no_match)

def test_multiple_entries():
    """多个条目按长词优先匹配。"""
    sd = SmartDictionary([
        DictEntry("abc", "XXX", word_boundary=True, min_hits=0),
        DictEntry("abcd", "YYYY", word_boundary=True, min_hits=0),
    ])
    result = sd.apply("abcd test")
    # "abcd" 先匹配（更长），不会匹配 "abc"
    assert "YYYY" in result, f"longer match first: {result}"
    assert "XXX" not in result, f"longer match should prevent shorter: {result}"

test("longer match priority", test_multiple_entries)

# ============================
# 5. 加载默认词典
# ============================

def test_default_dictionary_load():
    """load_asr_dictionary 返回可用的 SmartDictionary。"""
    sd = load_asr_dictionary()
    assert isinstance(sd, SmartDictionary), "should return SmartDictionary"
    assert len(sd._entries) > 0, "should have default entries"
    # 测试 "collision ship" → "collision shape"
    result = sd.apply("collision ship node for unity game physics")
    assert "collision shape" in result, f"default dict should apply: {result}"

test("default dictionary load and apply", test_default_dictionary_load)

def test_default_timescal():
    """timescal → timescale 应该总是替换（word_boundary=False）。"""
    sd = load_asr_dictionary()
    result = sd.apply("set timescal to 0.5")
    assert "timescale" in result, f"timescal should be fixed: {result}"

test("default timescal fix", test_default_timescal)

def test_default_badi_with_context():
    """巴蒂 → body 需要上下文关联词。"""
    sd = load_asr_dictionary()
    # 有游戏上下文 → 替换
    r1 = sd.apply("巴蒂回调函数处理碰撞")
    assert "body" in r1, f"context context should trigger: {r1}"
    # 无上下文 → 不替换
    r2 = sd.apply("巴蒂先生你好")
    assert "body" not in r2, f"no context should NOT trigger: {r2}"
    assert "巴蒂" in r2, f"should keep original: {r2}"

test("default badi with context", test_default_badi_with_context)

# ============================
# 结果
# ============================

print()
print(f"=== Results: {PASS} passed, {FAIL} failed ===")
