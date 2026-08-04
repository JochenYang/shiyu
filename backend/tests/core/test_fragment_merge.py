"""
跨段碎片合并测试 (pipeline.SubtitlePipeline._merge_fragments).

背景: ASR 分段 (Zipformer CTC / VAD) 常把句子切碎在词中间
("通过改" + "变整个游戏"), 且可能输出纯标点段 (孤立 ".").
v2.2 在 _stage_correct 前做碎片合并 + 孤立段过滤 + 段间子串去重,
避免碎片段被标点规则补上错误句号 (如 "通过改。").
"""
from app.core.pipeline import PipelineContext, SubtitlePipeline


def _merge(segments):
    return SubtitlePipeline._merge_fragments(segments)


class TestFragmentMerge:
    def test_merges_cjk_fragments_without_space(self):
        """中文碎片直接拼接: '通过改' + '变整个游戏' → '通过改变整个游戏'."""
        segs = [
            {"start": 1.6, "end": 6.7, "text": "我们从添加效果开始，我们可以通过改"},
            {"start": 6.7, "end": 8.5, "text": "变整个游戏的时间尺度来实现"},
        ]
        out = _merge(segs)
        assert len(out) == 1
        assert out[0]["text"] == "我们从添加效果开始，我们可以通过改变整个游戏的时间尺度来实现"
        assert out[0]["start"] == 1.6
        assert out[0]["end"] == 8.5

    def test_terminal_punctuation_prevents_merge(self):
        """prev 以句号/问号/感叹号/省略号结尾 → 视为完整句, 不合并."""
        for terminal in ["。", "？", "！", "…"]:
            segs = [
                {"start": 0.0, "end": 2.0, "text": f"这是第一句{terminal}"},
                {"start": 2.0, "end": 4.0, "text": "这是第二句"},
            ]
            assert len(_merge(segs)) == 2, f"terminal={terminal}"

    def test_merge_stops_at_duration_limit(self):
        """合并后时长超过 8s → 保持独立段."""
        segs = [
            {"start": 0.0, "end": 8.0, "text": "第一段没有终止标点"},
            {"start": 8.0, "end": 25.0, "text": "第二段很长但合并超时"},
        ]
        out = _merge(segs)
        assert len(out) == 2

    def test_merge_stops_at_char_limit(self):
        """合并后字符数超过 32 → 保持独立段 (避免超长字幕)."""
        segs = [
            {"start": 0.0, "end": 4.0, "text": "这是一个很长的没有终止标点的完整句子内容"},
            {"start": 4.0, "end": 6.0, "text": "下一段内容也很长，这里还有更多"},
        ]
        out = _merge(segs)
        assert len(out) == 2

    def test_dedups_in_segment_repeated_phrase(self):
        """段内连续重复短语 (ASR 整句重复识别) 被清理."""
        segs = [
            {"start": 0.0, "end": 7.0, "text":
             "包括我们的计时器，如果我们移除我们的计时器，如果我们移除玩家的碰撞体"},
        ]
        out = _merge(segs)
        assert out[0]["text"] == "包括我们的计时器，如果我们移除玩家的碰撞体"

    def test_filters_lone_punctuation_segments(self):
        """孤立标点段 (如 ".") 被过滤, 不影响相邻段合并."""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "正常内容"},
            {"start": 1.0, "end": 1.3, "text": "."},
            {"start": 1.3, "end": 2.0, "text": "后续内容"},
        ]
        out = _merge(segs)
        assert len(out) == 1
        assert out[0]["text"] == "正常内容后续内容"

    def test_dedups_repeated_phrase_at_next_head(self):
        """ASR 重叠识别: prev 尾部短语重复出现在 curr 开头附近 → 删除重复.
        残余 "将 timescale" (11 字) 超豁免阈值, 不合并, 保持独立段."""
        segs = [
            {"start": 0.0, "end": 7.0, "text": "你死了之后，我们将修改时间尺度输入 engine"},
            {"start": 7.0, "end": 9.0, "text": "将修改时间尺度输入 engine timescale"},
        ]
        out = _merge(segs)
        assert len(out) == 2
        # 重复短语被删除, 只保留残余
        assert out[0]["text"] == "你死了之后，我们将修改时间尺度输入 engine"
        assert out[1]["text"] == "将 timescale"

    def test_short_dedup_tail_merges_with_exemption(self):
        """去重残余 <=8 字 (重复句尾巴) → 允许合并到 36 字, 不独立成碎片."""
        segs = [
            {"start": 0.0, "end": 6.0, "text": "你死了之后，我们将修改时间尺度输入 engine"},
            {"start": 6.0, "end": 8.0, "text": "将修改时间尺度输入 engine ts"},
        ]
        out = _merge(segs)
        assert len(out) == 1
        assert out[0]["text"] == "你死了之后，我们将修改时间尺度输入 engine将 ts"

    def test_drops_duplicate_short_segment(self):
        """规则 A: ASR 整句重复识别的短段 (如 0.86s 的
        '将修改时间尺度输入 engine timescale') 与相邻段高度重叠 → 整段丢弃."""
        segs = [
            {"start": 13.9, "end": 21.1, "text": "脚本再打印出你死了之后我们将修改时间尺度输入 en gine"},
            {"start": 21.1, "end": 21.9, "text": "将修改时间尺度输入 engine timescale"},
        ]
        out = _merge(segs)
        assert len(out) == 1
        assert out[0]["text"] == "脚本再打印出你死了之后我们将修改时间尺度输入 en gine"

    def test_keeps_short_segment_without_overlap(self):
        """规则 A 不误伤: 短段与相邻段无重叠 → 保留并正常合并."""
        segs = [
            {"start": 0.0, "end": 4.0, "text": "这是一个完整的句子内容"},
            {"start": 4.0, "end": 4.6, "text": "然后我们继续"},
        ]
        out = _merge(segs)
        assert len(out) == 1
        assert out[0]["text"] == "这是一个完整的句子内容然后我们继续"

    def test_merges_very_short_segment_into_prev(self):
        """规则 B: 极短段 (< 1.5s) 读不完, 强制并入前段 (豁免长度上限)."""
        segs = [
            {"start": 107.7, "end": 112.4, "text": "free 意味着移除那个节点，所以我们在这里访问玩家"},
            {"start": 112.4, "end": 112.9, "text": "获取 collision"},
        ]
        out = _merge(segs)
        assert len(out) == 1
        assert out[0]["text"] == "free 意味着移除那个节点，所以我们在这里访问玩家获取 collision"

    def test_ascii_join_uses_space(self):
        """英文词相邻 → 空格连接."""
        segs = [
            {"start": 0.0, "end": 2.0, "text": "the engine"},
            {"start": 2.0, "end": 4.0, "text": "timescale is set"},
        ]
        out = _merge(segs)
        assert out[0]["text"] == "the engine timescale is set"

    def test_empty_and_punct_only_input(self):
        """空输入 / 全标点段 → 空结果."""
        assert _merge([]) == []
        assert _merge([{"start": 0, "end": 1, "text": "..."}]) == []


class TestPipelineMergeIntegration:
    def test_stage_correct_merges_fragments_before_correction(self):
        """_stage_correct 中碎片先合并再修正, 完整句尾标点阻断合并."""
        ctx = PipelineContext(
            raw_segments=[
                {"start": 0.0, "end": 4.0, "text": "我们可以通过改"},
                {"start": 4.0, "end": 8.0, "text": "变整个游戏。"},
                {"start": 8.0, "end": 12.0, "text": "你死了之后"},
            ],
            enable_deterministic_correction=False,
            enable_punctuation=False,
        )
        SubtitlePipeline(ctx)._stage_correct()
        texts = [s["text"] for s in ctx.corrected_segments]
        assert texts == ["我们可以通过改变整个游戏。", "你死了之后"]
