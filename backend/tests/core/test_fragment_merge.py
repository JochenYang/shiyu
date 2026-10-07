"""
段间去重与清理测试 (pipeline.SubtitlePipeline._dedup_segments)。

背景: 引擎现在在已测量的词边界上做 DP 断句并输出最终字幕行。因此 pipeline
**不再**按长度合并相邻段 —— 那会把引擎断好的句子重新拼回去 (实测 3 行被拼成
1 行 4.6s)。本文件覆盖保留下来的清理逻辑: 空段/纯标点段过滤、ASR 重复识别
产物的去重、以及「绝不合并」这一不变式。
"""
from app.core.pipeline import PipelineContext, SubtitlePipeline


def _dedup(segments):
    return SubtitlePipeline._dedup_segments(segments)


class TestNoLengthMerging:
    """核心不变式: 相邻段不因长度/时长而被合并。"""

    def test_short_adjacent_segments_are_kept_separate(self):
        """两段短句相邻且无标点 → 仍然保持两段 (旧版会合并成一段)。"""
        segs = [
            {"start": 0.0, "end": 1.0, "text": "对我做了介绍"},
            {"start": 1.0, "end": 2.0, "text": "那么我想说的是"},
        ]
        out = _dedup(segs)
        assert len(out) == 2
        assert [s["text"] for s in out] == ["对我做了介绍", "那么我想说的是"]

    def test_dp_segmented_captions_survive_pipeline(self):
        """引擎 DP 断句的 3 行不被合并回 1 行 (回归: 曾把 3 行拼成 4.6s 一行)。"""
        segs = [
            {"start": 0.01, "end": 1.45, "text": "对我做了介绍"},
            {"start": 1.45, "end": 3.53, "text": "那么我想说的是呢大家如果"},
            {"start": 3.53, "end": 4.65, "text": "对我的研究感兴趣呢"},
        ]
        out = _dedup(segs)
        assert len(out) == 3
        assert out[0]["start"] == 0.01
        assert out[-1]["end"] == 4.65

    def test_cjk_fragments_are_not_joined(self):
        """词中间被切断的相邻段不再自动拼接 (由引擎断句负责)。"""
        segs = [
            {"start": 1.6, "end": 6.7, "text": "我们可以通过改"},
            {"start": 6.7, "end": 8.5, "text": "变整个游戏的时间尺度来实现"},
        ]
        out = _dedup(segs)
        assert len(out) == 2

    def test_input_is_not_mutated(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "第一句"},
                {"start": 1.0, "end": 2.0, "text": "第二句"}]
        before = [dict(s) for s in segs]
        _dedup(segs)
        assert segs == before


class TestCleanup:
    """保留的清理能力。"""

    def test_filters_empty_and_lone_punctuation_segments(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "."},
            {"start": 3.0, "end": 4.0, "text": "有内容"},
        ]
        out = _dedup(segs)
        assert [s["text"] for s in out] == ["有内容"]

    def test_dedups_repeated_phrase_at_next_head(self):
        """ASR 重叠识别: 前段尾部短语重复出现在后段开头 → 整段被判定为重复而丢弃。

        该短段几乎全部由重复内容构成 (9/11 字重叠), 因此被整段丢弃 —— 这是
        期望行为: 内容是前段的重复, 保留会产出重复字幕。
        """
        segs = [
            {"start": 0.0, "end": 3.0, "text": "将修改时间尺度输入 engine"},
            {"start": 3.0, "end": 3.9, "text": "输入 engine timescale"},
        ]
        out = _dedup(segs)
        assert len(out) == 1
        assert out[0]["text"] == "将修改时间尺度输入 engine"

    def test_partial_overlap_keeps_the_new_tail(self):
        """后段大部分是新内容时, 只去掉重复的那一小段, 新内容保留。"""
        segs = [
            {"start": 0.0, "end": 3.0, "text": "这一段说的是把模型下载下来"},
            {"start": 3.0, "end": 6.0, "text": "把模型下载下来以后我们还要做校验和断句处理"},
        ]
        out = _dedup(segs)
        assert len(out) == 2
        assert "校验和断句处理" in out[1]["text"]
        # 重复部分已去掉, 不该原样再出现一次
        assert out[1]["text"].count("把模型下载下来") == 0

    def test_drops_duplicate_short_segment(self):
        """整句重复识别的短段被丢弃。"""
        segs = [
            {"start": 0.0, "end": 3.0, "text": "将修改时间尺度输入 engine timescale"},
            {"start": 3.0, "end": 3.9, "text": "输入 engine timescale"},
        ]
        out = _dedup(segs)
        assert len(out) == 1

    def test_dedups_in_segment_repeated_phrase(self):
        """段内连续重复短语被清理。"""
        segs = [{"start": 0.0, "end": 3.0, "text": "我们的计时器如果我们移除我们的计时器如果我们移除"}]
        out = _dedup(segs)
        assert out[0]["text"] == "我们的计时器如果我们移除"

    def test_keeps_short_segment_without_overlap(self):
        """不与前段重叠的短段保留 (不再被长度规则吞掉)。"""
        segs = [
            {"start": 0.0, "end": 3.0, "text": "前面一整句话在这里结束"},
            {"start": 3.0, "end": 3.8, "text": "好"},
        ]
        out = _dedup(segs)
        assert len(out) == 2
        assert out[1]["text"] == "好"

    def test_empty_input(self):
        assert _dedup([]) == []


class TestPipelineIntegration:
    """_stage_correct 用去重而非合并。"""

    def test_stage_correct_keeps_engine_segmentation(self):
        ctx = PipelineContext(
            skip_asr=True,
            raw_segments=[
                {"start": 0.0, "end": 1.0, "text": "我们可以通过改"},
                {"start": 1.0, "end": 2.0, "text": "变整个游戏"},
            ],
            engines_results={},
            enable_deterministic_correction=False,
            enable_punctuation=False,
            enable_validate=False,
        )
        SubtitlePipeline(ctx)._stage_correct()
        assert len(ctx.corrected_segments) == 2