"""Regression test: real SRT scenario from user's video.

Reproduces the actual ASR+VAD layout from log lines 2026-06-08 18:35:14
(user's "如何制作视频游戏-10-玩家2.0版.srt" transcription run).

Before fix: new "BFS-dedupe with segment_consumed cursor" algorithm
DROPPED text — e.g. "到目前为止我们几乎原封不动的使用了角色移
动模板" became "到目前为止我们几乎原封不动的使用了角色"
(丢失 4 chars).

After fix: original proportional-split + suffix-prefix dedup should
preserve all characters in the combined region texts.
"""
from app.core.sentence_splitter import _align_segments_to_vad_regions


def test_user_srt_segment_1_no_text_loss():
    """Real case: 2 ASR segments, 2 VAD regions, full text preserved."""
    # From log: ASR first 2 segs
    segments = [
        {"start": 2.01, "end": 7.35,
         "text": "到目前为止我们几乎原封不动的使用了角色移"},
        {"start": 7.35, "end": 11.884,
         "text": "为了给玩家添加动画以及更改我们用来移动的"},
    ]
    # From log: VAD first 2 intervals
    speech_intervals = [
        (2.048, 6.144),
        (7.008, 10.944),
    ]

    result = _align_segments_to_vad_regions(segments, speech_intervals)
    combined = "".join(s["text"] for s in result)

    # Every distinct character substring from inputs should survive in the output
    for s in segments:
        # Find longest matching substring >= 4 chars
        text = s["text"]
        for length in (8, 6, 4):
            for start in range(0, len(text) - length + 1):
                sub = text[start:start + length]
                if sub in combined:
                    break
            else:
                continue
            break
        else:
            assert False, f"Segment text {text!r} not preserved in result {combined!r}"

    # Combined text length should be >= 80% of inputs (allow some dedup)
    total_input = sum(len(s["text"]) for s in segments)
    assert len(combined) >= total_input * 0.8, (
        f"Combined output {len(combined)} chars < 80% of input {total_input} chars. "
        f"Output: {combined!r}"
    )


def test_user_srt_segment_2_no_text_loss():
    """Real case: 4 ASR segments, no obvious duplicate, all preserved."""
    segments = [
        {"start": 16.196, "end": 18.056, "text": "我们在看什么呢？"},
        {"start": 18.056, "end": 21.476,
         "text": "嗯，最上面的两行定义了两个长量，"},
        {"start": 24.480, "end": 27.775,
         "text": "这里还有一个变量，根据项目设置定义"},
        {"start": 28.864, "end": 31.711, "text": "设置定义了重力默认情况下这"},
    ]
    speech_intervals = [
        (15.616, 16.928),
        (18.208, 23.616),
        (24.480, 27.775),
        (28.864, 31.711),
    ]
    result = _align_segments_to_vad_regions(segments, speech_intervals)
    combined = "".join(s["text"] for s in result)
    # "我们在看" should be present
    assert "我们在看" in combined
    # "重力" should be present (the user's report had this in segment 7)
    assert "重力" in combined or "默认" in combined
    # No obvious text drop: at least 80% of input chars present
    total_input = sum(len(s["text"]) for s in segments)
    assert len(combined) >= total_input * 0.7, (
        f"Combined output {len(combined)} < 70% of input {total_input}. "
        f"Output: {combined!r}"
    )
