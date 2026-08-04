"""
字幕校验器 — 参照 JianyingPro OutputValidateProcessor。

剪映在 OutputValidateProcessor 里做的 5 类校验：
  1. continue_space_count        连续空格
  2. word_timestamp_overlap_count 词时间戳重叠
  3. utterance_timestamp_overlap_count 句时间戳重叠
  4. word_timestamp_self_overlap_count  自重叠
  5. utterance_timestamp_self_overlap_count 自重叠

本模块先实现 1+2+3+5 这 4 项 (词级别需要 ASR 输出字级时间戳,我们没有)。
"""
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# 校验阈值
MIN_SEG_DURATION = 0.3        # 最短字幕时长 (秒)
MAX_SEG_DURATION = 12.0        # 最长字幕时长 (秒)
MIN_TEXT_LEN = 2               # 最短字幕字符数
MAX_OVERLAP_RATIO = 0.05       # 允许的相邻段最大时间戳重叠比例 (5%)


def validate_segments(segments: List[Dict]) -> List[Dict]:
    """校验字幕段,返回问题列表 (不修改原数据)。

    Returns:
        问题列表,每个问题是一个 dict:
          {
            "index": int,         # 段索引
            "code": str,          # 错误码
            "message": str,       # 人类可读描述
            "severity": str,      # "error" | "warning"
          }
    """
    issues: List[Dict] = []
    if not segments:
        return issues

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        # ── 1. 字幕时长校验 (模仿剪映 thre_short_case_len / max_line_len 思路) ──
        duration = end - start
        if duration < 0:
            issues.append({
                "index": i, "code": "NEGATIVE_DURATION",
                "message": f"段 {i}: 结束时间 {end} < 起始 {start}",
                "severity": "error",
            })
        elif duration < MIN_SEG_DURATION:
            issues.append({
                "index": i, "code": "TOO_SHORT",
                "message": f"段 {i}: 时长 {duration:.2f}s < {MIN_SEG_DURATION}s",
                "severity": "warning",
            })
        elif duration > MAX_SEG_DURATION:
            issues.append({
                "index": i, "code": "TOO_LONG",
                "message": f"段 {i}: 时长 {duration:.2f}s > {MAX_SEG_DURATION}s",
                "severity": "warning",
            })

        # ── 2. 文本内容校验 ──
        if not text:
            issues.append({
                "index": i, "code": "EMPTY_TEXT",
                "message": f"段 {i}: 文本为空",
                "severity": "warning",
            })
        elif len(text) < MIN_TEXT_LEN:
            issues.append({
                "index": i, "code": "TOO_SHORT_TEXT",
                "message": f"段 {i}: 文本过短 ({len(text)} 字符)",
                "severity": "warning",
            })
        elif "  " in text:
            # 剪映的 continue_space_count 校验
            double_space_count = text.count("  ")
            issues.append({
                "index": i, "code": "DOUBLE_SPACE",
                "message": f"段 {i}: 出现连续空格 ({double_space_count} 处)",
                "severity": "warning",
            })

        # ── 3. 时间戳合理性 (剪映 utterance_timestamp_self_overlap) ──
        if start < 0 or end < 0:
            issues.append({
                "index": i, "code": "NEGATIVE_TIMESTAMP",
                "message": f"段 {i}: 负时间戳 ({start:.3f}, {end:.3f})",
                "severity": "error",
            })

    # ── 4. 相邻段重叠/倒置 (剪映 utterance_timestamp_overlap_count) ──
    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]
        if cur.get("end", 0) > nxt.get("start", 0):
            overlap = cur["end"] - nxt["start"]
            cur_dur = cur["end"] - cur["start"]
            if cur_dur > 0 and overlap / cur_dur > MAX_OVERLAP_RATIO:
                issues.append({
                    "index": i, "code": "OVERLAP",
                    "message": f"段 {i} ↔ {i+1}: 重叠 {overlap:.3f}s ({overlap/cur_dur*100:.1f}%)",
                    "severity": "warning",
                })

    return issues


def log_validation_report(segments: List[Dict], source: str = "") -> Tuple[int, int]:
    """运行校验并把结果打到日志。返回 (error_count, warning_count)。"""
    issues = validate_segments(segments)
    if not issues:
        logger.info("[SubtitleValidator] %s: %d segments passed validation", source, len(segments))
        return 0, 0

    n_err = sum(1 for x in issues if x["severity"] == "error")
    n_warn = sum(1 for x in issues if x["severity"] == "warning")

    if n_err > 0:
        logger.error("[SubtitleValidator] %s: %d errors, %d warnings in %d segments",
                     source, n_err, n_warn, len(segments))
    elif n_warn > 0:
        logger.warning("[SubtitleValidator] %s: %d warnings in %d segments",
                       source, n_warn, len(segments))
    else:
        logger.info("[SubtitleValidator] %s: %d segments passed", source, len(segments))

    # 详细列出前 10 个问题
    for issue in issues[:10]:
        if issue["severity"] == "error":
            logger.error("  %s", issue["message"])
        else:
            logger.warning("  %s", issue["message"])
    if len(issues) > 10:
        logger.info("  ... and %d more issues", len(issues) - 10)

    return n_err, n_warn
