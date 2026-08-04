"""
字幕生成流水线编排器 (Shiyu v2.0 — Zipformer-first 极简 4 阶段).

设计原则:
    - 每个阶段独立可测,通过 PipelineContext 字段传递中间结果。
    - 任一阶段异常不会中断流水线,降级到上一个稳定状态。
    - 输入的 segments 视为不可变,每阶段产出新对象。
    - 引擎不可用时降级: 多引擎 → 单引擎 → 空结果
    - VAD 阶段先于 ASR,让每个 ASR 引擎按 VAD 区间切分独立跑

v2.0 简化:
    - 砍 FUSE 阶段: 单 Zipformer 主走, 不再融合
    - 砍 CLAUSE 阶段: sherpa-onnx 自带 VAD 切分已足够
    - 砍 LLM 修正: 主人决策 (太慢)
    - 砍 VAD 单独 stage: sherpa-onnx CTC 自带分段 + Silero-VAD 双重冗余
    - 整合 CORRECTION + POST_PROCESS → CORRECT_TEXT
    - VALIDATE 改为轻量日志 (不独立 stage)

最终 4 阶段:
    1. ASR         主引擎转录 (Zipformer 主走 / Paraformer 备选 / SenseVoice 多语言备选)
    2. CORRECT     glossary 替换 + 确定性修正 (内部含标点) + ITN
    3. VALIDATE    字幕校验 (轻量日志)
    4. FORMAT      SRT/LRC/ASS 格式化
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """4 阶段流水线标识."""

    ASR = "asr"
    CORRECT = "correct"
    VALIDATE = "validate"
    FORMAT = "format"


# 主引擎: Zipformer-CTC-zh (CTC int8, 350MB); paraformer 备选 (CIF 边界准); sensevoice 多语言备选
PRIMARY_ENGINE = "zipformer_local"


@dataclass
class PipelineContext:
    """流水线运行时上下文.

    阶段间依赖:
        engines_results: {engine_name: [seg_dict, ...]}
        -> corrected_segments (CORRECT 后)
        -> final_segments   (VALIDATE 后, 通常等于 corrected)
        -> output_content   (FORMAT 后)
    """

    # ── 输入 ──
    audio_path: str = ""
    audio_cfg: Optional[Dict[str, Any]] = None
    engines: List[str] = field(default_factory=lambda: [PRIMARY_ENGINE])
    engine_getter: Optional[Callable[[str], Any]] = None
    language: str = "auto"
    glossary: Optional[Dict[str, str]] = None

    # ── 阶段开关 ──
    skip_asr: bool = False  # skip_asr=True 保留现有 raw_segments (仅 /correct 端点用)
    enable_deterministic_correction: bool = True
    enable_itn: bool = True
    enable_punctuation: bool = True
    enable_validate: bool = True
    log_validation: bool = True

    # ── 输出格式 ──
    output_format: str = "srt"
    output_content: str = ""
    title: str = "Subtitle"
    # 字幕时间偏移 (秒, 可负): 加到所有段 start/end 上
    # 主人手动微调用: 字幕晚 2s → -2.0, 字幕早 0.5s → +0.5
    subtitle_offset_sec: float = 0.0

    # ── 阶段输出 (None 表示未初始化) ──
    engines_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    # raw_segments = corrected_segments 的早期 alias (向后兼容 API)
    raw_segments: Optional[List[Dict[str, Any]]] = None
    corrected_segments: Optional[List[Dict[str, Any]]] = None
    final_segments: Optional[List[Dict[str, Any]]] = None

    # ── 元数据 ──
    duration: float = 0.0
    engine_meta: Optional[Dict[str, Any]] = None


class SubtitlePipeline:
    """v2.0 字幕生成流水线 (4 阶段, Zipformer-first).

    用法::

        ctx = PipelineContext(audio_path="x.wav", output_format="srt")
        pipeline = SubtitlePipeline(ctx)
        ctx = pipeline.run()
        print(ctx.output_content)

    Attributes:
        ctx: 流水线上下文,贯穿整个生命周期。
    """

    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx

    def run(self) -> PipelineContext:
        """按顺序执行所有启用阶段,返回最终 context."""
        self._stage_asr()
        self._stage_correct()
        self._stage_validate()
        self._stage_apply_offset()
        self._stage_format()
        return self.ctx

    # ── Stage 1: ASR ───────────────────────────────────────────────
    def _stage_asr(self) -> None:
        """单引擎 ASR → engines_results[primary].

        v2.0 简化: 多引擎 → 单主引擎 (Zipformer). 引擎不可用时降级到列表中的下一个.
        """
        if self.ctx.skip_asr:
            if self.ctx.engines_results is None:
                self.ctx.engines_results = {}
            return

        engines = list(self.ctx.engines) if self.ctx.engines else [PRIMARY_ENGINE]

        getter = self.ctx.engine_getter
        if getter is None:
            from app.main import get_engine
            getter = get_engine

        self.ctx.engines_results = {}

        for engine_name in engines:
            try:
                engine = getter(engine_name)
            except Exception as e:
                logger.info(
                    "[Pipeline:ASR] engine '%s' not available: %s, skipping",
                    engine_name, e,
                )
                continue

            try:
                avail, reason = engine.is_available()
            except Exception as e:
                logger.info(
                    "[Pipeline:ASR] '%s' is_available() raised: %s, skipping",
                    engine_name, e,
                )
                continue
            if not avail:
                logger.info(
                    "[Pipeline:ASR] '%s' not available (%s), skipping",
                    engine_name, reason,
                )
                continue

            try:
                result = engine.transcribe(
                    self.ctx.audio_path,
                    language=self.ctx.language,
                    glossary=self.ctx.glossary or {},
                )
                segs = result.to_segments_list() if hasattr(result, "to_segments_list") else []
                self.ctx.engines_results[engine_name] = segs
                logger.info(
                    "[Pipeline:ASR] %s produced %d segments",
                    engine_name, len(segs),
                )
                # 单引擎成功即可, 不再尝试后续 (v2.0 主走 Zipformer)
                break
            except Exception as e:
                logger.warning(
                    "[Pipeline:ASR] %s transcribe failed: %s",
                    engine_name, e,
                )

        # raw_segments = 第一个成功引擎的结果 (向后兼容)
        if self.ctx.engines_results:
            primary_name = next(iter(self.ctx.engines_results))
            primary_segs = self.ctx.engines_results[primary_name]
            self.ctx.raw_segments = [self._seg_to_dict(s) for s in primary_segs]
            self.ctx.engine_meta = {
                "engines": list(self.ctx.engines_results.keys()),
                "primary": primary_name,
            }
        else:
            self.ctx.raw_segments = []
            self.ctx.engine_meta = {"engines": [], "primary": None}

        self._compute_duration()

    def _compute_duration(self) -> None:
        """从音频文件计算总时长."""
        if not self.ctx.audio_path:
            return
        try:
            import librosa
            self.ctx.duration = librosa.get_duration(path=self.ctx.audio_path)
        except Exception:
            if self.ctx.engines_results:
                last_end = 0.0
                for segs in self.ctx.engines_results.values():
                    for seg in segs:
                        if seg.get("end", 0) > last_end:
                            last_end = seg["end"]
                self.ctx.duration = last_end

    # ── Stage 2: Correct ───────────────────────────────────────────

    # 跨段碎片合并常量:
    # 段尾出现这些字符时视为完整句子, 不与下段合并
    _FRAGMENT_TERMINALS = frozenset("。！？…!?")
    # 合并后最大时长 (秒) / 最大字符数 (兜底), 超过则保持独立段.
    # v2.2: 12s/60 字过宽会把多句串成 40+ 字长字幕, 收紧到 8s/32 字
    _MERGE_MAX_DURATION = 8.0
    _MERGE_MAX_CHARS = 32
    # 去重后的短残余段 (重复句尾巴, <=8 字) 允许合并到 36 字,
    # 避免 "shi" / "timescale" 这类小残余独立成碎片段
    _DUP_TAIL_MAX_CHARS = 8
    _MERGE_MAX_CHARS_EXEMPT = 36
    # 极短段 (< 1.5s) 强制并入前段时, 合并后字符上限 (内容优先于长度)
    _SHORT_SEG_MERGE_CHARS = 40
    # 空/纯标点段 (无字幕内容, 如孤立 ".") 过滤
    _PUNCT_ONLY_CHARS = frozenset("。，、；：？！….,;:!?\"'()（） ")

    @staticmethod
    def _merge_fragments(segments: List[Dict]) -> List[Dict]:
        """跨段碎片合并 + 孤立段过滤.

        ASR 分段 (Zipformer CTC / VAD) 常把句子切碎在词中间
        (如 "通过改" + "变整个游戏"), 且可能输出纯标点段 (孤立 ".").
        本方法:
          1. 过滤空段与纯标点段
          2. 规则 A: 丢弃与相邻段高度重叠的短段 (ASR 整句重复识别)
          3. 段间子串去重: 前段尾部 >=6 字短语若重复出现在后段
             开头 1/3 内 (ASR 重叠识别), 从后段删除, 避免合并后重复
          4. 规则 B: 极短段 (< 1.5s) 读不完, 强制并入前段 (豁免长度/时长)
          5. 相邻段 prev 不以终止标点结尾 (。！？…!?) 且合并后
             时长 <= _MERGE_MAX_DURATION、字符 <= _MERGE_MAX_CHARS → 合并
          6. 拼接: prev 尾与 curr 头都是 ASCII 字母数字时加空格, 否则直拼
          7. 合并完成后清理段内连续重复短语 (ASR 整句重复识别)

        Returns:
            新段列表 (不修改入参).
        """
        if not segments:
            return []

        cleaned: List[Dict] = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if all(c in SubtitlePipeline._PUNCT_ONLY_CHARS for c in text):
                continue  # 孤立标点段无字幕内容
            cleaned.append(dict(seg, text=text))

        if len(cleaned) < 2:
            # 单段也要做段内连续重复清理 (ASR 整句重复识别)
            return SubtitlePipeline._clean_in_segment_dups(cleaned)

        merged: List[Dict] = [cleaned[0]]
        for seg in cleaned[1:]:
            prev = merged[-1]
            prev_text = prev["text"]
            curr_text = seg["text"]
            seg_duration = seg["end"] - seg["start"]

            # 规则 A: ASR 整句重复识别的短段丢弃
            # (如 "…输入 en gine" + "将修改时间尺度输入 engine timescale" 0.86s):
            # curr 时长 < 2s 且 curr 前缀 (<=15 字) 在 prev 中出现
            # (匹配 >= 8 字且占 curr 35% 以上) → 整段丢弃, 内容已在前段
            if (
                seg_duration < 2.0
                and len(prev_text) >= 8
                and len(curr_text) >= 8
            ):
                prefix_len = min(15, len(curr_text))
                longest = 0
                for k in range(prefix_len, 5, -1):
                    if curr_text[:k] in prev_text:
                        longest = k
                        break
                if longest >= 8 and longest / len(curr_text) >= 0.35:
                    logger.debug(
                        "[MergeFragments] drop duplicate short segment: %r", curr_text
                    )
                    continue

            # 段间子串去重: ASR 重叠识别时 prev 尾部短语会重复出现在
            # curr 开头附近 (如 "…输入 engine" + "将修改…输入 engine timescale")
            for n in range(min(15, len(prev_text)), 5, -1):
                tail = prev_text[-n:]
                pos = curr_text.find(tail)
                if 0 <= pos <= len(curr_text) // 3:
                    deduped = (curr_text[:pos] + curr_text[pos + n:]).strip()
                    if deduped:
                        curr_text = deduped
                    break

            if not curr_text:
                continue  # 后段完全重复 → 丢弃

            # 规则 B: 极短段 (< 1.5s) 读不完, 强制并入 prev (豁免时长上限);
            # 合并后超过 _SHORT_SEG_MERGE_CHARS 则不并入, 留给普通合并逻辑
            # (避免把整句长段与碎片硬拼成超长字幕)
            if seg_duration < 1.5:
                combined_chars = len(prev_text) + len(curr_text)
                if combined_chars <= SubtitlePipeline._SHORT_SEG_MERGE_CHARS:
                    need_space = (
                        prev_text[-1].isascii() and prev_text[-1].isalnum()
                        and curr_text[0].isascii() and curr_text[0].isalnum()
                    )
                    prev["text"] = prev_text + (" " if need_space else "") + curr_text
                    prev["end"] = seg["end"]
                    continue

            if prev_text[-1] in SubtitlePipeline._FRAGMENT_TERMINALS:
                merged.append(dict(seg, text=curr_text))
                continue

            combined_duration = seg["end"] - prev["start"]
            combined_chars = len(prev_text) + len(curr_text)
            # 去重后的短残余段 (重复句尾巴) 允许少量超额, 避免独立碎片段;
            # 普通合并仍受 _MERGE_MAX_CHARS 约束
            char_limit = (
                SubtitlePipeline._MERGE_MAX_CHARS_EXEMPT
                if len(curr_text) <= SubtitlePipeline._DUP_TAIL_MAX_CHARS
                else SubtitlePipeline._MERGE_MAX_CHARS
            )
            if (
                combined_duration > SubtitlePipeline._MERGE_MAX_DURATION
                or combined_chars > char_limit
            ):
                merged.append(dict(seg, text=curr_text))
                continue

            # ASCII 字母数字相邻 → 空格连接 (英文词级), 否则直拼 (中文)
            need_space = (
                prev_text[-1].isascii() and prev_text[-1].isalnum()
                and curr_text[0].isascii() and curr_text[0].isalnum()
            )
            prev["text"] = prev_text + (" " if need_space else "") + curr_text
            prev["end"] = seg["end"]

        # 段内连续重复短语清理: ASR 偶发把同一短语识别两遍
        # (如 "…我们的计时器，如果我们移除我们的计时器，如果我们移除…")
        return SubtitlePipeline._clean_in_segment_dups(merged)

    @staticmethod
    def _clean_in_segment_dups(segments: List[Dict]) -> List[Dict]:
        """清理段内连续重复短语.

        6-20 字连续重复在正常字幕中几乎不存在 (排比句除外), 可安全去重.
        单独抽离以便单段输入也执行.
        """
        import re
        dup_pattern = re.compile(r"(.{6,20}?)\1")
        result: List[Dict] = []
        for seg in segments:
            text = seg["text"]
            deduped = dup_pattern.sub(r"\1", text)
            result.append(dict(seg, text=deduped))
        return result

    def _stage_correct(self) -> None:
        """glossary 替换 + 确定性修正 + 标点.

        v2.0 简化:
          - 引擎层 (Zipformer) 已做 ITN + <unk> 清理, 这里 ITN 幂等
          - LLM 修正已砍
          - glossary (前端自定义词典) 在此统一应用, 对所有引擎的 ASR 结果生效
          - 标点: deterministic_correct 内部已含 (Phase 0 恢复 + Phase 4 插入),
            成功后本层不再跑独立 punctuation, 避免双标点;
            deterministic 关闭或异常降级时, 仍由本层补标点
        """
        # 先跨段碎片合并 (ASR 分段常把句子切碎在词中间), 再应用 glossary
        source = self._merge_fragments(list(self.ctx.raw_segments or []))

        if not source:
            self.ctx.corrected_segments = []
            return

        # glossary (前端自定义词典) 统一应用: 覆盖所有引擎的 ASR 结果
        # 原段 dict 是浅拷贝, 直接改 text 字段即可
        glossary = self.ctx.glossary or {}
        if glossary:
            for seg in source:
                text = seg.get("text", "")
                if not text:
                    continue
                for key, value in glossary.items():
                    if not key or key == value:
                        continue
                    text = text.replace(key, value)
                seg["text"] = text

        # deterministic 成功时内部已含标点, 跳过本层独立 punctuation;
        # 关闭或降级时由本层补标点
        deterministic_applied = False
        if self.ctx.enable_deterministic_correction:
            try:
                from app.core.asr_corrector import deterministic_correct
                corrected = deterministic_correct(source)
                deterministic_applied = True
            except Exception as e:
                logger.warning("[Pipeline:Correct] deterministic failed: %s", e)
                corrected = source
        else:
            corrected = source

        if self.ctx.enable_punctuation and not deterministic_applied:
            try:
                from app.core.punctuation import get_punctuation_engine
                engine = get_punctuation_engine()
                corrected = engine.punctuate_segments(corrected)
            except ImportError:
                logger.info("[Pipeline:Correct] punctuation unavailable, pass-through")
            except Exception as e:
                logger.warning("[Pipeline:Correct] punctuation failed: %s", e)

        corrected.sort(key=lambda s: s.get("start", 0))
        self.ctx.corrected_segments = corrected

    # ── Stage 3: Validate ──────────────────────────────────────────
    def _stage_validate(self) -> None:
        """字幕校验 + 日志."""
        segments = self.ctx.corrected_segments or []
        self.ctx.final_segments = segments

        if not self.ctx.enable_validate or not segments:
            return

        try:
            from app.core.subtitle_validator import log_validation_report
            log_validation_report(segments, source="pipeline")
        except Exception as e:
            logger.warning("[Pipeline:Validate] failed: %s", e)

    # ── Stage 4: Apply offset (主人手动微调字幕时间) ─────────────
    def _stage_apply_offset(self) -> None:
        """应用 config.subtitle_offset_sec 到所有段 start/end.

        用途: 当 ASR 输出时间戳相对人声有固定偏差时, 主人手动调.
        例: 字幕比人声晚 2s → config.yaml 设 subtitle_offset_sec: -2.0
        """
        offset = self.ctx.subtitle_offset_sec
        if offset == 0.0:
            return  # 没配置, 跳过
        if not self.ctx.final_segments:
            return
        for s in self.ctx.final_segments:
            s["start"] = max(0.0, round(s["start"] + offset, 3))
            s["end"] = max(0.0, round(s["end"] + offset, 3))
        logger.info(
            "[Pipeline:Offset] Applied subtitle_offset_sec=%.3f to %d segments",
            offset, len(self.ctx.final_segments),
        )

    # ── Stage 5: Format ────────────────────────────────────────────
    def _stage_format(self) -> None:
        """SRT/LRC/ASS 格式化输出."""
        from app.core.subtitle_formatter import (
            generate_ass,
            generate_lrc,
            generate_srt,
            segments_from_transcription,
        )
        srt_segments = segments_from_transcription(self.ctx.final_segments or [])
        fmt = (self.ctx.output_format or "srt").lower()

        if fmt == "srt":
            self.ctx.output_content = generate_srt(srt_segments)
        elif fmt == "lrc":
            self.ctx.output_content = generate_lrc(srt_segments)
        elif fmt == "ass":
            self.ctx.output_content = generate_ass(srt_segments, title=self.ctx.title)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    # ── 工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _seg_to_dict(seg) -> Dict[str, Any]:
        """Normalize Segment / dict to dict."""
        if isinstance(seg, dict):
            return seg
        if hasattr(seg, "to_dict"):
            return seg.to_dict()
        return {
            "start": getattr(seg, "start", 0),
            "end": getattr(seg, "end", 0),
            "text": getattr(seg, "text", ""),
        }
