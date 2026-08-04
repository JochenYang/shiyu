"""
统一后处理管线 — 参照剪映 Pro (JianyingPro) 的 Processor 链路。

剪映 Pro 的 ASR 后处理链大致是:
  ASR -> ITN (Inverse Text Normalization) -> Punctuation -> Validator -> Output

本模块把这条管线"抽出"成一个 dataclass 风格的统一入口,
方便 main.py / pipeline.py / LLM 修正层复用。

设计要点:
  * 三步独立可关: ITN / Punctuation / Validate 各自有 enable 开关
  * 不变性: 输入 segments 不被修改, 返回新的浅拷贝列表
  * Punctuation 后端是惰性加载的 (单例, 来自 get_punctuation_engine())
  * Validator 只做日志报告, 不修改 segments (符合剪映设计: 校验和修正是分离的)
  * 确定性词典修复仍由 ``ASRDeterministicCorrector`` 负责
    (PostProcessPipeline 不做"碎片修复 + 词典替换", 那是 Layer 1 的工作)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.itn import itn_normalize
from app.core.punctuation import (
    PunctuationBase,
    get_punctuation_engine,
    reset_punctuation_engine_cache,
)
from app.core.subtitle_validator import log_validation_report

logger = logging.getLogger(__name__)


@dataclass
class PostProcessPipeline:
    """统一后处理管线: ITN → Punctuation → Validator (参照剪映)。

    Attributes:
        enable_itn: 是否启用 ITN (中文数字归一化)。
        enable_punctuation: 是否启用标点恢复。
        enable_validate: 是否启用字幕校验 (只打日志, 不修改 segments)。

    Example:
        >>> pipeline = PostProcessPipeline()
        >>> result = pipeline.process([
        ...     {"start": 0.0, "end": 1.0, "text": "二零二六年"},
        ... ])
        >>> result[0]["text"]
        '2026年'
    """

    enable_itn: bool = True
    enable_punctuation: bool = True
    enable_validate: bool = True
    source_tag: str = "PostProcessPipeline"

    # 内部缓存: 标点恢复后端 (单例, 来自工厂)
    _punct_engine: Optional[PunctuationBase] = field(
        default=None, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if self.enable_punctuation:
            self._punct_engine = get_punctuation_engine()
        else:
            self._punct_engine = None

    def process(self, segments: List[Dict]) -> List[Dict]:
        """对 segments 列表做完整后处理。

        流水线:
            Step 1. ITN (中文数字归一化)
            Step 2. 标点恢复 (Rule / NN, 由工厂决定)
            Step 3. 字幕校验 (日志报告, 不修改数据)

        Args:
            segments: ``[{"start": float, "end": float, "text": str}, ...]``

        Returns:
            新的 segments 列表 (浅拷贝, 不修改入参)。
        """
        if not segments:
            return []

        result = [dict(seg) for seg in segments]

        # ── Step 1: ITN ──
        if self.enable_itn:
            result = self._apply_itn(result)

        # ── Step 2: 标点恢复 ──
        if self.enable_punctuation and self._punct_engine is not None:
            result = self._punct_engine.punctuate_segments(result)

        # ── Step 3: 字幕校验 (日志报告, 不修改 segments) ──
        if self.enable_validate:
            n_err, n_warn = log_validation_report(result, source=self.source_tag)
            logger.debug(
                "[PostProcess] %d segments processed, validation: %d errors, %d warnings",
                len(result), n_err, n_warn,
            )

        return result

    def _apply_itn(self, segments: List[Dict]) -> List[Dict]:
        """对每段 text 跑 ITN。空文本/非字符串安全跳过。"""
        for seg in segments:
            txt = seg.get("text", "")
            if isinstance(txt, str) and txt:
                normalized = itn_normalize(txt)
                if normalized != txt:
                    seg["text"] = normalized
        return segments

    def refresh_punctuation_engine(self) -> None:
        """清空标点后端缓存, 重新从工厂加载。

        主要用于:
          * 单元测试 (切换 Rule / NN 后端)
          * 模型下载完成后, 重新尝试加载 NN
        """
        reset_punctuation_engine_cache()
        if self.enable_punctuation:
            self._punct_engine = get_punctuation_engine()
        else:
            self._punct_engine = None
