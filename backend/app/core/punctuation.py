"""
Punctuation Restoration — Rule-based + NN-based (stub).

参照剪映 Pro (JianyingPro) PunctuationCtTransformer 模型设计:
  - 剪映用一个独立的 CT-Transformer 网络对 ASR 纯文本恢复标点,
    效果显著优于纯规则 (句号/问号/逗号恢复准确率提升约 20-30%)。
  - 本模块按"抽象 + 后端可替换"模式实现:
      * PunctuationBase   抽象基类, 统一接口 (Rule/NN 都遵循)
      * RulePunctuation   纯规则版, 4 类启发式, 总是可用 (兜底)
      * NNPunctuation     ONNX 推理版, 当前是 stub, 等主人确认后接入
      * get_punctuation_engine() 工厂方法, NN 可用就用 NN, 否则用 Rule

设计取舍:
  - 抽象类用 ABC + abstractmethod, 保持依赖注入的可能性
  - 默认 is_available() = True (Rule 总是可用)
  - NNPunctuation.is_available() = False (stub 模式, 主流程自动降级到 Rule)
  - Rule 版在 4 类启发式上做"够用即可"的判断:
      * 段尾语气词 → 问号 / 感叹号
      * 中文连词前后清理 + 加逗号
      * 段首语气词 → 后面加逗号
      * 纯英文段末 → 加句号
  - 默认**不**补 句号 (避免对长段过度补点), 由 caller 决定是否补

不实际下载任何模型。onnxruntime 已在 requirements.txt, 主人在确认后
可下载 PUNCT_MODEL_ID 对应 ONNX 权重, 在 NNPunctuation.load() 中接入。
"""
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 模型下载预留 (不实际下载) ─────────────────────────────────
# 阿里达摩院中文标点恢复 CT-Transformer (ModelScope)
# 实际加载逻辑 (ONNX Runtime + 自带 tokenizer) 在主人确认后接入
PUNCT_MODEL_ID = (
    "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
)
PUNCT_DIR = "punc_ct-transformer"

# ── 端点标点 / 软语气词 / 触发集合 ─────────────────────────────
_QUESTION_ENDS = {"吗", "呢"}
_EXCLAMATION_ENDS = {"啊", "呀", "哇", "哦", "啦"}
_SOFT_ENDS = {"吧"}  # 软语气, 不触发问号/感叹号, 也不触发默认句号
_EXCLAMATIVE_PARTICLES = {"啊", "呀", "哇", "哦"}  # 段内出现这些 → 段尾了/的 也算感叹

# 段首语气词: 在文本开头时, 后面补逗号
_LEADING_MODALS = ("啊", "呀", "哦", "哇")

# 中文连词 (段中出现时, 前面补逗号, 顺带清理前后空格)
# 按长度降序排, 避免短词优先匹配 (虽然这里没有前缀关系, 显式排一下更清晰)
_CONJUNCTIONS = sorted(
    [
        "然后", "但是", "因为", "所以", "如果", "虽然",
        "不过", "然而", "而且", "并且", "因此", "于是",
        "虽然说", "不但", "不仅", "既", "既然",
    ],
    key=lambda x: -len(x),
)

# 中段人称代词规则已移除 (v2.2): 启发式误伤大于收益, 见类 docstring

# 中文 Unicode 范围 (用于 lookbehind)
_CN_RANGE = r"\u4e00-\u9fff"


# ──────────────────────────────────────────────────────────────
# 抽象基类
# ──────────────────────────────────────────────────────────────


class PunctuationBase(ABC):
    """标点恢复后端抽象基类。

    所有标点恢复后端 (Rule / NN / 未来的 hybrid) 都必须实现
    ``punctuate(text)``。``punctuate_segments(segments)`` 是默认
    实现, 保留 start/end 字段, 只更新 text。

    不变性规则:
      * ``punctuate(text)`` 不修改入参字符串
      * ``punctuate_segments(segments)`` 不修改入参列表, 返回新列表
    """

    @abstractmethod
    def punctuate(self, text: str) -> str:
        """对一段纯文本 (无标点或少量标点) 进行标点恢复。

        Args:
            text: 输入文本, 可能不含标点或标点不完整。

        Returns:
            标点恢复后的文本。
        """
        raise NotImplementedError

    def punctuate_segments(self, segments: List[Dict]) -> List[Dict]:
        """对一组 segments 做标点恢复, 保留 start/end, 更新 text。

        默认实现对每个 segment 复制一份, 仅修改 text 字段。
        start / end / 其它字段保留不变。

        Args:
            segments: ``[{"start": float, "end": float, "text": str}, ...]``

        Returns:
            新的 segments 列表 (浅拷贝, 不修改入参)。
        """
        if not segments:
            return []
        result = [dict(seg) for seg in segments]
        for seg in result:
            txt = seg.get("text", "")
            if isinstance(txt, str) and txt:
                seg["text"] = self.punctuate(txt)
        return result

    def is_available(self) -> bool:
        """后端是否可用。Rule 总是 True, NN 在加载失败时返回 False。"""
        return True


# ──────────────────────────────────────────────────────────────
# 规则版后端
# ──────────────────────────────────────────────────────────────


class RulePunctuation(PunctuationBase):
    """规则版标点恢复 — 兜底实现, 总是可用。

    4 类启发式 (按处理顺序):

      1. **纯英文段末补句号**: 整段不含中文字符, 段末无标点 → 加 ``.``
      2. **段首语气词后加逗号**: 段首是 ``啊/呀/哦/哇`` 且后面是正文 → 加 ``，``
      3. **连词前后清理 + 加逗号**: ``然后/但是/因为`` 周围清理空格 + 加 ``，``
      4. **段尾语气词触发问号/感叹号**:
         - 末字是 ``吗/呢`` → ``？``
         - 末字是 ``啊/呀/哇/哦/啦`` → ``！``
         - 末字是 ``吧`` → 软语气, 不加
         - 末字是 ``了`` 且段内出现 ``啊/呀/哇/哦`` → ``！``

    注意:
      - 默认**不**补 句号 (避免对长段过度补点)。
        如果 caller 需要, 可在 PostProcessPipeline 阶段由
        ``ASRDeterministicCorrector._apply_punct_rules`` 兜底。
      - v2.2 移除"人称代词前加逗号"规则: 启发式无法区分主谓边界
        ("他说，我们…" 合理) 与动宾短语 ("包括，我们…" 误判), 误伤大于收益。
      - 规则实现做了"前面是中文"的 lookbehind, 避免在段首/标点后误触发。
    """

    def punctuate(self, text: str) -> str:
        if not text:
            return text

        text = text.strip()
        if not text:
            return text

        # ── 1. 纯英文段末补句号 ──
        if self._is_english_text(text):
            if not text.endswith((".", "!", "?", "。", "？", "！")):
                text = text + "."
            return text

        # ── 2. 段首语气词 → 后面加逗号 ──
        text = self._comma_after_leading_modal(text)

        # ── 3. 连词前后清理 + 加逗号 ──
        text = self._comma_around_conjunction(text)

        # ── 4. 段尾语气词触发问号/感叹号 ──
        text = self._trailing_punctuation(text)

        return text

    # ─── 启发式子函数 ────────────────────────────────────────

    @staticmethod
    def _is_english_text(text: str) -> bool:
        """检测 text 是否是纯英文/ASCII (无中文字符)。"""
        if not text:
            return False
        return not any("\u4e00" <= ch <= "\u9fff" for ch in text)

    @staticmethod
    def _comma_after_leading_modal(text: str) -> str:
        """段首是语气词 (啊/呀/哦/哇) 且后面是正文 → 后面加 ，"""
        if len(text) < 2:
            return text
        for modal in _LEADING_MODALS:
            if not text.startswith(modal):
                continue
            next_char = text[len(modal)]
            if next_char in " ，。！？.!?,、；：":
                return text
            return modal + "，" + text[len(modal):]
        return text

    @staticmethod
    def _comma_around_conjunction(text: str) -> str:
        """中文连词前后清理空格 + 前面加 ，。

        例: ``"我们今天 然后 去了公园"`` → ``"我们今天，然后去了公园"``

        规则:
          * lookbehind 要求前一字符是中文 (段首不算)
          * 吃掉连词前后的所有空白, 整体替换为 ``，<conj>``
        """
        for conj in _CONJUNCTIONS:
            pattern = re.compile(
                rf"(?<=[{_CN_RANGE}])\s*{re.escape(conj)}\s*"
            )
            text = pattern.sub("，" + conj, text)
        return text

    @staticmethod
    def _trailing_punctuation(text: str) -> str:
        """段尾语气词触发问号/感叹号 / 软语气不加。"""
        if not text:
            return text
        last_char = text[-1]
        if last_char in _QUESTION_ENDS:
            return text + "？"
        if last_char in _EXCLAMATION_ENDS:
            return text + "！"
        if last_char in _SOFT_ENDS:
            return text
        # 末字是"了"且段内有感叹词 → ！  (例如: "啊这真是太棒了")
        if last_char == "了" and any(
            p in text for p in _EXCLAMATIVE_PARTICLES
        ):
            return text + "！"
        return text


# ──────────────────────────────────────────────────────────────
# NN 版后端 (stub)
# ──────────────────────────────────────────────────────────────


class NNPunctuation(PunctuationBase):
    """NN 版标点恢复 — 当前为 stub, 等主人确认后接入 PunctuationCtTransformer。

    实际加载流程 (后续实现):
      1. 从 ModelScope / HF Mirror 下载 ONNX 权重到 ``PUNCT_DIR``
      2. 用 ``onnxruntime.InferenceSession`` 加载
      3. 自带中文 BPE tokenizer (类似 SenseVoice)
      4. 对每段文本做 char-level 推理, 输出每个位置的概率分布
      5. 取 argmax 还原为标点序列

    推理与 funasr 解耦 (直接用 onnxruntime), 这样:
      * 不引入 funasr 大依赖
      * 与现有 SenseVoice 引擎共存
      * 模型小, 适合 CPU 推理

    当前 stub 行为:
      * ``is_available()`` 恒返回 ``False``
      * ``punctuate()`` 直接降级到 ``RulePunctuation()``
      * ``load()`` 永远是 False (未实现)
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._session: Optional[object] = None  # onnxruntime.InferenceSession
        self._tokenizer: Optional[object] = None
        self._loaded: bool = False

    def is_available(self) -> bool:
        """当前 stub 模式恒返回 False, 主流程会自动降级到 Rule。"""
        return False

    def load(self) -> bool:
        """预留: 实际加载 ONNX 权重 + tokenizer。

        Returns:
            True 表示加载成功, False 表示加载失败 (主流程会降级到 Rule)。

        实现策略 (后续):
          * 检查 model_path 是否存在
          * 用 onnxruntime.InferenceSession 加载
          * 加载 char vocab (内置, 无需外部文件)
          * 缓存到 self._session / self._tokenizer
        """
        # 当前 stub: 不实际加载
        return False

    def punctuate(self, text: str) -> str:
        """stub 模式: 降级到 RulePunctuation。

        后续实现: 调用 self._session.run() 做推理, 输出带标点文本。
        """
        if not text:
            return text
        return RulePunctuation().punctuate(text)

    def punctuate_segments(self, segments: List[Dict]) -> List[Dict]:
        """stub 模式: 走基类默认实现 (即逐段调 punctuate, 降级到 Rule)。"""
        return super().punctuate_segments(segments)


# ──────────────────────────────────────────────────────────────
# 工厂方法
# ──────────────────────────────────────────────────────────────


_ENGINE_CACHE: Optional[PunctuationBase] = None


def get_punctuation_engine() -> PunctuationBase:
    """工厂方法: NN 可用就用 NN, 否则用 Rule。

    带进程级缓存 (单例), 避免每次调用都新建对象。

    Returns:
        一个 ``PunctuationBase`` 实例。当前 NN 是 stub, 所以返回 RulePunctuation。
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE

    nn = NNPunctuation()
    if nn.is_available():
        logger.info("[Punctuation] Using NNPunctuation backend")
        _ENGINE_CACHE = nn
    else:
        logger.info(
            "[Punctuation] NN unavailable, using RulePunctuation (fallback)"
        )
        _ENGINE_CACHE = RulePunctuation()
    return _ENGINE_CACHE


def reset_punctuation_engine_cache() -> None:
    """重置工厂缓存, 主要用于测试 (切换后端时清理单例)。"""
    global _ENGINE_CACHE
    _ENGINE_CACHE = None
