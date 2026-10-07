"""
ASR Deterministic Corrector — 自研高效算法，零 API 成本。

双层架构的第一层（Layer 1），在 LLM 之前运行：
0. 标点恢复 (Phase B, 参照剪映 PunctuationCtTransformer)
1. 段碎片修复（跨段上下文推断）
2. ASR 领域术语词典替换（智能验算：词边界+上下文置信度）
3. 重叠内容去重
4. 智能标点插入 (弱化: Rule 3 跳过软语气词, 由 Phase 0 接管)

智能验算引擎（自研）：
- 每个替换自动做「前验+后验」双向验证
- 词边界感知：不会把"bar"匹配到"barometer"
- 上下文置信度：同源多目标时，用领域关联词打分排序
- 替换后校验：验证替换结果不产生畸形拼接

Phase B 标点恢复:
- Rule 版 (兜底): 4 类启发式 (段尾语气词/连词/中段代词/段首语气词/英文段末)
- NN 版 (stub): 预留 PunctuationCtTransformer ONNX 推理位, is_available=False
- 工厂 get_punctuation_engine() 决定当前用哪个后端

输出：在保留原始分段的前提下，产出可读的修正结果。
LLM 层不可用时，Layer 1 确保用户永远不看到"无变化"的 ASR 原文。
"""
import re
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# Phase B: 标点恢复后端 (Rule / NN, 工厂选择)
from app.core.punctuation import get_punctuation_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 英语常用词表（用于碎片检测）
# ~700 常用词 + 技术领域词
# 来源：OGDEN's Basic English + 编程/游戏开发常用词
# ---------------------------------------------------------------------------
_COMMON_EN_WORDS: Set[str] = {
    # 冠词/代词/基本动词
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "my", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "need", "dare", "ought", "used",
    "not", "no", "nor", "none", "nothing", "nobody", "nowhere",
    "all", "each", "every", "both", "few", "many", "much", "some", "any",
    "other", "another", "such", "same", "own",
    # 连词/介词
    "and", "or", "but", "if", "because", "although", "though", "while",
    "when", "where", "why", "how", "which", "who", "whom", "whose",
    "what", "whatever", "whenever", "wherever", "whether",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "onto", "upon", "within", "without", "through", "throughout",
    "about", "above", "across", "after", "against", "along", "among",
    "around", "before", "behind", "below", "beneath", "beside", "between",
    "beyond", "down", "during", "except", "inside", "near", "off",
    "outside", "over", "past", "since", "through", "toward", "under",
    "until", "up", "via", "with", "within",
    # 常见形容词/副词
    "good", "better", "best", "bad", "worse", "worst",
    "big", "bigger", "biggest", "small", "smaller", "smallest",
    "high", "higher", "highest", "low", "lower", "lowest",
    "new", "old", "first", "last", "next", "previous",
    "long", "short", "wide", "narrow", "deep", "shallow",
    "fast", "slow", "hard", "soft", "light", "heavy",
    "easy", "hard", "simple", "complex", "clear", "vague",
    "right", "wrong", "true", "false", "real", "fake",
    "full", "empty", "open", "closed", "early", "late",
    "now", "then", "here", "there", "always", "never",
    "often", "sometimes", "usually", "rarely", "seldom",
    "very", "really", "quite", "almost", "nearly", "just",
    "only", "also", "too", "so", "thus", "hence",
    "however", "therefore", "nevertheless", "meanwhile",
    "already", "yet", "still", "ever", "even",
    # 常见名词
    "time", "times", "way", "ways", "thing", "things",
    "people", "person", "place", "world", "life", "hand",
    "part", "parts", "case", "point", "number", "group",
    "problem", "fact", "example", "system", "data", "information",
    "work", "study", "research", "development", "process",
    "state", "level", "type", "form", "area", "field",
    "result", "reason", "change", "difference", "effect",
    "name", "line", "end", "start", "head", "body",
    "day", "week", "month", "year", "hour", "minute", "second",
    # 游戏开发/编程常见技术词
    "engine", "game", "unity", "unreal", "godot", "scene",
    "object", "component", "script", "class", "function",
    "method", "variable", "property", "value", "reference",
    "input", "output", "update", "fixedupdate", "start",
    "awake", "enable", "disable", "destroy", "create",
    "instantiate", "spawn", "load", "save", "reset",
    "transform", "position", "rotation", "scale", "vector",
    "velocity", "speed", "force", "mass", "gravity",
    "collision", "collider", "trigger", "rigidbody", "physics",
    "material", "texture", "mesh", "model", "animation",
    "controller", "character", "player", "camera", "light",
    "audio", "sound", "music", "effect", "particle",
    "timescale", "timestep", "delta", "framerate", "fps",
    "action", "event", "signal", "delegate", "callback",
    "coroutine", "async", "await", "task", "thread",
    "array", "list", "dict", "queue", "stack", "heap",
    "string", "number", "integer", "float", "boolean",
    "interface", "abstract", "virtual", "override", "static",
    "public", "private", "protected", "internal", "extern",
    "namespace", "package", "module", "import", "export",
    "return", "break", "continue", "throw", "try", "catch",
    "finally", "if", "else", "switch", "case", "default",
    "for", "while", "foreach", "in", "out", "ref",
    "null", "nil", "none", "true", "false", "void",
    "int", "float", "double", "char", "byte", "bool",
    "node", "path", "file", "dir", "folder", "root",
    "parent", "child", "sibling", "buddy", "group",
    "local", "global", "world", "screen", "view",
    "x", "y", "z", "w", "xy", "xyz",
}

# 确定可以独立存在的短英语词（"en" 不是英语独立词，所以如果看到 "en" + "gine" 会触发修复）
# 这些词即使很短也确定为独立词，不会触发碎片修复
_SHORT_STANDALONE_EN: Set[str] = {
    "a", "i", "in", "on", "at", "to", "of", "by", "as", "is", "it",
    "or", "an", "be", "we", "he", "she", "me", "my", "no", "so",
    "go", "do", "up", "us", "if", "am", "ok", "okay",
    "x", "y", "z", "xy", "xyz",
    "id", "ui", "2d", "3d",
}

# ---------------------------------------------------------------------------
# 智能验算引擎 — 上下文感知词典替换
# ---------------------------------------------------------------------------


@dataclass
class DictEntry:
    """单条词典条目（含验算元数据）。

    Attributes:
        source: 源字符串（ASR 错误形式）。
        target: 目标字符串（修正后形式）。
        context_hints: 关联词列表。扫描上下文中是否出现这些词，
                       用于计算领域置信度。
        word_boundary: 替换时是否需要检查词边界。
        min_hits: 在上下文中至少命中多少个 context_hints 才替换。
                  0 表示不强制关联词命中（但仍做词边界检查）。
        exclusive_mode: "any" = 任意 context_hints 命中即可；
                        "all" = 全部 context_hints 需都出现。
    """
    source: str
    target: str
    context_hints: List[str] = field(default_factory=list)
    word_boundary: bool = True
    min_hits: int = 0
    exclusive_mode: str = "any"  # "any" | "all"


class SmartDictionary:
    """智能词典引擎 —— 自研的自动验算替换器。

    核心机制：
    1. 长词优先匹配 + 词边界感知
    2. 多候选竞争：同一源词多个条目（如 "巴蒂" → "body" / "巴蒂" → 保留）
    3. 上下文置信度打分
    4. 替换后验证

    用法:
        sd = SmartDictionary(entries)
        fixed = sd.apply("源文本")
    """

    def __init__(self, entries: Optional[List[DictEntry]] = None):
        self._entries: List[DictEntry] = []
        if entries:
            for e in entries:
                self.add(e)

    def add(self, entry: DictEntry) -> "SmartDictionary":
        """添加一条词典项（自动按源词长度降序插入）。"""
        self._entries.append(entry)
        self._entries.sort(key=lambda x: (-len(x.source), x.min_hits))
        return self

    def add_simple(self, source: str, target: str) -> "SmartDictionary":
        """添加一条简单替换项（默认带词边界检查，不带上下文验算）。

        word_boundary=True 确保 "bar" 不会匹配到 "barometer"。
        如果需要无边界替换（如处理短拼接词），用 add() 手动设置 word_boundary=False。
        """
        self.add(DictEntry(source=source, target=target,
                           word_boundary=True, min_hits=0))
        return self

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def apply(self, text: str) -> str:
        """对文本执行智能词典替换（含自动验算）。

        Args:
            text: 待修正文本。

        Returns:
            修正后文本。未找到匹配则返回原文本。
        """
        if not text or not self._entries:
            return text

        buf = []
        pos = 0
        while pos < len(text):
            candidates = self._find_candidates_at(text, pos)
            if candidates:
                # 多候选竞争：选验算分最高的
                scored = [(self._score(text, pos, c), c) for c in candidates]
                scored.sort(key=lambda x: -x[0])
                best_score, best = scored[0]
                if best_score > 0:
                    buf.append(best.target)
                    pos += len(best.source)
                    continue
            buf.append(text[pos])
            pos += 1

        return "".join(buf)

    # ------------------------------------------------------------------
    # 候选匹配
    # ------------------------------------------------------------------

    def _find_candidates_at(self, text: str, pos: int) -> List[DictEntry]:
        """在 *pos* 位置查找所有匹配的候选条目。

        只返回最长匹配的条目（按 source 长度）：
        - 如果 "collision ship" (14字) 和 "collision" (10字) 都匹配同一位置，
          只返回 "collision ship"
        - 如果多个条目有相同的 source 长度，全部返回（用于多候选竞争）
        """
        candidates: List[DictEntry] = []
        remaining = text[pos:]
        longest = 0
        for entry in self._entries:
            src = entry.source
            if not remaining.startswith(src):
                continue
            if len(src) > longest:
                candidates = [entry]
                longest = len(src)
            elif len(src) == longest:
                candidates.append(entry)
        return candidates

    # ------------------------------------------------------------------
    # 自动验算核心
    # ------------------------------------------------------------------

    def _score(self, text: str, match_pos: int, entry: DictEntry) -> float:
        """对一条候选条目在特定位置进行验算评分。

        Returns 0.0 ~ 1.0 的评分。<= 0 表示不应用此替换。

        三因子加权：
          - 词边界分 (0~0.5)
          - 上下文分 (0~0.5)
          - 有效性分 (0~0.2, 加分项)
        """
        src = entry.source

        # ── 因子 1: 词边界检查 ──
        boundary_score = self._check_boundary(text, match_pos, src)
        if boundary_score <= 0 and entry.word_boundary:
            return 0.0  # 词边界不过关 → 硬拒绝

        # ── 因子 2: 上下文相关度 ──
        context_score = self._check_context(text, match_pos, src, entry)
        if context_score < 0:
            return 0.0  # 上下文不达标 → 硬拒绝

        # ── 因子 3: 替换后有效性 ──
        validity_bonus = self._check_validity(entry.source, entry.target)

        # ── 最终评分 ──
        score = boundary_score * 0.5 + context_score * 0.4 + validity_bonus * 0.1
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # 验算子因子
    # ------------------------------------------------------------------

    @staticmethod
    def _check_boundary(text: str, pos: int, source: str) -> float:
        """词边界检查。

        检查替换位置的前后字符，防止部分匹配。
        返回 0.0（完全越界）~ 0.5（干净边界）。

        规则：
        - 前一个字符是字母/数字 → 越界（如 "bar" 在 "barometer" 中）
        - 后一个字符是字母/数字 → 越界
        - 前后都是空格/中文/标点 → 干净边界 0.5
        - 一方是边界、另一方是特殊字符（如连字符）→ 宽松边界 0.3
        """
        before_char = text[pos - 1] if pos > 0 else " "
        after_pos = pos + len(source)
        after_char = text[after_pos] if after_pos < len(text) else " "

        # 中文 isalpha() 为 True，所以必须加 isascii() 限制
        # 只有 ASCII 字母/数字才算"连续词"（中文边界不算越界）
        before_bad = (before_char.isascii() and before_char.isalpha()) or before_char.isdigit()
        after_bad = (after_char.isascii() and after_char.isalpha()) or after_char.isdigit()

        if before_bad or after_bad:
            return 0.0  # 硬越界

        # 双方都是空格/中文/标点 → 干净
        both_boundary = (
            before_char in (" ", "　", "\t", "\n", "\r", "") or not before_char.isascii()
        ) and (
            after_char in (" ", "　", "\t", "\n", "\r", "") or not after_char.isascii()
        )
        return 0.5 if both_boundary else 0.3

    def _check_context(
        self, text: str, pos: int, source: str, entry: DictEntry,
    ) -> float:
        """上下文相关度检查。

        以匹配位置为中心 ±60 字符窗口，扫描关联词命中率。
        返回 -1.0（不足最小命中数）或 0.0 ~ 0.5。
        """
        if not entry.context_hints:
            return 0.5  # 无关联词要求 → 默认高分通过

        window_start = max(0, pos - 60)
        window_end = min(len(text), pos + len(source) + 60)
        window = text[window_start:window_end].lower()

        hits = sum(1 for hint in entry.context_hints if hint.lower() in window)
        if hits < entry.min_hits:
            return -1.0  # 上下文不达标

        # 命中率归一化
        rate = hits / max(len(entry.context_hints), 1)
        return min(0.5, 0.1 + rate * 0.4)  # 0.1 ~ 0.5

    @staticmethod
    def _check_validity(source: str, target: str) -> float:
        """替换后有效性检查。

        检查替换结果不会产生明显问题。
        返回 0.0 ~ 0.2 的加分。

        规则：
        - 目标为空 → 无效
        - 源 == 目标 → 无效（无意义替换）
        - 英->英：目标在常用词表中 → 加分
        - 目标以标点结尾、目标只含数字等特殊模式 → 降分
        """
        if not source or not target:
            return 0.0
        if source == target:
            return 0.0

        bonus = 0.1  # 基础分

        # 英->英替换，目标在常用词表中 → high confidence
        if target.isascii() and target.isalpha():
            if target.lower() in _COMMON_EN_WORDS:
                bonus = 0.2

        return min(0.2, bonus)

    # ------------------------------------------------------------------
    # 批量接口
    # ------------------------------------------------------------------

    def apply_to_texts(self, texts: List[str]) -> List[str]:
        """批量应用验算替换。"""
        return [self.apply(t) for t in texts]


# ---------------------------------------------------------------------------
# 配置常量
# Phase 1: PUNCT_MIN_LEN 从 _apply_punct_rules line 793 硬编码字面量 10 提为模块常量
# v2.2: Rule 3 (补句号) 移除; v2.2.1 以白名单版恢复, 该常量已不再使用
# ---------------------------------------------------------------------------
PUNCT_MIN_LEN: int = 10  # 历史: 补句号最小字符阈值 (已不再使用)

# v2.2.2: 段尾悬空字黑名单 — 命中且段长 > 4 时不补句号.
# 通用原则: 中文句子可以以绝大多数字结尾 (名词/形容词/动词补语/
# 语气词), 只有"悬空字"结尾时句子几乎必然未完 (单字及物动词后面
# 还要接宾语, 介词/方位/时间词后面要接名词, 连词/助词用于衔接),
# 因此用黑名单排除碎片段, 其余中文句尾一律补句号; 英文结尾无法
# 判断碎片 (gine/ship/collision), 一律不补.
_FRAGMENT_TAIL_CHARS: frozenset = frozenset(
    # 高频单字及物动词 (结尾时通常还要接宾语, 句子未完)
    "改 添 指 所 游 获 执 搜 调 取 转 变 移 删 打 放 拿 弄 想 需 要 "
    "说 讲 写 做 用 找 看 听 读 学 教 让 使 问 求 算 计 测 试 验 修 建 造 "
    # 被动/介词/方位/时间/量词 (悬空, 后面通常接名词)
    "被 已 把 从 向 给 为 上 下 中 后 前 里 外 内 时 分 秒 年 月 日 天 次 个 条 张 帧 段 行 列 "
    "组 队 台 部 间 侧 处 面 名 者 员 "
    # 连词/副词/其他明显悬空
    "死 度 载 除 是 点 节 这 那 与 及 或 但 而 且 并 才 就 都 也 又 还 "
    "再 只 仅 曾 正 将 会 能 可 以 需"
)


# ---------------------------------------------------------------------------
# 默认 ASR 纠错词典（v2：带智能验算元数据）
# 以 source 长度降序排列（长词优先匹配）
# ---------------------------------------------------------------------------

# 创建智能词典实例
_DEFAULT_SMART_DICT = SmartDictionary([
    # -- 领域术语映射（最常见的 ASR 误识别）--
    # "collision ship" → "collision shape"，关联游戏开发词
    DictEntry("collision ship", "collision shape",
              context_hints=["碰撞", "物体", "节点", "game", "unity",
                             "godot", "physics", "rigidbody"],
              word_boundary=True, min_hits=1),

    # -- ASR 拼写错误 --
    # "timescal" → "timescale"，启用词边界: 保护 "timescales"/"timescale"
    # 等正确拼写不被前缀替换破坏 (word_boundary=False 时 "timescales" → "timescalees")
    DictEntry("timescal", "timescale",
              word_boundary=True, min_hits=0),
    DictEntry("timest ep", "time step",
              word_boundary=True, min_hits=0),
    DictEntry("timestep", "time step",
              word_boundary=True, min_hits=0),

    # -- 中文同音字错误 --
    # "巴蒂" → "body"，默认带游戏开发上下文验算
    # 如果没有关联词命中，自然就不会替换（如"巴蒂"是人名时）
    DictEntry("巴蒂", "body",
              context_hints=["回调", "碰撞", "节点", "物体", "函数",
                             "脚本", "修改", "碰撞", "rigidbody"],
              word_boundary=True, min_hits=1),
    DictEntry("八的", "body的",
              context_hints=["回调", "碰撞", "节点", "物体", "函数",
                             "脚本", "修改"],
              word_boundary=True, min_hits=1),
    DictEntry("把的", "body的",
              context_hints=["回调", "碰撞", "节点", "物体", "函数",
                             "脚本", "修改"],
              word_boundary=True, min_hits=1),
    DictEntry("拔的", "body的",
              context_hints=["回调", "碰撞", "节点", "物体", "函数",
                             "脚本", "修改"],
              word_boundary=True, min_hits=1),
])


def load_asr_dictionary(path: Optional[str] = None) -> SmartDictionary:
    """从 JSON 文件加载用户自定义词典，与默认智能词典合并。

    JSON 格式（v2 — 推荐）:
        [
            {
                "source": "巴蒂", "target": "body",
                "context_hints": ["回调", "碰撞"],
                "word_boundary": true,
                "min_hits": 1
            },
            {
                "source": "timescal", "target": "timescale",
                "word_boundary": false,
                "min_hits": 0
            }
        ]

    也支持 v1 简单格式（向后兼容）:
        [
            {"source": "巴蒂", "target": "body", "category": "术语"},
            {"source": "timescal", "target": "timescale"}
        ]

    Args:
        path: 自定义词典 JSON 路径。None 则只返回默认词典。

    Returns:
        SmartDictionary 实例（包含默认+自定义条目）。
    """
    result = SmartDictionary()
    # 复制默认词典
    for entry in _DEFAULT_SMART_DICT._entries:
        result.add(DictEntry(
            source=entry.source, target=entry.target,
            context_hints=list(entry.context_hints),
            word_boundary=entry.word_boundary,
            min_hits=entry.min_hits,
        ))

    if path:
        try:
            p = Path(path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for entry in entries:
                    src = entry.get("source", "").strip()
                    tgt = entry.get("target", "").strip()
                    if not src or not tgt or src == tgt:
                        continue
                    # 兼容 v1 简单格式（无验算元数据）
                    de = DictEntry(
                        source=src, target=tgt,
                        context_hints=entry.get("context_hints", []),
                        word_boundary=entry.get("word_boundary",
                            len(src) <= 4 and src.isalpha()),
                        min_hits=entry.get("min_hits", 0),
                    )
                    result.add(de)
        except Exception as e:
            logger.warning("Failed to load ASR dictionary from %s: %s", path, e)

    return result


class ASRDeterministicCorrector:
    """确定性 ASR 文本修正引擎。

    双层架构的 Layer 1，无需 LLM API 调用，零成本运行。
    输出总是在"可读"水平，LLM 不可用时不会回退到原文。

    用法:
        corrector = ASRDeterministicCorrector()
        corrected = corrector.correct(segments)
    """

    def __init__(
        self,
        dictionary: Optional[SmartDictionary] = None,
        enable_fragment_fix: bool = True,
        enable_punctuation: bool = True,
        enable_dedup: bool = True,
    ):
        """
        Args:
            dictionary: 智能词典。None 用默认词典。
            enable_fragment_fix: 启用跨段碎片修复。
            enable_punctuation: 启用标点插入。
            enable_dedup: 启用重叠去重。
        """
        self._dict = dictionary if dictionary is not None else load_asr_dictionary()
        self._enable_fragment = enable_fragment_fix
        self._enable_punct = enable_punctuation
        self._enable_dedup = enable_dedup

        logger.info(
            "ASRDeterministicCorrector initialized (dict_entries=%d, fragment=%s, punct=%s, dedup=%s)",
            len(self._dict._entries) if self._dict else 0,
            enable_fragment_fix, enable_punctuation, enable_dedup,
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def correct(self, segments: List[Dict]) -> List[Dict]:
        """执行确定性修正。

        不可变性规则：不修改输入的 segments 对象，
        返回新的 segment 字典列表。

        Args:
            segments: [{start, end, text}, ...]

        Returns:
            修正后的 segments 列表（新对象，非输入引用）。
        """
        if not segments:
            return []

        # shallow copy: 不修改输入（不可变性规则）
        result = [dict(seg) for seg in segments]

        # Phase 0: 标点恢复 (参照剪映 PunctuationCtTransformer)
        # 标点恢复放在最前面,让后续步骤 (词典、碎片、dedup) 看到更完整的标点
        try:
            punc_engine = get_punctuation_engine()
            result = punc_engine.punctuate_segments(result)
        except Exception as e:
            logger.debug("[Punctuation] Engine unavailable, skipping: %s", e)

        # Phase 1: 跨段碎片修复
        if self._enable_fragment:
            self._fix_fragments(result)

        # Phase 2: 词典替换
        self._apply_dictionary(result)

        # Phase 3: 重叠去重（在标点之前，避免标点干扰重叠检测）
        if self._enable_dedup:
            self._deduplicate_overlap(result)

        # Phase 4: 标点插入（最后一步，不在原始内容上加干扰字符）
        if self._enable_punct:
            self._insert_punctuation(result)

        return result

    # ------------------------------------------------------------------
    # Phase 1 - 碎片修复
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_word(text: str) -> Optional[str]:
        """提取文本的最后一个英语单词。"""
        words = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text)
        return words[-1] if words else None

    @staticmethod
    def _extract_first_word(text: str) -> Optional[str]:
        """提取文本的第一个英语单词。"""
        words = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text)
        return words[0] if words else None

    @staticmethod
    def _remove_last_word(text: str, word: str) -> str:
        """从文本末尾移除指定的英语单词。"""
        stripped = text.rstrip()
        # Check if text ends with the word (word boundary)
        # Use negative lookahead to ensure it's not part of a longer word
        # We need to check that the word is at the end of the string
        pattern = re.compile(rf'\b{re.escape(word)}\b\s*$')
        return pattern.sub('', stripped).rstrip()

    @staticmethod
    def _remove_first_word(text: str, word: str) -> str:
        """从文本开头移除指定的英语单词。"""
        stripped = text.lstrip()
        pattern = re.compile(rf'^\s*\b{re.escape(word)}\b')
        return pattern.sub('', stripped).lstrip()

    def _fix_fragments(self, segments: List[Dict]) -> None:
        """修复跨段断词。

        算法原理：
            对于相邻段 (prev, curr)，检测 prev 的最后一个英文词
            和 curr 的第一个英文词是否能拼接成一个有效英语词。
            若是，完整词写回 prev 段末尾，curr 段只移除碎片词
            （不合并段，避免同一词在相邻两段各显示一遍）。
            curr 段仅剩碎片词时保持空字符串，由下游
            segments_from_transcription 过滤空段。

        触发条件（严格，避免误判）：
            1. 拼接后的词在常用英语词表中
            2. 第一段的碎片不是独立英语词（避免误伤"time"+"scale"）
            3. 第二段的碎片也不是独立英语词（同上）

        Fixes in-place on ``segments``.
        """
        if len(segments) < 2:
            return

        for i in range(len(segments) - 1):
            prev = segments[i]
            curr = segments[i + 1]

            prev_text = prev.get("text", "").strip()
            curr_text = curr.get("text", "").strip()

            if not prev_text or not curr_text:
                continue

            last_word = self._extract_last_word(prev_text)
            first_word = self._extract_first_word(curr_text)

            if not last_word or not first_word:
                continue

            last_lower = last_word.lower()
            first_lower = first_word.lower()

            # 如果两者都是独立英语词 → 不是碎片（如 "time" + "step"）
            if (last_lower in _SHORT_STANDALONE_EN or last_lower in _COMMON_EN_WORDS) and \
               (first_lower in _SHORT_STANDALONE_EN or first_lower in _COMMON_EN_WORDS):
                continue

            combined = last_lower + first_lower

            if combined in _COMMON_EN_WORDS:
                # 碎片确认：完整词写回 prev，curr 只移除碎片词
                rest_prev = self._remove_last_word(prev_text, last_word)
                rest_curr = self._remove_first_word(curr_text, first_word)

                prev["text"] = (rest_prev + " " + combined).strip()
                # curr 段只移除碎片词, 不再补完整词
                # (否则完整词会在相邻两段各出现一次, 字幕显示重复)
                curr["text"] = rest_curr

                logger.debug(
                    "Fragment fixed: seg %d '%s...' + seg %d '...%s' → '%s'",
                    i, last_word, i + 1, first_word, combined,
                )

    # ------------------------------------------------------------------
    # Phase 2 - 词典替换
    # ------------------------------------------------------------------

    def _apply_dictionary(self, segments: List[Dict]) -> None:
        """应用智能词典替换（含自动验算）。

        由 SmartDictionary 的 apply() 方法完成：
        - 词边界感知
        - 上下文置信度打分
        - 多候选竞争
        - 替换后有效性校验

        Fixes in-place on ``segments``.
        """
        if not self._dict:
            return

        for seg in segments:
            text = seg.get("text", "")
            if not text:
                continue

            original = text
            corrected = self._dict.apply(text)
            if corrected != original:
                seg["text"] = corrected
                logger.debug(
                    "Dict fix: '%.40s' → '%.40s'", original, corrected,
                )

    # ------------------------------------------------------------------
    # Phase 3 - 标点插入
    # ------------------------------------------------------------------

    # 中文句首连词/连接词 → 前面补逗号或句号
    _CONJUNCTION_STARTS = [
        "所以", "但是", "然而", "那么", "其次", "接着", "最后",
        "因此", "于是", "不过", "可是", "而且", "并且", "或者",
        "然后",
        "但", "可", "却",
    ]

    # 句子结尾语气词 → 触发问号或感叹号
    # 注意：优先级决定了冲突时的处理
    # Phase B: 移除"吧"/"呀"/"么"对齐新 RulePunctuation (把它们分别归为软语气/感叹/不触发)
    _QUESTION_ENDS = {"吗", "呢"}  # 明确问句 (Phase B 收紧: 吧是软语气, 呀是感叹, 么不触发)
    _EXCLAMATION_ENDS = {"啊", "呀", "哇", "哦", "啦", "哈"}  # 感叹（含疑问式感叹）

    # Phase B: 软语气词 (Rule 3 不补句号)
    # 段尾是这些字符时, 默认句号规则跳过 (因为新 RulePunctuation 已经处理过)
    # 例: "走走吧" 末字"吧"是软语气, 不强制加句号
    _SOFT_PARTICLES = {"吧", "啊", "呢", "吗", "哦", "啦", "呀", "哇"}

    # 英文句首大写标记
    _EN_SENTENCE_STARTS = {"the", "this", "that", "it", "i", "we", "they",
                           "he", "she", "there", "here", "these", "those",
                           "my", "our", "your", "his", "her", "its",
                           "a", "an", "in", "on", "at", "to", "for",
                           "so", "but", "and", "or", "if", "when",
                           "while", "because", "although", "since",
                           "however", "therefore", "meanwhile"}

    def _insert_punctuation(self, segments: List[Dict]) -> None:
        """智能标点插入。

        规则（按优先级）：
        1. 段开头是连词 → 前置逗号（如"所以我们" → "，所以我们"）
        2. 段首英文大写 → 前置句号
        3. 段尾语气词 → 问号或感叹号
        4. 段尾完整 → 句号

        Fixes in-place on ``segments``.
        """
        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not text:
                continue

            original = text
            prev_text = segments[i - 1].get("text", "").strip() if i > 0 else ""
            text = self._apply_punct_rules(text, prev_text, i, len(segments))
            seg["text"] = text

    def _apply_punct_rules(
        self, text: str, prev_text: str,
        seg_idx: int, total_segs: int,
    ) -> str:
        """逐规则应用标点修正。

        Args:
            text: 当前段文本。
            prev_text: 前一段文本（空字符串表示第一段）。
            seg_idx: 当前段索引。
            total_segs: 总段数。

        Returns:
            标点修正后的文本。
        """
        if not text:
            return text

        trimmed = text.strip()

        # ── Rule 1: 段首连词 → 前置逗号 ──
        for conj in self._CONJUNCTION_STARTS:
            if trimmed.startswith(conj):
                # 仅在"前段不以标点结尾"时插入逗号
                # 即：当前段是前段句子的延续才连接
                if prev_text and not re.search(r'[。！？.!?]$', prev_text.strip()):
                    # 前段以逗号/分号等软标点结尾时不再加前置逗号:
                    # 字幕是逐行显示的, "…，\n，然后…" 行首逗号是观感 bug
                    # (实测主人导出的 SRT 有 9 行行首逗号)。
                    if not re.search(r'[，、；;:,]$', prev_text.strip()):
                        text = text.replace(conj, "，" + conj, 1)
                # 若前段已有句号（或这是第一段），保持"所以"原样
                break
        trimmed = text.strip()

        # ── Rule 2: 段尾语气词 → 问号/感叹号 ──
        last_char = trimmed[-1] if trimmed else ""
        if last_char in self._QUESTION_ENDS:
            # 以"吗/呢/吧"结尾 → 大概率是疑问句
            text = text.rstrip() + "？"
        elif last_char in self._EXCLAMATION_ENDS:
            text = text.rstrip() + "！"

        trimmed = text.strip()

        # ── Rule 3: 段尾补句号 (v2.2.2 黑名单版) ──
        # 中文句尾除"悬空字"(单字及物动词/介词/方位词等, 见
        # _FRAGMENT_TAIL_CHARS) 外均可自然结束句子, 故黑名单排除碎片段,
        # 其余段尾补句号. 段尾语气词已由 Rule 2 处理 (吗→？ 啊/呀→！).
        if len(trimmed) > 4:
            last_ch = trimmed[-1]
            if (
                last_ch not in _FRAGMENT_TAIL_CHARS
                and last_ch not in "。！？.!?，、；：,;:"
                and not last_ch.isascii()
            ):
                text = text.rstrip() + "。"

        return text

    # ------------------------------------------------------------------
    # Phase 4 - 重叠去重
    # ------------------------------------------------------------------

    def _deduplicate_overlap(self, segments: List[Dict]) -> None:
        """检测并移除跨段重叠内容。

        能量凹陷切分（energy-dip）可能导致相邻段包含重复文本。
        例如段 4 的末尾内容在段 5 开头重复出现。

        算法：
            对于每对相邻段，检测前段末尾 N 字符
            是否与后段开头 N 字符相同（N >= 5）。
            若检测到重叠，从后段开头移除重叠部分。

        不是完美方案，但覆盖了最常见的重复模式。

        Fixes in-place on ``segments``.
        """
        if len(segments) < 2:
            return

        for i in range(len(segments) - 1):
            cur = segments[i]
            nxt = segments[i + 1]

            cur_text = cur.get("text", "").strip()
            nxt_text = nxt.get("text", "").strip()

            if not cur_text or not nxt_text:
                continue

            # 检测重叠：从 N=10 开始向下匹配
            overlap = ""
            for n in range(min(15, len(cur_text), len(nxt_text)), 3, -1):
                cur_tail = cur_text[-n:]
                nxt_head = nxt_text[:n]
                if cur_tail == nxt_head:
                    overlap = cur_tail
                    break

            if overlap and len(overlap) >= 5:
                # 防误判：纯英文单次重叠是碎片修复的结果，不是 ASR 能量重叠
                # ASR 能量重叠通常含中文、含空格、是短语级别
                is_single_en_word = (
                    len(overlap.split()) == 1
                    and all(ord(c) < 128 for c in overlap)
                )
                if is_single_en_word:
                    continue
                # 从后段开头移除重叠
                new_nxt = nxt_text[len(overlap):].strip()
                if new_nxt:
                    nxt["text"] = new_nxt
                    logger.debug(
                        "Dedup seg %d ↔ %d: removed '%s...' (overlap=%d chars)",
                        i, i + 1, overlap[:20], len(overlap),
                    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def deterministic_correct(
    segments: List[Dict],
    dictionary_path: Optional[str] = None,
    **kwargs,
) -> List[Dict]:
    """快捷函数：创建修正器并执行。

    Args:
        segments: 输入段列表。
        dictionary_path: 自定义词典路径（可选）。
        **kwargs: 传递给 ASRDeterministicCorrector 的其他参数。

    Returns:
        修正后的段列表。
    """
    dictionary = load_asr_dictionary(dictionary_path)
    corrector = ASRDeterministicCorrector(dictionary=dictionary, **kwargs)
    return corrector.correct(segments)
