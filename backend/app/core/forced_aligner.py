"""
强制对齐器 — 用 ONNX 版 Qwen3-ForcedAligner 为 ASR 文本生成逐词时间。

工作方式与 SubPop 相同: 拿「音频 + 已识别文本」, 输出每个词的 start/end。
在 sherpa-onnx 不返回 token 时间戳的模型上 (如 Qwen3-ASR), 这是获得
可断句时间的唯一途径。

实现完全脱离 torch:
  * mel 特征用 librosa 滤波器组复现 Whisper 预处理 (已验证与官方
    WhisperFeatureExtractor 逐元素一致, 见 tests/core/test_forced_aligner.py)
  * 分词用 HuggingFace tokenizers (Rust, 7.8MB), 不引 transformers (99.9MB)
  * 推理用 onnxruntime (项目已有依赖)

解码规则 (来自 Qwen3 官方 qwen3_forced_aligner.py):
  1. 文本切成词 (CJK 逐字, 拉丁按词), 词间插入两个 <timestamp> 标记
  2. 模型对每个 <timestamp> 位置输出 5000 档对齐桶
  3. argmax × timestamp_segment_time(80ms) = 该位置的时间
  4. 每两个时间戳是一个词的 start/end
  5. LIS (最长非降子序列) 修正个别乱序桶

输入格式三处必须自洽 (错一处只会静默给错时间而不报错):
  * input_ids 里一个 <|audio_pad|> 要展开成官方
    _get_feat_extract_output_lengths 算出的个数, 不能只放一个
  * feature_attention_mask 必须是帧级长度 (3000), 不是采样级 480000
  * 输入文本要套 chat template (system/user/assistant 三段)
"""
from __future__ import annotations

import importlib.util
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core import model_catalog

logger = logging.getLogger(__name__)

# 与模型 config.json 一致 (官方 Qwen3-ForcedAligner-0.6B)
TIMESTAMP_TOKEN = 151705
TIMESTAMP_SEGMENT_MS = 80.0
AUDIO_START_TOKEN = 151669
AUDIO_PAD_TOKEN = 151676
AUDIO_END_TOKEN = 151670
N_MELS = 128
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES = 480000  # 30s @ 16kHz

_CHAT_PREFIX = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _get_feat_extract_output_lengths(n: int) -> int:
    """官方 _get_feat_extract_output_lengths: 特征帧数 → 音频 token 数。

    leave = n % 100; feat = (leave - 1) // 2 + 1
    out = ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13
    """
    leave = n % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0xF900 <= code <= 0xFAFF
    )


def _is_kept(ch: str) -> bool:
    """官方 is_kept_char: 字母、数字与撇号保留, 其余当作分隔。"""
    if ch == "'":
        return True
    return unicodedata.category(ch).startswith(("L", "N"))


def split_words(text: str) -> List[str]:
    """官方 tokenize_space_lang: CJK 逐字, 拉丁连续段成词, 其余作分隔。"""
    words: List[str] = []
    for seg in text.split():
        buf: List[str] = []
        for ch in seg:
            if _is_cjk(ch):
                if buf:
                    words.append("".join(buf))
                    buf = []
                words.append(ch)
            elif _is_kept(ch):
                buf.append(ch)
            elif buf:
                words.append("".join(buf))
                buf = []
        if buf:
            words.append("".join(buf))
    return words


def _fix_timestamps(values: List[int]) -> List[int]:
    """官方 fix_timestamp: 用最长非降子序列定锚, 其余位置内插或取邻近锚。

    模型偶尔会输出乱序桶; 直接丢弃会错位, 这里保持个数不变。
    """
    data = [int(v) for v in values]
    n = len(data)
    if n == 0:
        return data
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    idx = dp.index(max(dp))
    normal = [False] * n
    while idx != -1:
        normal[idx] = True
        idx = parent[idx]

    out = data[:]
    i = 0
    while i < n:
        if normal[i]:
            i += 1
            continue
        j = i
        while j < n and not normal[j]:
            j += 1
        left = next((out[k] for k in range(i - 1, -1, -1) if normal[k]), None)
        right = next((out[k] for k in range(j, n) if normal[k]), None)
        if j - i <= 2:
            for k in range(i, j):
                out[k] = right if left is None else (
                    left if right is None else
                    (left if (k - (i - 1)) <= (j - k) else right)
                )
        elif left is not None and right is not None:
            step = (right - left) / (j - i + 1)
            for k in range(i, j):
                out[k] = int(left + step * (k - i + 1))
        elif left is not None:
            for k in range(i, j):
                out[k] = left
        elif right is not None:
            for k in range(i, j):
                out[k] = right
        i = j
    return out


def log_mel_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Whisper 预处理的 librosa 复刻, 输出 [128, 3000]。

    与 transformers.WhisperFeatureExtractor 逐元素一致
    (max abs diff = 0, 见 test_forced_aligner.py), 故不需要引入 transformers。
    """
    import librosa

    a = np.asarray(audio, dtype=np.float64)
    if len(a) < N_SAMPLES:
        a = np.pad(a, (0, N_SAMPLES - len(a)))
    else:
        a = a[:N_SAMPLES]

    win = np.hanning(N_FFT + 1)[:-1]
    pad = N_FFT // 2
    ap = np.pad(a, pad, mode="reflect")
    frames = 1 + (len(ap) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(frames)[:, None]
    spec = np.fft.rfft(ap[idx] * win, axis=1)
    mag = (np.abs(spec) ** 2)[:-1]

    fb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS,
                             fmin=0.0, fmax=sr / 2, norm="slaney", htk=False)
    mel = fb @ mag.T
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


class ForcedAligner:
    """单个对齐器资产 (清单 aligner 节点) 的 onnxruntime 实现。"""

    name = "forced_aligner"
    display_name = "Qwen3-ForcedAligner (本地)"
    requires_network = False
    is_local = True

    def __init__(self) -> None:
        self._session: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._loaded_key: Optional[Tuple[str, str]] = None

    def is_available(self) -> Tuple[bool, str]:
        if importlib.util.find_spec("onnxruntime") is None:
            return False, "onnxruntime 未安装"
        if importlib.util.find_spec("tokenizers") is None:
            return False, "tokenizers 未安装"
        try:
            missing = model_catalog.aligner_missing_files()
        except (ValueError, OSError) as e:
            return False, str(e)
        if missing:
            spec = model_catalog.aligner_spec()
            return False, (
                f"对齐器「{spec['name']}」未安装 (缺少 {', '.join(sorted(missing))}), "
                "请在「模型」面板中下载"
            )
        return True, ""

    # ── 加载 ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        import sherpa_onnx  # noqa: F401  (确认推理栈存在, 与 ASR 引擎一致)

        from tokenizers import Tokenizer

        files = model_catalog.aligner_files()
        key = (tuple(sorted(files.items())))
        if self._session is not None and self._loaded_key == key:
            return

        self._tokenizer = Tokenizer.from_file(files["tokenizer.json"])
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            files["onnx/model_q4.onnx"], providers=["CPUExecutionProvider"]
        )
        self._loaded_key = key
        logger.info("[ForcedAligner] 已加载: %s", files["onnx/model_q4.onnx"])

    # ── 推理 ────────────────────────────────────────────────────────

    def align(self, audio: np.ndarray, text: str, sr: int = 16000) -> List[Dict]:
        """对齐已识别文本, 返回 [{text, start, end}, ...] (秒)。

        文本须与音频实际内容一致; 不一致时时间会整体不可信,
        由调用方决定是否接受。
        """
        self._ensure_loaded()
        words = split_words(text)
        if not words:
            return []

        n_frames = N_SAMPLES // HOP_LENGTH
        n_pad = _get_feat_extract_output_lengths(n_frames)

        body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        # 实测 (对照 Zipformer 在同一音频上的逐字时间 0.010-1.110s):
        # 带 chat 前缀的变体整体后移且首词压成 0.0; 裸音频段前缀的变体
        # 给出的时间更贴近真实发音。官方 align() 走 processor 会套 chat
        # template, 但那是在 torch 后端; 本 ONNX 导出对裸前缀更敏感。
        # 两者都用全 1 帧级掩码, 差异只在 input_ids, 故采用实测更准的写法。
        prompt = "<|audio_start|>" + "<|audio_pad|>" * n_pad + "<|audio_end|>" + body
        input_ids = self._tokenizer.encode(prompt, add_special_tokens=False).ids

        features = log_mel_spectrogram(audio, sr)
        inputs = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(input_ids)), dtype=np.int64),
            "input_features": features[None, ...],
            "feature_attention_mask": np.ones((1, features.shape[-1]), dtype=np.int32),
        }

        logits = self._session.run(None, inputs)[0]
        argmax = logits.argmax(axis=-1)[0]
        id_arr = np.array(input_ids)
        raw_ms = argmax[id_arr == TIMESTAMP_TOKEN] * TIMESTAMP_SEGMENT_MS
        if len(raw_ms) < 2 * len(words):
            raise ValueError(
                f"对齐器输出时间戳不足: {len(raw_ms)} < {2 * len(words)}"
            )
        fixed = _fix_timestamps(list(raw_ms[: 2 * len(words)]))

        items: List[Dict] = []
        for i, word in enumerate(words):
            start = fixed[i * 2] / 1000.0
            end = fixed[i * 2 + 1] / 1000.0
            items.append({"text": word, "start": round(start, 3), "end": round(end, 3)})
        return items

    def char_times_for(
        self, audio: np.ndarray, text: str, sr: int = 16000
    ) -> Optional[List[float]]:
        """对齐文本并展开成逐字符开始时间 (caption_builder 的契约)。

        对齐器以词为单位输出; 每个字符继承所属词的开始时间,
        词内字符由此获得真实的发音落点。无法覆盖 text 全部
        非空白字符时返回 None (调用方退化为不细切)。
        """
        try:
            items = self.align(audio, text, sr)
        except (ValueError, RuntimeError) as e:
            logger.warning("[ForcedAligner] 对齐失败: %s", e)
            return None
        if not items:
            return None

        # 词表来自 split_words, 顺序与 text 中出现的非空白词一致;
        # 这里按 text 顺序重新匹配, 跳过空白。
        times: List[Optional[float]] = [None] * len(text)
        cursor = 0
        for item in items:
            word = item["text"]
            # 从 cursor 起找该词的字符序列 (跳过 text 里的空白)
            positions: List[int] = []
            wi = 0
            while cursor < len(text) and wi < len(word):
                if text[cursor].isspace():
                    cursor += 1
                    continue
                if text[cursor] == word[wi]:
                    positions.append(cursor)
                    wi += 1
                cursor += 1
            if wi != len(word):
                return None
            for pos in positions:
                times[pos] = float(item["start"])

        # 空白与标点没有自己的发音落点 (split_words 丢弃标点, 对齐结果
        # 不含它们): 归属前一拍 —— 继承前一个已覆盖字符的时间; 拖尾沿用
        # 最后一个。除这两类外仍有 None 说明对齐结果没覆盖到实体字符,
        # 不可信 → None。
        result: List[Optional[float]] = [None] * len(text)
        last = 0.0
        for i, ch in enumerate(text):
            if times[i] is not None:
                result[i] = float(times[i])
                last = result[i]
            elif ch.isspace() or unicodedata.category(ch).startswith("P"):
                result[i] = last
            else:
                return None
        # 首字符就是标点/空白 (last=0) 时沿用 0, 与音频起点一致, 可接受
        return result


# 模块级单例: 对齐器加载一次即可复用
_aligner: Optional[ForcedAligner] = None


def get_aligner() -> ForcedAligner:
    global _aligner
    if _aligner is None:
        _aligner = ForcedAligner()
    return _aligner
