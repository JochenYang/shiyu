"""
sherpa-onnx 离线识别引擎 — 清单驱动, 三个本地模型共用一条推理路径。

清单里的每个本地模型 (Zipformer-CTC-zh / SenseVoice-small / FireRedASR2-CTC)
都通过 sherpa-onnx 的 ``OfflineRecognizer`` 解码, 区别只在工厂方法与
token→字符时间的映射方式。合并成一条路径后:

  * 不再有自研 ONNX 推理 + 独立 fbank/LFR/CMVN 预处理 (SenseVoice 旧实现)
  * 不再有按字符数比例插值分配时间戳的细切 (改为 caption_builder 的 DP 断句)
  * 时间戳一律来自模型逐 token 输出, 缺失时降级为「不细切」而不是估算

逐 token 时间到逐字符时间的映射:
  * Zipformer-CTC-zh: token 是 byte-level BPE 符号, 经 ``bbpe_map`` 逐字节
    还原 (复刻 sherpa-onnx C++ ``SymbolTable::DecodeByteBpe``)
  * SenseVoice / FireRedASR2-CTC: token 本身就是字符 (中文) 或子词 (英文),
    按 token 覆盖的字符数直接展开
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from app.core import model_catalog
from app.core.engine_base import BaseTranscriber, Segment, TranscribeResult
from app.core.text_splitter import PAUSE_THRESHOLD

logger = logging.getLogger(__name__)

# 长音频分块: 30s 一块, 相邻块 2s 重叠上下文, 避免句子在块边界被截断
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2.0
# 切点搜索: 在块尾这段时间内找能量最低点作为切点, 避免在人声中间截断
BOUNDARY_SEARCH_SECONDS = 5.0
BOUNDARY_PROBE_SECONDS = 0.1

# CTC 峰值滞后补偿 (秒): CTC 在看到后续上下文后才发射 token, 时间戳整体滞后
# 约 100-200ms。可在 config.yaml engines.<name>.ctc_delay_sec 调整, 0 关闭。
CTC_DELAY_SEC = 0.15

# SenseVoice 的语言取值 (sherpa-onnx --sense-voice-language)
_SENSE_VOICE_LANGUAGES = {"auto", "zh", "en", "ko", "ja", "yue"}

def _chunk_bounds(audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
    """把长音频切成无缝相接的块, 切点落在块尾的低能量处。

    返回 [(start, stop), ...] 的**采样点**区间, 相邻区间首尾相接
    (前一块的 stop 就是后一块的 start), 因此内容不会丢失。

    切点搜索参照 SubPop 的实测做法: 在块尾 5s 内以 100ms 为窗找能量最低
    处收尾, 避免在人声中间拦腰截断; 找不到合适位置就按固定长度切。
    """
    chunk = CHUNK_SECONDS * sample_rate
    search = int(BOUNDARY_SEARCH_SECONDS * sample_rate)
    probe = max(1, int(BOUNDARY_PROBE_SECONDS * sample_rate))
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    total = len(audio)
    while cursor < total:
        stop = min(cursor + chunk, total)
        if stop < total:
            # 至少保留 1s 内容, 且不越过块尾
            lo = max(cursor + sample_rate, stop - search)
            hi = stop
            if hi - lo >= probe:
                # 按 probe 窗口算能量, 取最低处作为切点
                starts = range(lo, hi - probe + 1, probe)
                energies = [float(np.mean(np.square(audio[i:i + probe]))) for i in starts]
                best = next(iter(starts)) + int(np.argmin(energies)) * probe
                stop = min(stop, best + probe // 2)
        if stop <= cursor:
            stop = min(cursor + chunk, total)
        bounds.append((cursor, stop))
        cursor = stop
    return bounds


def _clean_tokens(tokens: List[str]) -> List[str]:
    """去掉 SenseVoice 的富标签 token (<|zh|> / <|NEUTRAL|> / <|Speech|> 等)。"""
    return [t for t in tokens if not (t.startswith("<|") and t.endswith("|>"))]


def _char_times_from_tokens(
    text: str,
    tokens: List[str],
    timestamps: List[float],
    byte_bpe: bool,
) -> Optional[List[float]]:
    """把逐 token 时间戳展开成逐字符开始时间。

    Args:
        text: 解码文本 (已清特殊符号)。
        tokens: ``result.tokens``。
        timestamps: ``result.timestamps`` (与 tokens 等长)。
        byte_bpe: token 是否为 byte-level BPE 符号 (Zipformer-CTC-zh)。

    Returns:
        与 ``text`` 等长的逐字符开始时间; 无法可靠映射时返回 ``None``
        (调用方退化为不细切, 绝不用比例插值伪造时间)。
    """
    if not text or not tokens or not timestamps:
        return None
    if not isinstance(tokens, (list, tuple)) or not isinstance(timestamps, (list, tuple)):
        return None
    if len(tokens) != len(timestamps):
        return None

    if byte_bpe:
        try:
            from app.core.bbpe_map import tokens_to_char_times
        except ImportError:
            return None
        try:
            decoded = tokens_to_char_times(list(tokens), list(timestamps))
        except Exception:
            return None
        if decoded is None:
            return None
        char_times, decoded_text = decoded
        if decoded_text.replace(" ", "") != text.replace(" ", ""):
            return None
        if len(char_times) != len(text):
            return None
        return [float(t) for t in char_times]

    # token 即字符 (中文) 或子词 (英文): 按 token 覆盖的字符数展开
    clean_tokens = _clean_tokens(list(tokens))
    clean_times = [float(t) for t, raw in zip(timestamps, tokens) if not (str(raw).startswith("<|") and str(raw).endswith("|>"))]
    if len(clean_tokens) != len(clean_times):
        return None
    # 按「非空白字符序列」对齐: 中文无空格、英文带空格都能处理,
    # 且空白位置不需要自己的时间。BPE 类词表的词首空格标记 (U+2581)
    # 也在此归一掉, 以后若接入这类模型无需再改这里。
    normalized = [t.replace("\u2581", " ") for t in clean_tokens]
    flat_chars: List[str] = []
    flat_times: List[float] = []
    for token, time_value in zip(normalized, clean_times):
        for ch in token:
            if ch.isspace():
                continue
            flat_chars.append(ch)
            flat_times.append(time_value)

    text_positions = [(i, ch) for i, ch in enumerate(text) if not ch.isspace()]
    if [ch for _, ch in text_positions] != flat_chars:
        return None

    char_times = [0.0] * len(text)
    for (index, _), time_value in zip(text_positions, flat_times):
        char_times[index] = time_value
    # 空白位置沿用前一个非空白字符的时间
    last = flat_times[0] if flat_times else 0.0
    for i, ch in enumerate(text):
        if ch.isspace():
            char_times[i] = last
        else:
            last = char_times[i]
    return char_times


class SherpaOfflineEngine(BaseTranscriber):
    """单个本地模型 (清单项) 的 sherpa-onnx 推理实现。"""

    # 由子类或实例覆盖
    name = "sherpa_local"
    display_name = "sherpa-onnx (本地)"
    requires_network = False
    is_local = True

    #: 词表类型 → token→字符时间的映射方式。清单里的 engine 字段直接决定,
    #  不从子类继承 —— 否则注册中心实例化基类时会把 byte_bpe 丢成 False,
    #  导致 zipformer 的 BBPE 解码静默失败 (逐字符时间全部拿不到, 长段无法断句)。
    _BYTE_BPE_ENGINES = frozenset({"zipformer"})
    def __init__(
        self,
        model_id: str,
        num_threads: int = 4,
        ctc_delay_sec: float = CTC_DELAY_SEC,
    ):
        self._model_id = model_id
        spec = model_catalog.model_spec(model_id)
        self.name = spec.get("engine_name", self.name)
        self.display_name = spec["name"]
        # 是否 byte-level BPE 词表: 由清单的 engine 类型决定 (见 _BYTE_BPE_ENGINES)
        self.byte_bpe = spec["engine"] in self._BYTE_BPE_ENGINES
        self._num_threads = max(1, int(num_threads))
        self._ctc_delay_sec = max(0.0, float(ctc_delay_sec))
        self._recognizer: Optional[Any] = None
        self._loaded_key: Optional[Tuple[str, str]] = None

    # ── 配置 ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, model_id: str, config: dict) -> "SherpaOfflineEngine":
        """按 config.yaml 的 ``engines.<name>`` 片段构造。"""
        return cls(
            model_id=model_id,
            num_threads=int(config.get("num_threads", 4)),
            ctc_delay_sec=float(config.get("ctc_delay_sec", CTC_DELAY_SEC)),
        )

    def is_available(self) -> Tuple[bool, str]:
        """模型是否就绪 (只查清单文件, 不触发下载)。"""
        if importlib.util.find_spec("sherpa_onnx") is None:
            return False, "sherpa-onnx 未安装"

        if not model_catalog.is_installed(self._model_id):
            missing = model_catalog.missing_files(self._model_id)
            spec = model_catalog.model_spec(self._model_id)
            return False, (
                f"模型「{spec['name']}」未安装 (缺少 {', '.join(sorted(missing))}), "
                "请在「模型」面板中下载"
            )
        return True, ""

    # ── 推理 ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """惰性创建 OfflineRecognizer; 模型路径变化时重建。

        清单里每个模型的文件名与工厂参数并不一一对应 (Whisper 是 encoder+decoder,
        Qwen3-ASR 还要 conv_frontend 与 tokenizer 目录), 所以这里按文件名特征
        组装参数, 不在代码里写死单文件模型的两个路径。
        """
        import sherpa_onnx

        files = model_catalog.resolve_files(self._model_id)
        factory_name = self._factory_name()
        key = (factory_name, tuple(sorted(files.items())))
        if self._recognizer is not None and self._loaded_key == key:
            return

        factory = getattr(sherpa_onnx.OfflineRecognizer, factory_name)
        kwargs: Dict[str, Any] = {"num_threads": self._num_threads, "provider": "cpu"}
        kwargs.update(self._factory_args(files, factory_name))

        self._recognizer = factory(**kwargs)
        self._loaded_key = key
        logger.info("[Sherpa] %s 已加载 (%s): %s",
                    self._model_id, factory_name, files)

    def _factory_args(self, files: Dict[str, str], factory_name: str) -> Dict[str, Any]:
        """按工厂方法把清单文件映射成关键字参数。

        注: Whisper 不在清单中。sherpa-onnx 的 Whisper 导出（含官方
        csukuangfj/sherpa-onnx-whisper-*）的 decoder 只有 logits 与
        self-k/v cache, 缺 cross-attention, 因此 enable_token_timestamps
        打开也拿不到逐 token 时间（实测 timestamps=0, 整段退化成 start=end=0）。
        没有时间戳就无法在测量词边界上断句, 接入只会让字幕变差。
        """
        if factory_name == "from_qwen3_asr":
            # Qwen3-ASR 是 LLM 逐 token 生成文本, sherpa-onnx 默认
            # max_new_tokens=128: 30s 块内文本超过 128 token 就整段截断
            # (实测返回空结果, 只在 C++ 层报一行警告)。按块长给足余量:
            # 中文 30s ≈ 300+ 字, 512 足够。
            return {
                "conv_frontend": self._pick(files, "conv_frontend"),
                "encoder": self._pick(files, "encoder"),
                "decoder": self._pick(files, "decoder"),
                "tokenizer": self._tokenizer_dir(files),
                "max_new_tokens": 512,
            }
        if factory_name == "from_sense_voice":
            # use_itn=True 让 SenseVoice 输出阿拉伯数字与标点 (官方推荐)
            return {
                "model": self._pick(files, "model"),
                "tokens": self._pick(files, "tokens"),
                "use_itn": True,
                "language": "",
            }
        # zipformer / fire_red_asr_ctc: 单 onnx + 词表
        return {
            "model": self._pick(files, "model"),
            "tokens": self._pick(files, "tokens"),
        }

    @staticmethod
    def _pick(files: Dict[str, str], needle: str) -> str:
        """按文件名子串取路径 (如 'encoder' → large-v3-encoder.int8.onnx)。"""
        matches = [p for name, p in files.items() if needle in name.lower()]
        if len(matches) != 1:
            raise ValueError(f"模型文件中匹配 {needle!r} 的应恰为 1 个, 实际 {matches}")
        return matches[0]

    @staticmethod
    def _tokenizer_dir(files: Dict[str, str]) -> str:
        """Qwen3-ASR 的 tokenizer 是一个目录 (含 vocab.json / merges.txt)。"""
        import os

        for name, path in files.items():
            if name.startswith("tokenizer/"):
                return os.path.dirname(path)
        raise ValueError("Qwen3-ASR 缺少 tokenizer 目录")

    def _factory_name(self) -> str:
        """清单 ``engine`` 字段 → sherpa-onnx 工厂方法名。"""
        engine = model_catalog.model_spec(self._model_id)["engine"]
        mapping = {
            "zipformer": "from_zipformer_ctc",
            "sensevoice": "from_sense_voice",
            "fire_red_asr_ctc": "from_fire_red_asr_ctc",
            "qwen3_asr": "from_qwen3_asr",
        }
        try:
            return mapping[engine]
        except KeyError:
            raise ValueError(f"未知的引擎类型: {engine}") from None

    @staticmethod
    def _read_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """读音频并统一到 16kHz 单声道 float32。"""
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        return audio, sample_rate

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        **kwargs,
    ) -> TranscribeResult:
        """转录音频文件, 返回带逐字符时间戳的段列表。"""
        self._ensure_loaded()
        audio, sample_rate = self._read_audio(audio_path)
        if len(audio) == 0:
            return TranscribeResult(segments=[], language=language)

        chunk_size = CHUNK_SECONDS * sample_rate
        if len(audio) <= chunk_size:
            segments = self._decode(audio, sample_rate, time_offset=0.0, language=language)
            segments = self._finish(segments)
            return TranscribeResult(segments=segments, language=language)

        all_segments: List[Segment] = []
        covered_end = 0.0
        for chunk_start, chunk_stop in _chunk_bounds(audio, sample_rate):
            # 第二块起向左回退重叠区, 让块边界的句子有完整解码上下文
            start_idx = max(0, chunk_start - int(OVERLAP_SECONDS * sample_rate))
            chunk_end = chunk_stop
            chunk = audio[start_idx:chunk_end]
            time_offset = start_idx / sample_rate

            # 内容区 [lo, hi) 必须与相邻块**接续无缝**: 重叠区只是解码上下文,
            # 不能从内容区里扣 (否则重叠那段两边都不认, 内容丢失: 实测 30s
            # 块边界丢 2.1s)。
            lo = max(chunk_start / sample_rate, covered_end)
            hi = chunk_stop / sample_rate
            for seg in self._decode(chunk, sample_rate, time_offset, language):
                # 无逐字符时间的段 (Qwen3-ASR 等) start=end=块起点, 裁剪只会
                # 把它们全部丢掉。先尝试用本地强制对齐器补时间, 补上后再裁剪;
                # 对齐器不可用时退回「按块区间归位」: 该段文字来自本块,
                # 起止就用块的内容区, 不再是编造的逐字符时间。
                if not getattr(seg, "char_times", None):
                    seg = self._attach_segment_times(seg, lo, hi, chunk, time_offset)
                    if seg is None:
                        continue
                trimmed = self._trim_segment(seg, lo, hi)
                if trimmed is None:
                    continue
                all_segments.append(trimmed)
                covered_end = max(covered_end, trimmed.end)
        return TranscribeResult(segments=self._finish(all_segments), language=language)

    def _decode(
        self,
        audio: np.ndarray,
        sample_rate: int,
        time_offset: float,
        language: str,
    ) -> List[Segment]:
        """解码一块音频, 返回带绝对时间与逐字符时间的段。"""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        del stream
        return self._result_to_segments(result, time_offset, language)

    def _result_to_segments(
        self,
        result: Any,
        time_offset: float = 0.0,
        language: str = "auto",
    ) -> List[Segment]:
        """把 sherpa-onnx 结果转成 Segment 列表。

        sherpa-onnx 可能已按静音内部切分 (``segment_texts`` 非空), 也可能只给
        整段输出; 两种情况都保留逐字符时间供 DP 断句使用。
        """
        seg_texts = getattr(result, "segment_texts", None)
        seg_ts = getattr(result, "segment_timestamps", None)
        seg_dur = getattr(result, "segment_durations", None)

        if (
            isinstance(seg_texts, list)
            and isinstance(seg_ts, list)
            and seg_texts
            and len(seg_texts) == len(seg_ts)
        ):
            segments, cursor = [], 0
            # 段级结果也带逐 token 时间 (result.tokens 与 result.text 顺序一致);
            # 把 token 按段文本顺序消费, 每段得到自己的逐字符时间。
            # SenseVoice 的 token 即字符 (含标点), 可直接顺序对齐。
            all_tokens = list(getattr(result, "tokens", None) or [])
            all_times = list(getattr(result, "timestamps", None) or [])
            token_ok = len(all_tokens) == len(all_times) and bool(all_tokens)

            for i, raw_text in enumerate(seg_texts):
                text = (raw_text or "").strip()
                if not text:
                    continue
                start = max(0.0, float(seg_ts[i]) + time_offset - self._ctc_delay_sec)
                duration = (
                    float(seg_dur[i])
                    if isinstance(seg_dur, list) and i < len(seg_dur)
                    else 0.0
                )
                seg = Segment(start=round(start, 3), end=round(start + duration, 3), text=text)

                # 从游标起消费与本段文本非空白字符数相同的 token
                if token_ok:
                    want = sum(1 for c in text if not c.isspace())
                    taken_t = all_tokens[cursor: cursor + want]
                    taken_s = all_times[cursor: cursor + want]
                    # token 顺序必须与文本一致, 否则不猜
                    if [t for t in taken_t if not str(t).isspace()] == [c for c in text if not c.isspace()]:
                        seg.char_times = [
                            max(0.0, float(t) + time_offset - self._ctc_delay_sec)
                            for t in taken_s
                        ]
                        cursor += want
                segments.append(seg)
            if segments:
                return segments

        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return []

        timestamps = getattr(result, "timestamps", None) or []
        if not isinstance(timestamps, list) or not timestamps:
            # 没有 token 时间: 不猜时长, 交给上层按段处理
            return [Segment(start=time_offset, end=time_offset, text=text)]

        start_sec = max(0.0, float(timestamps[0]) + time_offset - self._ctc_delay_sec)
        end_sec = max(0.0, float(timestamps[-1]) + time_offset - self._ctc_delay_sec)

        import re

        cleaned_text = re.sub(r"<unk>\s*", "", text).strip()
        char_times = None
        if cleaned_text:
            char_times = _char_times_from_tokens(
                cleaned_text,
                list(getattr(result, "tokens", None) or []),
                list(timestamps),
                self.byte_bpe,
            )
            if char_times:
                # 与段起止同基准减补偿, 再加块偏移
                char_times = [max(0.0, t - self._ctc_delay_sec + time_offset) for t in char_times]

        return [
            Segment(
                start=round(start_sec, 3),
                end=round(end_sec, 3),
                text=cleaned_text,
                char_times=char_times,
            )
        ]

    def _finish(self, segments: List[Segment]) -> List[Segment]:
        """统一后处理: 断句 → ITN → 去重 → 强制时间单调。"""
        segments = self._segment_captions(segments)
        segments = self._apply_itn(segments)
        segments = self._dedup_overlap(segments)
        segments = self._enforce_monotonic(segments)
        return segments

    @staticmethod
    def _enforce_monotonic(segments: List[Segment]) -> List[Segment]:
        """强制字幕时间单调不重叠。

        分块解码时相邻块的重叠区可能让后一行起点早于前行终点 (实测 57.85 < 57.93)。
        字幕文件不允许重叠区间, 这里把重叠部分归给先出现的一行: 前行终点收到
        后行起点。收尾后若前行变成零宽则与后行合并, 避免产出空区间。

        注意这里只调整**已经测量到**的边界 (行起点), 不发明新的时间。
        """
        if not segments:
            return segments

        result: List[Segment] = []
        for seg in segments:
            if result:
                previous = result[-1]
                if seg.start < previous.end:
                    if seg.start > previous.start:
                        previous.end = seg.start
                    else:
                        # 完全被前一行覆盖: 文字并入前一行, 丢弃空区间
                        previous.text += seg.text
                        previous.end = max(previous.end, seg.end)
                        continue
            result.append(seg)
        return result

    @staticmethod
    def _apply_itn(segments: List[Segment]) -> List[Segment]:
        """中文数字 → 阿拉伯数字 (九百八十 → 980)。

        必须在断句之后做: ITN 会改变文本长度, 先做就会让逐字符时间与文本
        对不上, 退回到按比例插值 —— 那正是要消除的漂移来源。断句只改时间
        边界, 不改文本, 所以在这里改写文本是安全的。
        """
        try:
            from app.core.itn import itn_normalize
        except ImportError:
            return segments
        for seg in segments:
            normalized = itn_normalize(seg.text)
            if normalized != seg.text:
                seg.text = normalized
                # 文本长度已变, 旧的逐字符时间不再对应, 显式丢弃而不是留下
                # 长度不匹配的数组让下游误用
                seg.char_times = None
        return segments

    def _segment_captions(self, segments: List[Segment]) -> List[Segment]:
        """用 DP 断句把长段切成字幕行; 无逐字符时间时保持原样。"""
        from app.core.caption_builder import build_captions

        output: List[Segment] = []
        for seg in segments:
            times = seg.char_times
            if not times or len(times) != len(seg.text):
                output.append(seg)
                continue
            try:
                rows = build_captions(
                    seg.text,
                    times,
                    seg.start,
                    seg.end,
                )
            except Exception as e:
                logger.warning("[Sherpa] 断句失败, 保留原段: %s", e)
                output.append(seg)
                continue
            if not rows:
                output.append(seg)
                continue
            for row in rows:
                row_times = row.get("char_times")
                if row_times and len(row_times) == len(row["text"]):
                    output.append(
                        Segment(start=row["start"], end=row["end"], text=row["text"],
                                char_times=row_times)
                    )
                else:
                    # 清理后长度对不上 (罕见): 保测量起点, 丢行内细时间
                    output.append(
                        Segment(start=row["start"], end=row["end"], text=row["text"])
                    )
        return output

    def _attach_segment_times(
        self, seg: Segment, lo: float, hi: float, chunk: np.ndarray, time_offset: float
    ) -> Optional[Segment]:
        """给无逐字符时间的段补上时间。

        首选强制对齐器 (逐字符测量时间); 不可用时退化为块级归位
        (start=lo, end=块内有内容的部分)。两种都拿不到 → None。

        ``time_offset`` 是本块在整段音频里的起点。对齐器只看 chunk,
        返回的是**相对块起点**的时间, 必须加上 time_offset 才是绝对
        时间 —— 否则第二块起会把段裁到 [lo, hi) 之外而全部丢弃
        (实测: 10 分钟视频只剩第一个 30s 块的字幕)。
        """
        try:
            from app.core.forced_aligner import get_aligner

            aligner = get_aligner()
            ok, _ = aligner.is_available()
            if ok:
                times = aligner.char_times_for(chunk, seg.text)
                if times and len(times) == len(seg.text):
                    seg.char_times = [t + time_offset for t in times]
                    return seg
        except ImportError:
            pass

        # 对齐器不可用: 文字至少来自本块, 用内容区做段区间 (绝对时间)
        if hi > lo:
            seg.start = lo
            seg.end = hi
            return seg
        return None


    def _trim_segment(self, seg: Segment, lo: float, hi: float) -> Optional[Segment]:
        """把段裁剪到块内容区 [lo, hi), 同步裁剪文本与逐字符时间。"""
        char_times = seg.char_times
        if not (isinstance(char_times, (list, tuple)) and len(char_times) == len(seg.text)):
            # 无逐字符时间: 直接钳制, 不尝试对齐
            seg.start = max(seg.start, lo)
            seg.end = min(seg.end, hi)
            if seg.end <= seg.start + 0.05:
                return None
            return seg

        i0 = next((i for i, t in enumerate(char_times) if t >= lo), None)
        i1 = len(char_times) - 1
        for i in range(len(char_times) - 1, -1, -1):
            if char_times[i] < hi:
                i1 = i
                break
        if i0 is None or i1 < i0 or char_times[i0] >= hi:
            return None

        # 对齐停顿点: 起点回退到句首, 终点扩展到句尾, 避免半句字幕
        while i0 > 0 and char_times[i0] - char_times[i0 - 1] <= PAUSE_THRESHOLD:
            i0 -= 1
        while i1 + 1 < len(char_times) and char_times[i1 + 1] - char_times[i1] <= PAUSE_THRESHOLD:
            i1 += 1

        if i0 >= len(seg.text) or not seg.text[i0 : i1 + 1].strip():
            return None
        seg.text = seg.text[i0 : i1 + 1]
        seg.char_times = list(char_times[i0 : i1 + 1])
        seg.start = max(seg.start, min(lo, float(seg.char_times[0])))
        seg.end = max(min(seg.end, hi), float(seg.char_times[-1]))
        seg.end = max(seg.end, seg.start)
        return seg

    @staticmethod
    def _dedup_overlap(segments: List[Segment]) -> List[Segment]:
        """相邻段「时间重叠 + 文本包含」→ 去重 (分块解码的重叠产物)。"""
        if not segments:
            return segments
        result: List[Segment] = [segments[0]]
        for seg in segments[1:]:
            prev = result[-1]
            if seg.start < prev.end + 0.05 and prev.start < seg.end - 0.05:
                a, b = prev.text, seg.text
                if a and b:
                    if b == a:
                        continue
                    if b in a and len(b) < len(a):
                        continue
                    if a in b and len(a) < len(b):
                        result[-1] = seg
                        continue
            result.append(seg)
        return result


class ZipformerEngine(SherpaOfflineEngine):
    """Zipformer-CTC-zh 的具名别名 (词表类型由清单决定, 见 _BYTE_BPE_ENGINES)。"""


class SenseVoiceEngine(SherpaOfflineEngine):
    """SenseVoice-small 的具名别名。"""


class FireRedEngine(SherpaOfflineEngine):
    """FireRedASR2-CTC 的具名别名。"""
