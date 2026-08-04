"""
Paraformer-large ASR engine — sherpa-onnx wrapper.

模型: sherpa-onnx-paraformer-zh-2023-09-14
- int8 量化, 232MB
- Paraformer (阿里达摩院) CIF (Continuous Integrate-and-Fire) 架构
- **显式 token boundary 预测** — 段边界 ±20ms 量级, 优于 CTC 的 ±50-300ms
- 中文专用 (8416 tokens)

Shiyu v2.1 第二引擎, 主人测试 Zipformer CTC 边界不准后切换.
SenseVoice 仍保留作为多语言备选.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .engine_base import BaseTranscriber, Segment, TranscribeResult

if TYPE_CHECKING:
    import sherpa_onnx  # noqa: F401


def _merge_chunks_segments(
    chunk_segments_list: List[List[Segment]],
    overlap_tolerance: float = 0.05,
) -> List[Segment]:
    """拼接各块解码段, 按绝对时间戳过滤分块重叠区.

    Paraformer 每块只输出 1 个长段 (CIF 无 VAD 切分), 分块加 2s 上下文
    重叠后, 同一句话会被相邻两块各转录一次. 这里维护 covered_end
    (已保留段的覆盖终点), 只保留 start >= covered_end - overlap_tolerance
    的段:

    - overlap_tolerance=50ms: 吸收 CIF 边界噪声, 防止误杀恰好跨边界的段
    - 某块所有段都落在重叠区 (整块被丢弃) → 跳过该块, 不推进 covered_end,
      避免覆盖范围虚增导致后续段的起点被误判为重叠

    入参按块顺序排列 (块内段按时间有序), 输出同样保持时间有序.
    纯函数, 不依赖引擎实例, 便于用构造数据单测.
    """
    merged: List[Segment] = []
    covered_end = 0.0
    for chunk_segments in chunk_segments_list:
        kept = []
        for seg in chunk_segments:
            if seg.start >= covered_end - overlap_tolerance:
                kept.append(seg)
            elif seg.end > covered_end:
                # 跨边界段: 裁剪起点到 covered_end, 保留句尾内容.
                # 注意: 会原地修改 seg.start (入参 Segment 对象)
                seg.start = covered_end
                kept.append(seg)
        if not kept:
            continue  # 整块落在重叠区 → 跳过该块, covered_end 不推进
        merged.extend(kept)
        covered_end = max(covered_end, max(seg.end for seg in kept))
    return merged


class ParaformerEngine(BaseTranscriber):
    """Paraformer-large via sherpa-onnx (int8, 232MB, CIF 架构).

    优势: 显式 token 边界预测, 段起点/终点比 Zipformer-CTC 准 5-10x.
    缺点: 仅支持中文, 模型比 Zipformer 略大 (232MB vs 350MB int8).
    """

    name = "paraformer_local"
    display_name = "Paraformer-large (中文, sherpa-onnx)"
    requires_network = False
    is_local = True

    def __init__(self, model_dir: str, num_threads: int = 4):
        self._model_dir = model_dir
        self._num_threads = num_threads
        self._recognizer: Optional["sherpa_onnx.OfflineRecognizer"] = None

    @classmethod
    def from_config(cls, config: dict) -> "ParaformerEngine":
        """Factory: create a ParaformerEngine from config dict."""
        import os
        from pathlib import Path
        model_dir = config.get("model_dir") or "~/.shiyu/models/paraformer-large"
        model_dir = str(Path(os.path.expanduser(model_dir)).resolve())
        num_threads = int(config.get("num_threads", 4))
        return cls(model_dir=model_dir, num_threads=num_threads)

    def _ensure_loaded(self):
        """Lazily create sherpa-onnx OfflineRecognizer for Paraformer."""
        if self._recognizer is not None:
            return
        import sherpa_onnx
        model_path = os.path.join(self._model_dir, "model.int8.onnx")
        tokens_path = os.path.join(self._model_dir, "tokens.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Paraformer model not found: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Paraformer tokens not found: {tokens_path}")
        # Paraformer int8 模型约 232MB; < 100MB 视为不完整
        if os.path.getsize(model_path) < 100_000_000:
            raise FileNotFoundError(
                f"Paraformer model too small (incomplete?): {model_path} "
                f"({os.path.getsize(model_path)} bytes)"
            )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=self._num_threads,
            provider="cpu",
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        **kwargs,
    ) -> TranscribeResult:
        """转录音频文件, 返回带时间戳的 Segment 列表.

        Paraformer-CIF 输出特点:
        - segment_texts/timestamps/durations 都为空 (CIF 单段输出, 无 VAD 切分)
        - 但 tokens[] 和 timestamps[] 是 token-level 显式对齐 (CIF 边界)
        - 段边界精度 ±20ms, 比 Zipformer-CTC 准 5-10x
        """
        import sherpa_onnx
        import soundfile as sf
        import numpy as np

        self._ensure_loaded()

        # 读取音频
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except Exception as e:
                raise RuntimeError(
                    f"无法重采样 {sample_rate}Hz → 16kHz (需要 librosa): {e}"
                )

        if len(audio) == 0:
            return TranscribeResult(segments=[], language="zh")

        # 长音频分块: Paraformer 也怕 9 分钟音频一次性 ONNX 推理爆内存
        CHUNK_SIZE = 30 * 16000  # 30s
        OVERLAP_SAMPLES = int(2.0 * sample_rate)  # 2s 上下文重叠, 缓解块边界截断
        total_samples = len(audio)

        if total_samples <= CHUNK_SIZE:
            return TranscribeResult(
                segments=self._post_process(
                    self._decode_chunk(audio, sample_rate, time_offset=0.0)
                ),
                language="zh",
            )

        # 每块独立解码, 拼接时按时间戳去重重叠区
        chunk_segments_list = []
        for chunk_start in range(0, total_samples, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_samples)
            # 第二块起向左多取 2s 上下文, 句子跨块时不再在块边界截断
            decode_start = max(0, chunk_start - OVERLAP_SAMPLES)
            chunk = audio[decode_start:chunk_end]
            # 时间戳相对 chunk 音频起点 (decode_start), 偏移必须以它为基准;
            # 若按 chunk_start 算会整体 +2s 错位 (重叠区时长)
            time_offset = decode_start / sample_rate
            chunk_segments_list.append(
                self._decode_chunk(chunk, sample_rate, time_offset)
            )

        return TranscribeResult(
            segments=self._post_process(
                self._merge_chunks_segments(chunk_segments_list)
            ),
            language="zh",
        )

    @staticmethod
    def _post_process(segments: List[Segment]) -> List[Segment]:
        """后处理 Paraformer 输出 (Shiyu v2.1 流水线):
        1) ITN: 中文数字 → 阿拉伯数字
        2) 清 <unk>
        3) 按标点+长度细切: 一段 29s → 多段 5-7s, 改善可读性

        Paraformer-CIF 单段输出 (无 VAD 切分), 段长 20-30s, 必须 splitter 切细.
        """
        try:
            from .itn import itn_normalize
        except ImportError:
            itn_normalize = None

        # Step 1+2: 字符级后处理
        cleaned: List[Segment] = []
        for s in segments:
            text = s.text
            if itn_normalize is not None:
                text = itn_normalize(text)
            import re
            text = re.sub(r"<unk>\s*", "", text).strip()
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            cleaned.append(Segment(start=s.start, end=s.end, text=text))

        if not cleaned:
            return []

        # Step 3: 按标点+长度细切
        try:
            from .text_splitter import split_text_at_boundaries
            seg_dicts = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in cleaned
            ]
            split_dicts = split_text_at_boundaries(seg_dicts)
            return [
                Segment(
                    start=round(d["start"], 3),
                    end=round(d["end"], 3),
                    text=d["text"],
                )
                for d in split_dicts
            ]
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "[Paraformer] split_text_at_boundaries failed, returning as-is: %s", e,
            )
            return cleaned

    def _decode_chunk(
        self, audio, sample_rate: int, time_offset: float
    ) -> List[Segment]:
        """Decode a single audio chunk, return segments with absolute timestamps."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        del stream
        return self._result_to_segments(result, time_offset=time_offset)

    def _result_to_segments(
        self, result, time_offset: float = 0.0
    ) -> List[Segment]:
        """把 sherpa-onnx Paraformer OfflineRecognitionResult 转成 List[Segment].

        Paraformer 输出:
          - segment_texts/timestamps/durations: 通常为空 (CIF 单段)
          - tokens[]: 字符级列表 (含 BPE 拆开的空格/标点)
          - timestamps[]: 每个 token 的时间戳 (秒, 绝对时间)
        策略: 整段输出 (1.44s-121.76s), 不切分, 让 text_splitter 后续按标点切.
        """
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return []

        ts = getattr(result, "timestamps", None) or []
        if not ts:
            return [Segment(start=0.0, end=0.0, text=text)]

        # Paraformer 的 ts[] 是 token 级别; text 第一个 token 时间为 start, 最后一个为 end
        # 取首个 token 时间 (CIF 边界, 比 CTC 字符对齐准)
        start_sec = float(ts[0]) + time_offset
        end_sec = float(ts[-1]) + time_offset

        if end_sec <= start_sec:
            return []
        return [Segment(
            start=round(start_sec, 3),
            end=round(end_sec, 3),
            text=text,
        )]

    def is_available(self) -> Tuple[bool, str]:
        """检查 Paraformer 模型文件是否齐备."""
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False, "sherpa-onnx not installed (pip install sherpa-onnx>=1.10.21)"

        from pathlib import Path
        model_dir = Path(self._model_dir)
        if not model_dir.exists():
            return False, f"model directory not found: {model_dir}"
        model_file = model_dir / "model.int8.onnx"
        tokens_file = model_dir / "tokens.txt"
        if not model_file.exists() or model_file.stat().st_size < 100_000_000:
            return False, f"model.int8.onnx missing or too small: {model_file}"
        if not tokens_file.exists() or tokens_file.stat().st_size < 100:
            return False, f"tokens.txt missing or empty: {tokens_file}"
        return True, ""


# ── BaseTranscriber Adapter ────────────────────────────────────────


class ParaformerTranscriber(BaseTranscriber):
    """Adapter wrapping ParaformerEngine as a BaseTranscriber."""

    name = "paraformer_local"
    display_name = "Paraformer-large (中文, sherpa-onnx)"
    requires_network = False
    is_local = True

    def __init__(self, model_dir: str, num_threads: int = 4):
        self._engine = ParaformerEngine(model_dir, num_threads)

    @classmethod
    def from_config(cls, config: dict) -> "ParaformerTranscriber":
        kwargs = ParaformerEngine.init_kwargs_from_config(config)
        return cls(**kwargs)

    def transcribe(self, audio_path: str, language: str = "auto", **kwargs) -> TranscribeResult:
        return self._engine.transcribe(audio_path, language, **kwargs)

    def is_available(self) -> Tuple[bool, str]:
        return self._engine.is_available()


# 给 ParaformerEngine 加 init_kwargs_from_config 类方法 (跟 ZipformerEngine 一致)
@classmethod
def init_kwargs_from_config(cls, config: dict) -> dict:
    import os
    from pathlib import Path
    model_dir = config.get("model_dir") or "~/.shiyu/models/paraformer-large"
    model_dir = str(Path(os.path.expanduser(model_dir)).resolve())
    num_threads = config.get("num_threads", 4)
    return {"model_dir": model_dir, "num_threads": num_threads}


ParaformerEngine.init_kwargs_from_config = init_kwargs_from_config
