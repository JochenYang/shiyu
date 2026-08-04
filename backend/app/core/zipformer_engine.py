"""
Zipformer-CTC-zh ASR engine — sherpa-onnx wrapper.

模型: sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03
- int8 量化, AISHELL-1 WER 1.74% (官方)
- Zipformer (k2-fsa/icefall) SOTA 流式架构, 离线 one-shot 解码
- CTC 帧级时间戳 (per-token start time)
- 中文专用 (BBPE 1000 vocab)

Shiyu v2.0 默认 ASR 引擎。SenseVoice 保留作为多语言备选。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from .engine_base import BaseTranscriber, Segment, TranscribeResult

if TYPE_CHECKING:
    import sherpa_onnx  # noqa: F401  (for type hints only)


class ZipformerEngine(BaseTranscriber):
    """Zipformer-CTC-zh via sherpa-onnx (int8, 350MB).

    Default ASR engine for Shiyu v2.0 — Chinese only, offline one-shot decoding.
    """

    name = "zipformer_local"
    display_name = "Zipformer CTC (中文, sherpa-onnx)"
    requires_network = False
    is_local = True

    def __init__(self, model_dir: str, num_threads: int = 4):
        self._model_dir = model_dir
        self._num_threads = num_threads
        # sherpa_onnx.OfflineRecognizer 实例 (lazy 加载, 用 object 避开 TYPE_CHECKING 时的 import)
        self._recognizer: Optional["sherpa_onnx.OfflineRecognizer"] = None

    @classmethod
    def from_config(cls, config: dict) -> "ZipformerEngine":
        """Factory: create a ZipformerEngine from config dict.

        Reads:
          - config["model_dir"] (optional, defaults to ~/.shiyu/models/zipformer-ctc-zh)
          - config["num_threads"] (optional, defaults to 4)
        """
        from pathlib import Path
        model_dir = config.get("model_dir") or "~/.shiyu/models/zipformer-ctc-zh"
        model_dir = str(Path(os.path.expanduser(model_dir)).resolve())
        num_threads = int(config.get("num_threads", 4))
        return cls(model_dir=model_dir, num_threads=num_threads)

    def _ensure_loaded(self):
        """Lazily create sherpa-onnx OfflineRecognizer."""
        if self._recognizer is not None:
            return
        import sherpa_onnx
        model_path = os.path.join(self._model_dir, "model.int8.onnx")
        tokens_path = os.path.join(self._model_dir, "tokens.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Zipformer model not found: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Zipformer tokens not found: {tokens_path}")
        # provider='cpu' 默认; 后续 pipeline 可扩展 cuda/coreml
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=model_path,
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

        sherpa-onnx 1.13+ 没有 read_audio; 我们用 soundfile 读 wav, 自己重采样到 16k.

        长音频 (主人 SRT 案例 9 分钟) 会爆内存 — 因为 accept_waveform 一次性
        喂整段 34MB numpy 给 ONNX. 修复: 分块喂, 每块最多 30 秒 (480k samples).
        """
        import sherpa_onnx
        import soundfile as sf
        import numpy as np

        self._ensure_loaded()

        # 读取音频 (soundfile 自动处理多格式)
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # 重采样到 16k (训练采样率)
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except Exception as e:
                raise RuntimeError(
                    f"无法重采样 {sample_rate}Hz → 16kHz (需要 librosa): {e}"
                )

        # 空音频直接返回
        if len(audio) == 0:
            return TranscribeResult(segments=[], language="zh")

        # 长音频 → 分块 + 2s 上下文重叠 + 去重拼接
        # v2.2 修复 (P2): 块边界 2s 重叠上下文, 避免句子在块边界被截断;
        # 拼接时维护 covered_end, 丢弃与已覆盖区间重叠的重复识别段
        CHUNK_SIZE = 30 * 16000  # 480k samples = 30s
        OVERLAP_SECONDS = 2.0
        DEDUP_TOLERANCE = 0.05  # 秒, 与已覆盖区间重叠超过该值即丢弃
        total_samples = len(audio)

        # 单段足够短 → 走原路径
        if total_samples <= CHUNK_SIZE:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            self._recognizer.decode_stream(stream)
            result = stream.result
            segments = self._result_to_segments(result)
            # v2.0 顺序: 先 ITN+清<unk>+细切, 再 VAD 校准子段
            segments = self._post_process(segments)
            segments = self._align_with_vad(audio, sample_rate, segments)
            return TranscribeResult(
                segments=segments,
                language="zh",
            )

        all_segments = []
        covered_end = 0.0
        for chunk_start in range(0, total_samples, CHUNK_SIZE):
            # 第二块起回退 2s 上下文 (让块边界处的句子有完整解码上下文)
            start_idx = max(0, chunk_start - int(OVERLAP_SECONDS * sample_rate))
            chunk_end = min(chunk_start + CHUNK_SIZE, total_samples)
            chunk = audio[start_idx:chunk_end]
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, chunk)
            self._recognizer.decode_stream(stream)
            result = stream.result

            # 段的时间戳是 chunk 内的相对时间, 加上偏移变绝对时间.
            # 注意: chunk 实际从 start_idx 开始 (第二块起回退 2s 上下文),
            # 偏移必须以 start_idx 为基准; 若按 chunk_start 算会整体 +2s 错位
            time_offset = start_idx / sample_rate
            chunk_segments = self._result_to_segments(result, time_offset=time_offset)
            for seg in chunk_segments:
                if seg.start >= covered_end - DEDUP_TOLERANCE:
                    all_segments.append(seg)
                    covered_end = max(covered_end, seg.end)
                elif seg.end > covered_end:
                    # 跨边界段: 裁剪起点到 covered_end, 保留句尾内容, 避免丢字
                    seg.start = covered_end
                    all_segments.append(seg)
                    covered_end = max(covered_end, seg.end)

            # 立即释放 stream
            del stream

        all_segments = self._post_process(all_segments)
        all_segments = self._align_with_vad(audio, sample_rate, all_segments)
        return TranscribeResult(
            segments=all_segments,
            language="zh",
        )

    @staticmethod
    def _align_with_vad(
        audio,
        sample_rate: int,
        segments: List[Segment],
        boundary_tolerance: float = 1.0,
    ) -> List[Segment]:
        """用 Silero-VAD speech interval 校准 Zipformer CTC 时间戳.

        v2.0 改进: 主人 9 分钟 Godot 教程几乎全程说话, VAD 段间只有 1-1.5s
        静音, 旧版 "找最近 VAD 边界" 会把段起点推到 speech 结束 (晚 0.5s).

        v2.2 修复 (P0): 旧版把落在 interval 内部的段整体替换为 interval
        起点/终点, 连续讲话时一个 interval 覆盖多个段, 导致所有段时间戳
        完全相同 (复现: 5 段全部 0.0-30.0). 新版统一为 "最近边界 + 容差吸附":
        段起点找最近的 interval 起点, 段终点找最近的 interval 终点, 距离
        <= boundary_tolerance 才吸附, 否则保留 Zipformer 原始值;
        校准后再强制相邻段单调 (后段起点不早于前段终点).

        Args:
            audio: 16kHz mono float32 numpy.
            sample_rate: 通常 16000.
            segments: Zipformer 输出的 segments.
            boundary_tolerance: 与最近 VAD 边界的最大距离 (秒), 超过保留原值.

        Returns:
            时间戳校准后的 segments.
        """
        if not segments or len(audio) == 0:
            return segments
        try:
            from .vad_detector import VADDetector
            detector = VADDetector()
            speech_intervals = detector.detect_speech(
                audio, sample_rate,
                min_speech_duration=0.1,
                min_silence_duration=0.25,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "[Zipformer] VAD alignment skipped: %s", e,
            )
            return segments

        if not speech_intervals:
            return segments

        def _snap_to_interval_start(t: float) -> float:
            """段起点: 吸附到最近的 interval 起点, 距离超过容差则保留原值.

            v2.2 修复 (P0): 不再对 interval 内部的 t 整体替换为 interval
            起点, 只做最近起点吸附 + 容差判断.
            """
            best = min(speech_intervals, key=lambda iv: abs(iv[0] - t))
            return best[0] if abs(best[0] - t) <= boundary_tolerance else t

        def _snap_to_interval_end(t: float) -> float:
            """段终点: 吸附到最近的 interval 终点, 距离超过容差则保留原值."""
            best = min(speech_intervals, key=lambda iv: abs(iv[1] - t))
            return best[1] if abs(best[1] - t) <= boundary_tolerance else t

        aligned: List[Segment] = []
        for s in segments:
            new_start = _snap_to_interval_start(s.start)
            new_end = _snap_to_interval_end(s.end)
            if new_end <= new_start:
                # 吸附后区间非法 (起点 >= 终点), 回退到原始值
                new_start, new_end = s.start, s.end
            # 强制单调: 后段起点不能早于前段终点 (吸附可能造成时间戳回退)
            if aligned and new_start < aligned[-1].end:
                new_start = aligned[-1].end
            if new_end < new_start:
                new_end = new_start
            if new_end - new_start < 0.1 and aligned:
                # 吸附+单调钳位产生的极窄段: 文本并入前段, 避免 0 时长字幕事件
                prev = aligned[-1]
                aligned[-1] = Segment(
                    start=prev.start,
                    end=new_end,
                    text=prev.text + s.text,
                )
                continue
            aligned.append(
                Segment(start=new_start, end=new_end, text=s.text)
            )
        return aligned

    @staticmethod
    def _post_process(segments: List[Segment]) -> List[Segment]:
        """后处理 Zipformer 输出 (Shiyu v2.0 流水线):
        1) ITN: 中文数字 → 阿拉伯数字 (九百八十 → 980)
        2) 清 <unk>: Zipformer CTC 在训练集外字符回退符号, 不在最终字幕中显示
        3) 按标点+长度细切: 一段 29s → 多段 5-7s, 改善可读性
           (Zipformer 自带 VAD 切分粗, 主人 9 分钟 Godot 视频 18 段 27-29s)
        """
        try:
            from .itn import itn_normalize
        except ImportError:
            itn_normalize = None

        # Step 1+2: 字符级后处理 (ITN + 清 <unk>)
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

        # Step 3: 按标点+长度细切 (text_splitter 按 char-count 比例分配时间戳)
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
            # 后处理失败, 返回 cleaned (主人 9 分钟测试时不会崩)
            import logging
            logging.getLogger(__name__).debug(
                "[Zipformer] split_text_at_boundaries failed, returning as-is: %s", e,
            )
            return cleaned

    def _result_to_segments(self, result, time_offset: float = 0.0) -> List[Segment]:
        """把 sherpa-onnx OfflineRecognitionResult 转成 List[Segment].

        sherpa-onnx 可能已经在内部做了 VAD 切分 (result.segment_texts 非空),
        也可能只给整段 (segment_texts 为空) — 两种情况都要处理.

        时间戳来源:
          - VAD 切分时: segment_timestamps[i] + segment_durations[i]
          - 整段时: timestamps[0] ~ timestamps[-1] 覆盖整个解码区间
        """
        # 情况 1: sherpa-onnx 已 VAD 切分 (短静音会自动分段)
        # 用 isinstance 兜底 MagicMock 测试和异常返回值
        seg_texts = getattr(result, "segment_texts", None)
        seg_ts = getattr(result, "segment_timestamps", None)
        seg_dur = getattr(result, "segment_durations", None)
        if (
            isinstance(seg_texts, list)
            and isinstance(seg_ts, list)
            and len(seg_texts) == len(seg_ts)
            and len(seg_texts) > 0
        ):
            segments: List[Segment] = []
            for i, t in enumerate(seg_texts):
                t = (t or "").strip()
                if not t:
                    continue
                start = float(seg_ts[i]) + time_offset
                dur = float(seg_dur[i]) if isinstance(seg_dur, list) and i < len(seg_dur) else 0.0
                segments.append(Segment(start=round(start, 3), end=round(start + dur, 3), text=t))
            if segments:
                return segments

        # 情况 2: 整段输出, 用 timestamps 推算首尾
        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return []
        timestamps = getattr(result, "timestamps", None) or []
        if not isinstance(timestamps, list) or not timestamps:
            return [Segment(start=time_offset, end=time_offset, text=text)]
        start_sec = float(timestamps[0]) + time_offset
        end_sec = float(timestamps[-1]) + time_offset
        return [Segment(start=round(start_sec, 3), end=round(end_sec, 3), text=text)]

    def is_available(self) -> Tuple[bool, str]:
        """检查 sherpa-onnx + 模型文件是否齐备."""
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
        # 真实 int8 模型约 350MB; < 100MB 视为不完整 (允许测试 stub 通过需另外 mock)
        if not model_file.exists() or model_file.stat().st_size < 100_000_000:
            return False, f"model.int8.onnx missing or too small: {model_file}"
        if not tokens_file.exists() or tokens_file.stat().st_size < 100:
            return False, f"tokens.txt missing or empty: {tokens_file}"
        return True, ""


# ── BaseTranscriber Adapter ────────────────────────────────────────

class ZipformerTranscriber(BaseTranscriber):
    """Adapter wrapping ZipformerEngine as a BaseTranscriber."""

    name = "zipformer_local"
    display_name = "Zipformer CTC (中文, sherpa-onnx)"
    requires_network = False
    is_local = True

    def __init__(self, model_dir: str, num_threads: int = 4):
        self._engine = ZipformerEngine(model_dir, num_threads)

    @classmethod
    def from_config(cls, config: dict) -> "ZipformerTranscriber":
        kwargs = ZipformerEngine.init_kwargs_from_config(config)
        return cls(**kwargs)

    def transcribe(self, audio_path: str, language: str = "auto", **kwargs) -> TranscribeResult:
        return self._engine.transcribe(audio_path, language, **kwargs)

    def is_available(self) -> Tuple[bool, str]:
        return self._engine.is_available()


# 给 ZipformerEngine 加 init_kwargs_from_config 类方法
@classmethod
def init_kwargs_from_config(cls, config: dict) -> dict:
    import os
    from pathlib import Path
    model_dir = config.get("model_dir") or "~/.shiyu/models/zipformer-ctc-zh"
    model_dir = str(Path(os.path.expanduser(model_dir)).resolve())
    num_threads = config.get("num_threads", 4)
    return {"model_dir": model_dir, "num_threads": num_threads}


ZipformerEngine.init_kwargs_from_config = init_kwargs_from_config