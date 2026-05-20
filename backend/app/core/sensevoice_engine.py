"""
SenseVoice ONNX Inference Engine.
Wraps model loading, token decoding, and CTC-based transcription.

Key design: CTC frame-level timestamps for accurate subtitle timing.
Each decoded token carries the frame index where it was emitted, which
maps directly to real audio time via:  frame_idx * lfr_n * frame_shift / 1000
"""
import json
import re
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional


class SenseVoiceEngine:
    """ONNX runtime wrapper for SenseVoice-Small model."""
    
    # Special tokens to strip from output
    RICH_TAGS = re.compile(r"<\|[^|]+\|>")
    
    # Language tokens mapping (auto uses zh as default for SenseVoice)
    LANG_MAP = {
        "<|zh|>": "zh", "<|en|>": "en", "<|ja|>": "ja",
        "<|ko|>": "ko", "<|yue|>": "yue", "<|nospeech|>": "nospeech"
    }
    # Language string to integer ID (from SenseVoice ONNX official export)
    LANG_ID_MAP = {
        "auto": 0, "zh": 3, "en": 4,
        "yue": 7, "ja": 11, "ko": 12, "nospeech": 13
    }
    # Text normalization integer IDs (from SenseVoice ONNX official export)
    TEXTNORM_ID_MAP = {"withitn": 14, "woitn": 15}

    # LFR / frame timing constants (must match audio_processor defaults)
    LFR_N = 6           # Low Frame Rate skip factor
    FRAME_SHIFT_MS = 10  # Original fbank frame shift in ms
    # Each CTC frame = LFR_N * FRAME_SHIFT_MS = 60ms
    FRAME_TO_SEC = LFR_N * FRAME_SHIFT_MS / 1000.0  # 0.06
    
    def __init__(self, model_path: str, tokens_path: str, device: str = "cpu"):
        """Initialize ONNX session.
        
        Args:
            model_path: Path to model.onnx
            tokens_path: Path to tokens.json
            device: "cpu" or "cuda"
        """
        self.model_path = model_path
        self.tokens_path = tokens_path
        
        # Load tokens
        with open(tokens_path, "r", encoding="utf-8") as f:
            self.tokens = json.load(f)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}
        
        # Setup ONNX session
        providers = ["CUDAExecutionProvider"] if device == "cuda" and self._has_cuda() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def _has_cuda(self) -> bool:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    
    def _ctc_greedy_decode(self, logits: np.ndarray, input_length: int) -> List[Tuple[int, int]]:
        """CTC greedy decode: argmax + deduplicate blanks.
        
        Args:
            logits: [T, V] softmax probabilities or raw logits
            input_length: Actual frame count before padding
        
        Returns:
            List of (token_id, frame_index) tuples.
            frame_index is the CTC frame where the token was first emitted.
        """
        preds = np.argmax(logits[:input_length], axis=-1)
        # Deduplicate: keep first occurrence of each consecutive token
        decoded = []
        prev = -1
        for frame_idx, p in enumerate(preds):
            if p != prev and p != 0:  # 0 is blank token
                decoded.append((int(p), frame_idx))
            prev = p
        return decoded
    
    def _ids_to_text(self, id_frame_pairs: List[Tuple[int, int]], language: str = "auto") -> Tuple[str, str]:
        """Convert token IDs to text, strip special tags.
        
        Accepts (token_id, frame_index) tuples from _ctc_greedy_decode.
        
        Returns:
            (clean_text, detected_language)
        """
        token_ids = [pair[0] for pair in id_frame_pairs]
        tokens = [self.id_to_token.get(i, "") for i in token_ids]
        text = "".join(tokens).replace("▁", " ").strip()
        
        # Detect language from tags
        detected_lang = "auto"
        for tag, lang in self.LANG_MAP.items():
            if tag in text:
                detected_lang = lang
                break
        
        # Strip all rich tags
        clean = self.RICH_TAGS.sub("", text).strip()
        # Remove SenseVoice newline markers (\N)
        clean = clean.replace("\\N", " ")
        # Collapse multiple spaces
        clean = re.sub(r"\s+", " ", clean)
        return clean, detected_lang

    def _decode_with_frames(self, id_frame_pairs: List[Tuple[int, int]]) -> Tuple[str, str, List[int]]:
        """Convert CTC output to clean text with per-character frame positions.
        
        Each character in the returned text has a corresponding frame index
        in char_frames, enabling accurate timestamp calculation.
        
        Returns:
            (clean_text, detected_language, char_frames)
            char_frames[i] = CTC frame index for clean_text[i]
        """
        char_data = []  # [(char, frame_idx), ...]
        detected_lang = "auto"
        
        for token_id, frame_idx in id_frame_pairs:
            token_str = self.id_to_token.get(token_id, "")
            
            # Step 1: Detect and skip language tags
            if token_str in self.LANG_MAP:
                detected_lang = self.LANG_MAP[token_str]
                continue
            
            # Step 2: Skip all rich tags <|...|>
            if self.RICH_TAGS.fullmatch(token_str):
                continue
            
            # Step 3: Normalize token text
            token_str = token_str.replace("▁", " ").replace("\\N", " ")
            
            for c in token_str:
                char_data.append((c, frame_idx))
        
        # Step 4: Collapse consecutive spaces
        collapsed = []
        prev_space = False
        for c, f in char_data:
            if c == " ":
                if prev_space:
                    continue
                prev_space = True
            else:
                prev_space = False
            collapsed.append((c, f))
        
        # Step 5: Strip leading/trailing spaces
        while collapsed and collapsed[0][0] == " ":
            collapsed.pop(0)
        while collapsed and collapsed[-1][0] == " ":
            collapsed.pop()
        
        text = "".join(c for c, _ in collapsed)
        frames = [f for _, f in collapsed]
        
        return text, detected_lang, frames
    
    def _build_inputs(self, features: np.ndarray, language: str = "auto",
                      use_itn: bool = True) -> dict:
        """Build ONNX input dict from features and options.
        
        Shared by transcribe() and transcribe_with_timestamps().
        """
        inputs = {self.input_names[0]: features.astype(np.float32)}
        
        for name in self.input_names[1:]:
            if "length" in name.lower():
                inputs[name] = np.array([features.shape[1]], dtype=np.int32)
            elif "language" in name.lower():
                lang_id = self.LANG_ID_MAP.get(language, 0)
                inputs[name] = np.array([lang_id], dtype=np.int32)
            elif "textnorm" in name.lower():
                tn_key = "withitn" if use_itn else "woitn"
                tn_id = self.TEXTNORM_ID_MAP.get(tn_key, 15)
                inputs[name] = np.array([tn_id], dtype=np.int32)
            else:
                inputs[name] = np.zeros((1,), dtype=np.int32)
        
        return inputs

    def transcribe(self, features: np.ndarray, 
                   language: str = "auto",
                   use_itn: bool = True) -> dict:
        """Run inference on preprocessed features.
        
        Args:
            features: [1, T, D] float32 from audio_processor
            language: Target language hint (auto/zh/en/ja/ko/yue)
            use_itn: Whether to apply inverse text normalization
        
        Returns:
            dict with keys: text, language, duration_frames
        """
        inputs = self._build_inputs(features, language, use_itn)
        outputs = self.session.run(self.output_names, inputs)
        logits = outputs[0][0]  # [T, V]
        
        decoded_pairs = self._ctc_greedy_decode(logits, features.shape[1])
        text, detected_lang = self._ids_to_text(decoded_pairs, language)
        
        return {
            "text": text,
            "language": detected_lang,
            "duration_frames": features.shape[1]
        }
    
    # CTC peak delay compensation (CTC emits tokens after seeing context)
    CTC_DELAY_SEC = 0.15

    @staticmethod
    def _is_mostly_cjk(text: str) -> bool:
        """Check if text is primarily CJK (Chinese/Japanese/Korean)."""
        cjk = sum(1 for c in text if (
            '\u4e00' <= c <= '\u9fff' or   # CJK Unified
            '\u3040' <= c <= '\u30ff' or   # Hiragana + Katakana
            '\uac00' <= c <= '\ud7af'      # Hangul
        ))
        return cjk > len(text) * 0.3

    def transcribe_with_timestamps(self, features: np.ndarray,
                                    audio_duration_sec: float,
                                    language: str = "auto") -> List[dict]:
        """Transcribe with CTC frame-level timestamps.
        
        Uses actual CTC frame emission positions for accurate segment timing.
        Compensates for CTC peak delay (~150ms) by shifting start times earlier.
        Detects text language to use word-aware splitting for English.
        """
        # Step 1: Run inference
        inputs = self._build_inputs(features, language, use_itn=True)
        outputs = self.session.run(self.output_names, inputs)
        logits = outputs[0][0]  # [T, V]
        
        # Step 2: CTC decode with frame positions
        id_frame_pairs = self._ctc_greedy_decode(logits, features.shape[1])
        text, detected_lang, char_frames = self._decode_with_frames(id_frame_pairs)
        
        if not text or not char_frames:
            return []
        
        # Step 3: Detect language and choose splitting parameters
        is_cjk = self._is_mostly_cjk(text)
        max_len = 25 if is_cjk else 42  # English allows more chars per line
        
        # Step 4: Split text into subtitle segments
        segments_text = self._split_text_for_subtitles(text, max_len, is_cjk)
        
        if not segments_text:
            return []
        
        # Step 5: Map each segment to real timestamps via char_frames
        # Uses sequential search in remaining text to avoid position errors
        segments = []
        remaining = text
        consumed = 0  # Total characters consumed from original text
        
        for seg_text in segments_text:
            pos = remaining.find(seg_text)
            if pos == -1:
                # Fallback: assume immediately after last segment
                seg_start_idx = consumed
            else:
                seg_start_idx = consumed + pos
            
            seg_end_idx = seg_start_idx + len(seg_text) - 1
            
            # Clamp to valid char_frames range
            seg_start_idx = min(seg_start_idx, len(char_frames) - 1)
            seg_end_idx = min(seg_end_idx, len(char_frames) - 1)
            
            # Convert frame indices to seconds with CTC delay compensation
            start_sec = max(0, char_frames[seg_start_idx] * self.FRAME_TO_SEC - self.CTC_DELAY_SEC)
            end_sec = char_frames[seg_end_idx] * self.FRAME_TO_SEC
            
            # Ensure minimum segment duration (0.5s) for readability
            if end_sec - start_sec < 0.5:
                end_sec = start_sec + 0.5
            
            segments.append({
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "text": seg_text
            })
            
            # Advance past this segment in remaining text
            advance = (pos + len(seg_text)) if pos != -1 else len(seg_text)
            remaining = remaining[advance:]
            consumed += advance
        
        # Step 6: Fix segment boundaries
        for i in range(len(segments) - 1):
            if segments[i]["end"] < segments[i + 1]["start"]:
                segments[i]["end"] = segments[i + 1]["start"]
        
        if segments:
            segments[-1]["end"] = min(
                max(segments[-1]["end"], segments[-1]["start"] + 0.5),
                audio_duration_sec
            )
        
        return segments

    def _split_text_for_subtitles(self, text: str, max_len: int = 25,
                                   is_cjk: bool = True) -> List[str]:
        """Split text into subtitle-friendly segments using 3-level strategy.
        
        Level 1: Split by sentence boundaries (。！？；!?;  .!? for English)
        Level 2: Split long sentences at commas/colons
        Level 3: Hard-cut fallback (word-aware for English, char-level for CJK)
        """
        # Level 1: Sentence boundary split
        if is_cjk:
            parts = re.split(r"([。！？；!?;.])", text)
        else:
            # For English: split at sentence-ending punctuation followed by space
            parts = re.split(r"([.!?;](?:\s|$))", text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            s = s.strip()
            if s:
                sentences.append(s)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        # Level 2: Comma/colon split for long segments
        refined = []
        for sent in sentences:
            if len(sent) <= max_len:
                refined.append(sent)
                continue
            if is_cjk:
                sub_parts = re.split(r"([，、,：:、])", sent)
            else:
                # For English: split at commas
                sub_parts = re.split(r"(,\s*)", sent)
            current = ""
            for j in range(0, len(sub_parts) - 1, 2):
                clause = sub_parts[j] + (sub_parts[j + 1] if j + 1 < len(sub_parts) else "")
                if not current:
                    current = clause
                elif len(current) + len(clause) <= max_len:
                    current += clause
                else:
                    if current.strip():
                        refined.append(current.strip())
                    current = clause
            if len(sub_parts) % 2 == 1 and sub_parts[-1]:
                if current and len(current) + len(sub_parts[-1]) <= max_len:
                    current += sub_parts[-1]
                else:
                    if current.strip():
                        refined.append(current.strip())
                    current = sub_parts[-1]
            if current.strip():
                refined.append(current.strip())

        # Level 3: Hard-cut fallback
        final = []
        for seg in refined:
            if len(seg) <= max_len:
                final.append(seg)
                continue
            
            if is_cjk:
                # CJK: character-level cut with punctuation attachment
                i = 0
                while i < len(seg):
                    end = min(i + max_len, len(seg))
                    chunk = seg[i:end]
                    if end < len(seg) and seg[end] in "，、,：:。！？；!?;":
                        chunk += seg[end]
                        end += 1
                    final.append(chunk.strip())
                    i = end
            else:
                # English: word-boundary-aware splitting
                words = seg.split(" ")
                current = ""
                for word in words:
                    if not word:
                        continue
                    if not current:
                        current = word
                    elif len(current) + 1 + len(word) <= max_len:
                        current += " " + word
                    else:
                        if current.strip():
                            final.append(current.strip())
                        current = word
                if current.strip():
                    final.append(current.strip())

        # Fallback: if no splitting happened
        if not final:
            final = [text.strip()] if text.strip() else []

        return final

    def transcribe_waveform_with_vad(self, waveform: np.ndarray, sample_rate: int, audio_cfg: dict,
                                     duration_sec: float, language: str = "zh",
                                     custom_glossary: dict = None) -> List[dict]:
        """Process long audio by splitting into safe chunks using energy-based VAD.
        
        Features:
        - Energy VAD to find silence boundaries
        - Merging short segments into max 25-second chunks
        - Context padding
        - Glossary mapping for domain-specific terminology correction
        """
        import librosa
        from app.core.audio_processor import compute_fbank_kaldi, apply_lfr, apply_cmvn, load_cmvn
        import os
        
        # 1. 能量 VAD 寻找静音切分点 (而不是过滤掉)
        intervals = librosa.effects.split(waveform, top_db=40, frame_length=2048, hop_length=512)
        
        n_mels = audio_cfg.get("n_mels", 80)
        frame_length = audio_cfg.get("frame_length", 25)
        frame_shift = audio_cfg.get("frame_shift", 10)
        lfr_m = audio_cfg.get("lfr_m", 7)
        lfr_n = audio_cfg.get("lfr_n", 6)
        cmvn_file = audio_cfg.get("cmvn_file", None)
        
        cmvn = load_cmvn(cmvn_file) if cmvn_file and os.path.exists(cmvn_file) else None
        
        # 术语纠错表 Glossary
        base_glossary = {
            "labelbel": "Label",
            "go it": "Godot",
            "goold it": "Godot",
            "good it": "Godot",
            "goold": "Godot",
            "个到特": "Godot",
            "高到特": "Godot",
            "狗道特": "Godot",
            "狗到特": "Godot",
            "c家家": "C++",
            "思家家": "C++",
            "c加加": "C++",
            "拍森": "Python",
            "泰普斯克里普特": "TypeScript",
            "加瓦": "Java",
            "瑞阿克特": "React",
            "微优伊": "Vue",
            "维优伊": "Vue",
            "尤尼提": "Unity",
            "昂瑞尔": "Unreal"
        }
        
        if custom_glossary:
            base_glossary.update(custom_glossary)
            
        glossary = base_glossary
        
        # 2. 将 VAD 区间转换为静音缝隙 (Silences)
        silences = []
        last_end = 0
        for start, end in intervals:
            if start > last_end:
                silences.append((last_end, start))
            last_end = end
        if last_end < len(waveform):
            silences.append((last_end, len(waveform)))
            
        # 3. 100% 覆盖切割法：只在静音中点下刀，绝不丢弃任何样本
        chunks = []
        current_start = 0
        min_chunk_sec = 8.0   # 最小切片长度，低于这个不切
        max_chunk_sec = 28.0  # 极限长度（备用）
        
        for sil_start, sil_end in silences:
            cut_point = (sil_start + sil_end) // 2
            
            # 如果累积到切分点已经超过推荐长度，就切一刀
            if (cut_point - current_start) / sample_rate > min_chunk_sec:
                chunks.append((current_start, cut_point))
                current_start = cut_point
                
        # 兜底：处理剩余部分
        if current_start < len(waveform):
            # 如果尾巴实在太短，直接融合给上一段
            if chunks and (len(waveform) - current_start) / sample_rate < 2.0:
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, len(waveform)))
            else:
                chunks.append((current_start, len(waveform)))
                
        if not chunks:
            chunks = [(0, len(waveform))]

        all_segments = []
        
        for start_idx, end_idx in chunks:
            # 因为已经是 100% 无缝切割（且切点在静音中），不再需要激进的 padding
            # 加极小的 padding (50ms) 以免爆音边界
            pad = int(sample_rate * 0.05)
            s_idx = max(0, start_idx - pad)
            e_idx = min(len(waveform), end_idx + pad)
            
            chunk_wave = waveform[s_idx:e_idx]
            chunk_duration = len(chunk_wave) / sample_rate
            
            fbank = compute_fbank_kaldi(chunk_wave, sample_rate, n_mels, frame_length, frame_shift)
            features = apply_lfr(fbank, lfr_m, lfr_n)
            if cmvn is not None:
                features = apply_cmvn(features, cmvn).astype(np.float32)
            features = np.expand_dims(features, axis=0)
            
            # 对单个分段强制进行高质量推理
            chunk_segments = self.transcribe_with_timestamps(features, chunk_duration, language)
            
            chunk_start_sec = s_idx / sample_rate
            for seg in chunk_segments:
                # 合并时间戳
                seg["start"] = round(seg["start"] + chunk_start_sec, 3)
                seg["end"] = round(seg["end"] + chunk_start_sec, 3)
                
                # 术语纠错后处理
                text = seg["text"]
                for bad, good in glossary.items():
                    text = text.replace(bad, good)
                seg["text"] = text
                
                all_segments.append(seg)
                
        return all_segments