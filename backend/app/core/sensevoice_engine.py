"""
SenseVoice ONNX Inference Engine.
Wraps model loading, token decoding, and CTC-based transcription.
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
    
    def _ctc_greedy_decode(self, logits: np.ndarray, input_length: int) -> List[int]:
        """CTC greedy decode: argmax + deduplicate blanks.
        
        Args:
            logits: [T, V] softmax probabilities or raw logits
            input_length: Actual frame count before padding
        
        Returns:
            List of token IDs
        """
        preds = np.argmax(logits[:input_length], axis=-1)
        # Deduplicate
        decoded = []
        prev = -1
        for p in preds:
            if p != prev and p != 0:  # 0 is blank token
                decoded.append(int(p))
            prev = p
        return decoded
    
    def _ids_to_text(self, ids: List[int], language: str = "auto") -> Tuple[str, str]:
        """Convert token IDs to text, strip special tags.
        
        Returns:
            (clean_text, detected_language)
        """
        tokens = [self.id_to_token.get(i, "") for i in ids]
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
        # Build input dict matching model signature: speech, speech_lengths, language, textnorm
        inputs = {self.input_names[0]: features.astype(np.float32)}
        
        for name in self.input_names[1:]:
            if "length" in name.lower():
                # speech_lengths: int32, actual feature frame count
                inputs[name] = np.array([features.shape[1]], dtype=np.int32)
            elif "language" in name.lower():
                # language: int32, mapped from SenseVoice official export
                lang_id = self.LANG_ID_MAP.get(language, 0)
                inputs[name] = np.array([lang_id], dtype=np.int32)
            elif "textnorm" in name.lower():
                # textnorm: int32, withitn=14 / woitn=15
                tn_key = "withitn" if use_itn else "woitn"
                tn_id = self.TEXTNORM_ID_MAP.get(tn_key, 15)
                inputs[name] = np.array([tn_id], dtype=np.int32)
            else:
                inputs[name] = np.zeros((1,), dtype=np.int32)
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Assume first output is logits [1, T, V]
        logits = outputs[0][0]  # [T, V]
        
        # Decode
        decoded_ids = self._ctc_greedy_decode(logits, features.shape[1])
        text, detected_lang = self._ids_to_text(decoded_ids, language)
        
        return {
            "text": text,
            "language": detected_lang,
            "duration_frames": features.shape[1]
        }
    
    def transcribe_with_timestamps(self, features: np.ndarray,
                                    audio_duration_sec: float,
                                    language: str = "auto") -> List[dict]:
        """Transcribe with sentence-level timestamps.
        
        Splits text by sentence/clause boundaries (periods, commas, etc.)
        and estimates timestamps proportional to character count.
        """
        result = self.transcribe(features, language)
        text = result["text"]

        # Split by sentence and clause boundaries for subtitle-friendly segments
        # Each segment should be roughly one clause/sentence for editing convenience
        parts = re.split(r"([。！？；!?;])", text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            s = s.strip()
            if s:
                sentences.append(s)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        # Further split long segments at comma boundaries
        # A single subtitle longer than ~25 chars is hard to read
        refined = []
        for sent in sentences:
            if len(sent) <= 25:
                refined.append(sent)
                continue
            # Split at commas/顿号/冒号
            sub_parts = re.split(r"([，、,：:、])", sent)
            current = ""
            for j in range(0, len(sub_parts) - 1, 2):
                clause = sub_parts[j] + (sub_parts[j + 1] if j + 1 < len(sub_parts) else "")
                if not current:
                    current = clause
                elif len(current) + len(clause) <= 25:
                    current += clause
                else:
                    refined.append(current.strip())
                    current = clause
            if len(sub_parts) % 2 == 1 and sub_parts[-1]:
                if current and len(current) + len(sub_parts[-1]) <= 25:
                    current += sub_parts[-1]
                else:
                    if current.strip():
                        refined.append(current.strip())
                    current = sub_parts[-1]
            if current.strip():
                refined.append(current.strip())

        # Step 3: Hard-cut fallback for segments still > max_len
        # (long sentences with no comma/colon to split on)
        final = []
        for seg in refined:
            if len(seg) <= 25:
                final.append(seg)
                continue
            # Hard-cut at max_len boundary, attach trailing punctuation to prev chunk
            i = 0
            while i < len(seg):
                end = min(i + 25, len(seg))
                chunk = seg[i:end]
                # If next chunk starts with punctuation, pull it into current chunk
                if end < len(seg) and seg[end] in "，、,：:。！？；!?;":
                    chunk += seg[end]
                    end += 1
                final.append(chunk.strip())
                i = end

        # Fallback: if no splitting happened, use whole text
        if not final:
            final = [text.strip()] if text.strip() else []

        segments = []
        current_time = 0.0
        total_chars = sum(len(s) for s in final)

        for seg_text in final:
            ratio = len(seg_text) / max(total_chars, 1)
            seg_duration = audio_duration_sec * ratio
            segments.append({
                "start": current_time,
                "end": current_time + seg_duration,
                "text": seg_text
            })
            current_time += seg_duration

        return segments