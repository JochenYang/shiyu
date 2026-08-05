"""
Unified ASR engine interface.
All transcriber engines implement BaseTranscriber,
enabling drop-in replacement in main.py and the frontend.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Segment:
    """Single subtitle segment with timing."""
    start: float  # seconds
    end: float    # seconds
    text: str
    # 可选: 逐字符开始时间 (秒, len == len(text)); 仅 zipformer 整段输出时
    # 由 sherpa-onnx 逐 token 时间戳重建, 供细切时精确分配, 其他引擎为 None
    # 注意: 仅为内部中间态 (zipformer _result_to_segments → _post_process),
    # to_dict() 不含该字段, 不下发到 API/前端
    char_times: List[float] = None  # type: ignore

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class TranscribeResult:
    """Result from any ASR engine."""
    segments: List[Segment] = field(default_factory=list)
    language: str = "auto"

    def to_segments_list(self) -> List[dict]:
        return [s.to_dict() for s in self.segments]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text.strip())


class BaseTranscriber(ABC):
    """Abstract base for all ASR engines.

    Each engine:
      - Must implement transcribe() returning TranscribeResult
      - Should set name and display_name as class-level metadata
      - Should handle its own config/API keys internally
    """

    # Engine identifier (used in config, API routing)
    name: str = "base"
    # Human-readable name for frontend dropdown
    display_name: str = "Base Engine"
    # Whether this engine requires API key / internet
    requires_network: bool = False
    # Whether this engine runs locally
    is_local: bool = True

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        **kwargs,
    ) -> TranscribeResult:
        """Transcribe audio file and return segments with timestamps.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language hint ("auto", "zh", "en", "ja", etc.)
            **kwargs: Engine-specific options

        Returns:
            TranscribeResult with typed segments
        """
        ...

    def load_config(self) -> dict:
        """Load engine-specific config. Override if needed."""
        return {}

    def is_available(self) -> Tuple[bool, str]:
        """Check if this engine is ready to use.

        Returns:
            (available, reason_if_unavailable)
        """
        return True, ""

    def batch_transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        **kwargs,
    ) -> TranscribeResult:
        """Default batch implementation; calls transcribe() once.
        Override for engines that handle batching internally.
        """
        return self.transcribe(audio_path, language, **kwargs)
