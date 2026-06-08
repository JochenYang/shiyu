"""
Cloud transcription engine supporting both:
1. Standard OpenAI Whisper API (POST /audio/transcriptions, multipart file upload)
   - Providers: OpenAI, Groq, DeepSeek, etc.
   - Returns native segment timestamps (verbose_json)
2. DashScope compatible mode (POST /chat/completions, input_audio base64)
   - Provider: DashScope Qwen3-ASR-Flash
   - Returns plain text only; timestamps generated locally via VAD
   - When available, uses SenseVoice CTC forced alignment for accurate timestamps
"""
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

import requests

from .audio_processor import find_ffmpeg
from . import text_splitter

CONFIG_FILE = Path.home() / ".shiyu" / "config.json"


def _friendly_http_error(resp, endpoint: str, mode: str) -> RuntimeError:
    """Build a user-readable error from a non-2xx HTTP response.

    Inspects Content-Type to avoid dumping HTML bodies to the UI:
    - ``text/html`` → likely wrong URL/path/port (gateway 404 page)
    - ``application/json`` → parse the standard error envelope
    - anything else → first 200 chars of the body

    The error message always includes the endpoint so the user can verify
    their ``api_url`` without grep-ing logs.
    """
    ctype = (resp.headers.get("Content-Type") or "").lower()
    body = resp.text or ""
    if "text/html" in ctype:
        snippet = body.split(">", 1)[0].split("<", 1)[0] if body else ""
        msg = (
            f"Cloud API error {resp.status_code} (HTML response, likely wrong URL). "
            f"Endpoint: {endpoint}. Mode: {mode}. "
            f"Check that your api_url is correct and the path is supported by this provider. "
            f"Server hint: {snippet[:120]!r}"
        )
    elif "application/json" in ctype:
        try:
            data = resp.json()
            # Standard OpenAI-compatible error envelope
            err_obj = data.get("error") if isinstance(data, dict) else None
            if isinstance(err_obj, dict):
                inner = err_obj.get("message") or err_obj.get("type") or str(err_obj)
            elif err_obj is not None:
                inner = str(err_obj)
            else:
                inner = str(data)[:200]
            msg = (
                f"Cloud API error {resp.status_code} ({mode}): {inner}. "
                f"Endpoint: {endpoint}"
            )
        except (ValueError, KeyError, TypeError):
            msg = (
                f"Cloud API error {resp.status_code} ({mode}, malformed JSON). "
                f"Endpoint: {endpoint}. Body: {body[:200]!r}"
            )
    else:
        msg = (
            f"Cloud API error {resp.status_code} ({mode}). "
            f"Endpoint: {endpoint}. Content-Type: {ctype!r}. "
            f"Body: {body[:200]!r}"
        )
    return RuntimeError(msg)

# On Windows, curl (Schannel TLS) avoids OpenSSL SSLEOFError with some servers
_USE_CURL = sys.platform == "win32"

# Max base64-encoded payload size for DashScope (conservative: 9.5MB to stay within 10MB limit)
_DASHSCOPE_MAX_B64_BYTES = int(9.5 * 1024 * 1024)


# ── Config ──────────────────────────────────────────────────────────


def load_engine_config() -> Dict[str, Any]:
    """Load cloud engine config from ~/.shiyu/config.json"""
    defaults = {"api_url": "", "api_key": "", "model": "", "enabled": False}
    if not CONFIG_FILE.exists():
        return dict(defaults)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("engine", {})
        result = dict(defaults)
        result.update({k: cfg[k] for k in result if k in cfg})
        return result
    except (json.JSONDecodeError, IOError):
        return dict(defaults)


def save_engine_config(api_url: str, api_key: str, model: str, enabled: bool) -> None:
    """Save cloud engine config to ~/.shiyu/config.json"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["engine"] = {"api_url": api_url, "api_key": api_key, "model": model, "enabled": enabled}
    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(CONFIG_FILE)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass  # Windows may not support chmod


# ── LLM Correction Config ───────────────────────────────────────────


def load_llm_correction_config() -> Dict[str, Any]:
    """从 ~/.shiyu/config.json 加载 LLM 修正配置。

    字段: enabled(bool), provider(str), api_url(str), api_key(str), model(str)
    返回完整 dict（含 api_key 明文）。
    """
    defaults: Dict[str, Any] = {
        "enabled": False,
        "provider": "openai",
        "api_url": "",
        "api_key": "",
        "model": "",
    }
    if not CONFIG_FILE.exists():
        return dict(defaults)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("llm_correction", {})
        result = dict(defaults)
        result.update({k: cfg[k] for k in result if k in cfg})
        return result
    except (json.JSONDecodeError, IOError):
        return dict(defaults)


def load_llm_correction_config_masked() -> Dict[str, Any]:
    """返回 LLM 修正配置（含 api_key 原文）。
    
    注意：api_key 随配置返回，前端可回显。本地配置文件本身明文存储。
    """
    cfg = load_llm_correction_config()
    return {
        "enabled": cfg["enabled"],
        "provider": cfg["provider"],
        "api_url": cfg["api_url"],
        "api_key": cfg["api_key"],
        "has_key": bool(cfg["api_key"]),
        "model": cfg["model"],
    }


def save_llm_correction_config(enabled: bool, provider: str, api_url: str,
                                api_key: str, model: str) -> None:
    """保存 LLM 修正配置到 ~/.shiyu/config.json data["llm_correction"]。

    如果 api_key 为空字符串，保留已有的 key 不覆盖（避免前端掩码值清空密钥）。
    """
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    existing = data.get("llm_correction", {})
    resolved_key = api_key if api_key else existing.get("api_key", "")
    data["llm_correction"] = {
        "enabled": enabled,
        "provider": provider,
        "api_url": api_url,
        "api_key": resolved_key,
        "model": model,
    }
    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(CONFIG_FILE)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


# ── Provider detection ──────────────────────────────────────────────


def _is_dashscope_url(api_url: str) -> bool:
    """Detect if the API URL points to DashScope (legacy / backward-compat).

    Kept for code paths that still special-case DashScope. New code should
    rely on the ``request_mode`` config field instead of URL sniffing.
    """
    return "dashscope.aliyuncs.com" in api_url


def _resolve_auth_header(config: dict) -> dict:
    """Build the HTTP auth header from engine config.

    Default scheme matches OpenAI / OpenAI-compatible providers
    (``Authorization: Bearer <api_key>``). Providers using a different
    auth header can override via ``auth_header.name`` and
    ``auth_header.value_template`` (``{api_key}`` is substituted with
    the configured key).

    Examples:
      # OpenAI / DashScope compatible
      (uses defaults)

      # Xiaomi MiMo:
      auth_header: {"name": "api-key", "value_template": "{api_key}"}
    """
    auth = config.get("auth_header") or {}
    name = auth.get("name", "Authorization")
    template = auth.get("value_template", "Bearer {api_key}")
    value = template.format(api_key=config.get("api_key", ""))
    return {name: value}


def _normalize_base_url(api_url: str) -> str:
    """Normalize API base URL: strip trailing paths and ensure correct base.

    Handles various user input formats:
    - https://dashscope.aliyuncs.com/api/v1              → compatible-mode/v1
    - https://dashscope.aliyuncs.com/compatible-mode/v1   → stay
    - https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions → strip
    """
    url = api_url.rstrip("/")
    # Strip trailing /chat/completions if user entered the full endpoint
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    # Switch from native API path to compatible mode
    if "/api/v1" in url:
        url = url.replace("/api/v1", "/compatible-mode/v1")
    return url.rstrip("/")


def _run_hidden(cmd, **kwargs):
    """Run subprocess without flashing a console window on Windows."""
    opts = dict(kwargs)
    if sys.platform == "win32":
        opts.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    return subprocess.run(cmd, **opts)


def _compress_audio_for_upload(audio_path: str, max_b64_bytes: int = _DASHSCOPE_MAX_B64_BYTES) -> str:
    """Compress audio to mp3 for smaller base64 payload.

    Step 1: Convert to mono 16kHz mp3 at 32kbps (typical 5-10x compression)
    Step 2: If still too large, reduce bitrate to 16kbps
    Step 3: Return path to compressed file (caller must clean up)

    A 60s 16kHz WAV (~1.9MB) compresses to ~240KB mp3 → ~320KB base64.
    """
    ffmpeg_exe = find_ffmpeg()
    compressed = tempfile.mktemp(suffix=".mp3")

    for bitrate in ["32k", "16k"]:
        cmd = [
            ffmpeg_exe, "-y", "-i", audio_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "libmp3lame", "-b:a", bitrate,
            compressed,
        ]
        try:
            _run_hidden(cmd, check=True, capture_output=True, timeout=120)
        except FileNotFoundError:
            # ffmpeg not found; return original and let API handle it
            return audio_path
        except subprocess.CalledProcessError:
            # Codec not available; try next bitrate or return original
            continue

        # Check if compressed file is small enough for base64
        file_size = os.path.getsize(compressed)
        b64_size = int(file_size * 4 / 3)  # base64 expansion ~33%
        if b64_size <= max_b64_bytes:
            return compressed

    # If compression still not enough, return compressed anyway (best effort)
    if os.path.exists(compressed):
        return compressed
    return audio_path


# ── VAD timestamp generator ────────────────────────────────────────


def _generate_timestamps_vad(audio_path: str, full_text: str) -> List[dict]:
    """Generate approximate timestamps by mapping text segments to VAD speech regions.

    Step 1: Load audio and detect speech activity regions with librosa VAD
    Step 2: Split text into subtitle-sized segments by punctuation
    Step 3: Calculate total speech duration and assign each text segment
            a proportional share of speech time based on character count
    Step 4: Map each segment to sequential speech regions, producing
            accurate start/end timestamps aligned to actual speech

    This works well because speech regions correspond to spoken content,
    and character count roughly correlates with speech duration.
    """
    # Defensive zhconv: ensure full_text is Simplified Chinese
    try:
        from zhconv import convert as _z
        full_text = _z(full_text, "zh-cn")
    except Exception:
        pass

    import librosa

    # Step 1: Load audio and detect speech regions
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_duration = len(waveform) / sr

    # Use energy-based VAD to find speech intervals
    intervals = librosa.effects.split(waveform, top_db=35, frame_length=2048, hop_length=512)

    if len(intervals) == 0:
        # No speech detected; treat entire audio as one region
        intervals = [(0, len(waveform))]

    # Convert sample indices to seconds
    speech_regions = [(s / sr, e / sr) for s, e in intervals]
    total_speech_sec = sum(e - s for s, e in speech_regions)

    if total_speech_sec <= 0:
        total_speech_sec = audio_duration
        speech_regions = [(0.0, audio_duration)]

    # Step 2: Split text into subtitle segments
    is_cjk = text_splitter.is_mostly_cjk(full_text)
    max_len = text_splitter.DEFAULT_MAX_CHARS if is_cjk else text_splitter.DEFAULT_MAX_CHARS_EN  # Phase 1: 引用 text_splitter 共享常量
    text_segments = text_splitter.split_text_for_subtitles(full_text, max_len, is_cjk)

    if not text_segments:
        return [{"start": 0.0, "end": audio_duration, "text": full_text}]

    # Step 3: Calculate proportional duration for each text segment
    total_chars = sum(len(s) for s in text_segments)
    if total_chars == 0:
        total_chars = 1

    # Step 4: Walk through speech regions, assigning text segments sequentially
    segments = []
    region_idx = 0
    region_consumed = 0.0  # How much of current region has been used (seconds)

    for seg_text in text_segments:
        # Proportional speech time for this text segment
        char_ratio = len(seg_text) / total_chars
        seg_speech_duration = char_ratio * total_speech_sec

        # Find start position in speech regions
        if region_idx >= len(speech_regions):
            # Ran out of regions; extend from last known position
            last_end = speech_regions[-1][1] if speech_regions else audio_duration
            seg_start = last_end
            seg_end = min(last_end + seg_speech_duration, audio_duration)
        else:
            r_start, r_end = speech_regions[region_idx]
            seg_start = r_start + region_consumed

            # Walk through regions to fill this segment's duration
            remaining_duration = seg_speech_duration
            seg_end = seg_start

            while remaining_duration > 0 and region_idx < len(speech_regions):
                r_start, r_end = speech_regions[region_idx]
                available = r_end - (r_start + region_consumed)

                if available <= remaining_duration:
                    # Consume rest of this region
                    seg_end = r_end
                    remaining_duration -= available
                    region_idx += 1
                    region_consumed = 0.0
                else:
                    # Partially consume this region
                    region_consumed += remaining_duration
                    seg_end = r_start + region_consumed
                    remaining_duration = 0

        # Ensure minimum segment duration (0.5s)
        if seg_end - seg_start < 0.5:
            seg_end = min(seg_start + 0.5, audio_duration)

        segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg_text,
        })

    # Fix: ensure last segment doesn't exceed audio duration
    if segments:
        segments[-1]["end"] = min(segments[-1]["end"], round(audio_duration, 3))
        # Ensure no overlap between consecutive segments
        for i in range(len(segments) - 1):
            if segments[i]["end"] > segments[i + 1]["start"]:
                segments[i]["end"] = segments[i + 1]["start"]

    return segments


# ── Public API ──────────────────────────────────────────────────────


def transcribe_cloud(audio_path: str, language: str = "auto",
                     glossary: dict = None) -> Tuple[str, list]:
    """Transcribe audio using the configured cloud API provider.

    Routing is **config-driven** via the ``request_mode`` field in
    ``~/.shiyu/config.json`` (engine section):

      * ``"whisper"``  (default) → OpenAI-compatible
        ``POST {base}/audio/transcriptions`` (multipart upload). Works
        with OpenAI, Groq, DeepSeek, and any provider exposing the
        OpenAI Whisper endpoint.
      * ``"chat_completions"`` → OpenAI-compatible
        ``POST {base}/chat/completions`` with an ``input_audio`` message
        part. Works with DashScope (Qwen-ASR), Xiaomi MiMo, and any
        provider following the OpenAI multimodal chat shape.

    Provider-specific quirks (auth header, asr_options) come from the
    config rather than hard-coded URL sniffing. See ``_resolve_auth_header``
    and the engine config schema for details.
    """
    config = load_engine_config()
    api_url = config["api_url"].rstrip("/")
    api_key = config["api_key"]
    model = (config.get("model") or "").strip()
    request_mode = (config.get("request_mode") or "whisper").lower().strip()

    if not api_url or not api_key:
        raise ValueError("Cloud engine not configured. Set API URL and key in settings.")

    # Explicit config-driven routing
    if request_mode == "chat_completions":
        _srt, segments = _transcribe_dashscope(
            audio_path, language, api_url, api_key, model, glossary or {}
        )
        srt_content = _segments_to_srt(segments)
        return srt_content, segments

    if request_mode != "whisper":
        raise ValueError(
            f"Unknown engine.request_mode: {request_mode!r}. "
            f"Expected 'whisper' or 'chat_completions'."
        )

    # Default: OpenAI Whisper /audio/transcriptions
    segments = None
    try:
        _srt, segments = _transcribe_whisper(audio_path, language, api_url, api_key, model, glossary or {})
    except Exception:
        pass

    # Whisper path: error if no segments returned
    if segments is None:
        raise RuntimeError(
            "Cloud API (whisper) transcription failed. "
            "Check that your API URL and key are correct."
        )

    srt_content = _segments_to_srt(segments)
    return srt_content, segments


# ── Standard Whisper API (OpenAI, Groq, DeepSeek, etc.) ─────────────


def _transcribe_whisper(audio_path: str, language: str, api_url: str, api_key: str,
                        model: str, glossary: dict) -> Tuple[str, list]:
    """Transcribe via standard OpenAI Whisper /audio/transcriptions with multipart file upload."""
    # Normalize base URL (strip /chat/completions if present)
    base = _normalize_base_url(api_url)
    endpoint = f"{base}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    form_fields = {"response_format": "verbose_json"}
    if model:
        form_fields["model"] = model
    if language and language != "auto":
        form_fields["language"] = language

    if _USE_CURL:
        result = _post_multipart_via_curl(endpoint, headers, form_fields, audio_path)
    else:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(endpoint, headers=headers, files=files, data=form_fields, timeout=120)
            if resp.status_code == 401:
                raise PermissionError("Cloud API authentication failed. Check your API key.")
            if resp.status_code != 200:
                raise _friendly_http_error(resp, endpoint, "Whisper /audio/transcriptions")
            result = resp.json()

    segments = _extract_segments(result)
    if not segments:
        # API returned text but no segments; generate timestamps via VAD
        text = result.get("text", "") or ""
        if text:
            text = text.strip()
            # zhconv: Traditional → Simplified
            try:
                from zhconv import convert as _z
                text = _z(text, "zh-cn")
            except Exception:
                pass
            if glossary:
                for bad, good in glossary.items():
                    text = text.replace(bad, good)
            segments = _generate_timestamps_vad(audio_path, text)
        else:
            segments = []
    else:
        # Apply glossary to native segments
        if glossary:
            for seg in segments:
                for bad, good in glossary.items():
                    seg["text"] = seg["text"].replace(bad, good)

    srt_content = _segments_to_srt(segments)
    return srt_content, segments


# ── DashScope compatible mode (chat completions + input_audio) ──────


def _transcribe_dashscope_payload(compressed_path: str, api_url: str, api_key: str, model: str) -> str:
    """Send a single compressed audio file to DashScope and return plain text."""
    import logging, traceback
    _log = logging.getLogger("shiyu")
    _log.info(f"[cloud_engine v4] Sending payload: path={compressed_path}, size={os.path.getsize(compressed_path)} bytes")
    try:
        return _transcribe_dashscope_payload_impl(compressed_path, api_url, api_key, model, _log)
    except AttributeError as e:
        if 'strip' in str(e):
            _log.error(f"[cloud_engine v4] BUG: .strip() on NoneType at:\n{traceback.format_exc()}")
            raise RuntimeError(f"Internal bug: .strip() called on NoneType. Check log for traceback.")
        raise


def _transcribe_dashscope_payload_impl(compressed_path: str, api_url: str, api_key: str, model: str, _log) -> str:
    """Generic OpenAI-compatible chat-completions + input_audio ASR call.

    Works with any provider that follows the OpenAI chat completions
    schema with an ``input_audio`` content part. Provider-specific quirks
    (auth header, asr_options) come from the engine config rather than
    hard-coded detection, so adding a new provider is a config-only
    change — no code edits required.
    """
    # Re-load the full engine config to read auth_header / asr_options
    full_cfg = load_engine_config()
    auth_header = _resolve_auth_header(full_cfg)
    asr_options = full_cfg.get("asr_options") or {}
    provider_name = full_cfg.get("display_name", "Cloud")

    with open(compressed_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    # Determine MIME type from extension
    ext = Path(compressed_path).suffix.lower()
    mime_map = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg", ".webm": "audio/webm",
        ".aac": "audio/aac", ".opus": "audio/opus",
    }
    mime = mime_map.get(ext, "audio/wav")

    # Build chat completions request body.
    # Per OpenAI-compatible docs: content array should contain ONLY one
    # input_audio element, NO additional text elements alongside it.
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:{mime};base64,{audio_b64}"
                        }
                    }
                ],
            }
        ],
        "stream": False,
    }
    # Provider-specific ASR options (e.g. DashScope: enable_itn, MiMo: language)
    if asr_options:
        body["asr_options"] = asr_options

    # Normalize URL and build endpoint
    base = _normalize_base_url(api_url)
    endpoint = f"{base}/chat/completions"

    headers = {**auth_header, "Content-Type": "application/json"}
    resp = requests.post(endpoint, headers=headers, json=body, timeout=120)
    if resp.status_code == 401:
        raise PermissionError(f"{provider_name} authentication failed. Check your API key.")
    if resp.status_code != 200:
        raise _friendly_http_error(resp, endpoint, f"{provider_name} /chat/completions")
    result = resp.json()

    # Check for error response
    if "error" in result:
        err_obj = result["error"]
        if isinstance(err_obj, dict):
            err_msg = err_obj.get("message") or err_obj.get("type") or str(err_obj)
        else:
            err_msg = str(err_obj)
        raise RuntimeError(f"{provider_name} API error: {err_msg}")

    # Parse chat completions response
    _log.debug(f"[cloud_engine v3/{provider_name}] Raw response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    try:
        content = result["choices"][0]["message"]["content"]
        _log.info(f"[cloud_engine v3/{provider_name}] content type={type(content).__name__}, value={repr(content)[:200]}")
    except (KeyError, IndexError):
        _log.error(f"[cloud_engine v3/{provider_name}] Unexpected response: {json.dumps(result, ensure_ascii=False)[:1000]}")
        raise RuntimeError(
            f"Unexpected {provider_name} response format: "
            f"{json.dumps(result, ensure_ascii=False)[:500]}"
        )

    if content is None or (isinstance(content, str) and not content.strip()):
        return ""  # Return empty string — caller's loop handles via `if not text: continue`

    try:
        text = content.strip() if isinstance(content, str) else str(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {provider_name} transcription content: {e}")

    # Traditional Chinese → Simplified Chinese
    try:
        from zhconv import convert as _zhconv_convert
        text = _zhconv_convert(text, "zh-cn")
    except ImportError:
        pass
    except Exception:
        pass
    return text


def _refine_timestamps_with_ctc(audio_path: str, full_text: str,
                                 sample_rate: int,
                                 original_segments: list) -> list:
    """Refine segment timestamps using SenseVoice CTC forced alignment.

    When SenseVoice model is available, uses CTC Viterbi alignment to get
    accurate frame-level timestamps for the DashScope-transcribed text.
    Falls back to original segments if model or alignment fails.

    Args:
        audio_path: Path to the audio file.
        full_text: Full transcription text from DashScope.
        sample_rate: Audio sample rate.
        original_segments: Original segments to fall back on.

    Returns:
        Refined segment list with improved timestamps, or None on failure.
    """
    import logging
    import numpy as np
    _log = logging.getLogger("shiyu")

    # Try to load SenseVoice model + SentencePiece
    try:
        # 1. Load audio
        import librosa
        waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        audio_duration = len(waveform) / sr

        # 2. Compute fbank features
        from app.core.audio_processor import compute_fbank_kaldi, apply_lfr, apply_cmvn, load_cmvn
        from app.core.config_loader import get_model_dir

        model_dir = get_model_dir()
        if not os.path.exists(os.path.join(model_dir, "model.onnx")):
            return None

        # 3. Load SentencePiece for forced alignment
        import sentencepiece as spm
        sp_model = None
        for fname in os.listdir(model_dir):
            if fname.endswith(".bpe.model"):
                sp_model = spm.SentencePieceProcessor()
                sp_model.Load(os.path.join(model_dir, fname))
                break
        if sp_model is None:
            return None

        # 4. Load ONNX and run encoder
        import onnxruntime as ort
        session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"]
        )

        # 5. Compute fbank
        audio_cfg = {"n_mels": 80, "frame_length": 25, "frame_shift": 10}
        fbank = compute_fbank_kaldi(waveform, sr,
                                     n_mels=audio_cfg["n_mels"],
                                     frame_length=audio_cfg["frame_length"],
                                     frame_shift=audio_cfg["frame_shift"])
        lfr_features = apply_lfr(fbank, m=7, n=6)
        cmvn_path = os.path.join(model_dir, "am.mvn")
        if os.path.exists(cmvn_path):
            cmvn = load_cmvn(cmvn_path)
            features = apply_cmvn(lfr_features, cmvn)
        else:
            features = lfr_features
        features = features[np.newaxis, :, :].astype(np.float32)
        input_len = np.array([features.shape[1]], dtype=np.int32)

        # Language = auto (id 0), text_norm = withitn (id 14)
        lang_id = np.array([0], dtype=np.int64)
        text_norm = np.array([14], dtype=np.int64)

        input_names = [inp.name for inp in session.get_inputs()]
        onnx_inputs = {
            input_names[0]: features,
            input_names[1]: input_len,
            input_names[2]: lang_id,
            input_names[3]: text_norm,
        }
        outputs = session.run(None, onnx_inputs)
        logits = outputs[0][0]  # [T, V]

        # 6. Run forced alignment
        from app.core.ctc_aligner import align_text_to_timestamps
        char_timestamps = align_text_to_timestamps(
            logits, full_text, sp_model,
            frame_to_sec=0.06, ctc_delay_sec=0.15
        )

        if not char_timestamps:
            return None

        # 7. Split text into subtitle segments
        is_cjk = text_splitter.is_mostly_cjk(full_text)
        max_len = text_splitter.DEFAULT_MAX_CHARS if is_cjk else text_splitter.DEFAULT_MAX_CHARS_EN  # Phase 1: 引用 text_splitter 共享常量
        sub_texts = text_splitter.split_text_for_subtitles(full_text, max_len, is_cjk)
        if not sub_texts:
            return None

        # 8. Map segments to aligned timestamps
        # Build char-index → timestamp lookup
        char_map = []
        for ct in char_timestamps:
            if ct["char"].strip():
                char_map.append((ct["char"], ct["start"], ct["end"]))

        # Walk through sub_texts sequentially, accumulating character positions
        segments = []
        char_pos = 0
        for sub in sub_texts:
            seg_chars = list(sub)
            seg_len = len(seg_chars)
            if seg_len == 0:
                continue
            end_pos = min(char_pos + seg_len, len(char_map))
            if end_pos <= char_pos:
                continue
            start_sec = char_map[char_pos][1]
            end_sec = char_map[end_pos - 1][2]
            # Enforce minimum 0.5s duration
            if end_sec - start_sec < 0.5:
                end_sec = start_sec + 0.5
            segments.append({
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "text": sub
            })
            char_pos = end_pos

        if segments:
            # Fix boundaries: ensure no overlap
            for i in range(len(segments) - 1):
                if segments[i]["end"] < segments[i + 1]["start"]:
                    segments[i]["end"] = segments[i + 1]["start"]
                elif segments[i]["end"] > segments[i + 1]["start"]:
                    segments[i]["end"] = segments[i + 1]["start"]
            # Clamp last segment duration
            if segments:
                segments[-1]["end"] = min(
                    max(segments[-1]["end"], segments[-1]["start"] + 0.5),
                    audio_duration
                )
            _log.info(f"[CTC Align] Refined {len(segments)} segments with forced alignment")
            return segments

    except Exception as e:
        _log.warning(f"[CTC Align] Failed, falling back to original: {e}")
    return None


def _find_energy_dip(
    waveform, target_idx: int, sample_rate: int,
    window_sec: float = 1.5, hop_length: int = 512
) -> int:
    """在 target_idx 附近找帧能量最低点作为切点。
    """
    import numpy as np
    window_samples = int(window_sec * sample_rate)
    search_start = max(0, target_idx - window_samples)
    search_end = min(len(waveform), target_idx + window_samples)

    best_idx = target_idx
    best_energy = float("inf")

    pos = search_start
    while pos + hop_length <= search_end:
        frame = waveform[pos:pos + hop_length]
        energy = np.sqrt(np.mean(frame ** 2))
        if energy < best_energy:
            best_energy = energy
            best_idx = pos
        pos += hop_length
    return best_idx


def _transcribe_dashscope(audio_path: str, language: str, api_url: str, api_key: str,
                          model: str, glossary: dict) -> Tuple[str, list]:
    """Transcribe via DashScope with energy-dip aware chunking for accurate timestamps.

    Uses fixed 30s windows shifted to nearest energy dip — ensures 100% coverage,
    no gaps, and minimal sentence cutting.
    """
    import librosa
    import soundfile as sf

    # Step 1: Load audio and create energy-dip aware chunks (no gaps, 100% coverage)
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_duration = len(waveform) / sr

    CHUNK_SEC = 30
    chunk_samples = int(CHUNK_SEC * sr)
    boundary_samples = set()
    boundary_samples.add(0)
    for start_sample in range(chunk_samples, len(waveform), chunk_samples):
        dip = _find_energy_dip(waveform, start_sample, sr, window_sec=1.5)
        boundary_samples.add(dip)
    boundary_samples.add(len(waveform))
    sorted_boundaries = sorted(boundary_samples)

    merged = []
    for i in range(len(sorted_boundaries) - 1):
        start_sample = sorted_boundaries[i]
        end_sample = sorted_boundaries[i + 1]
        # Skip tiny gaps (<0.3s) — merge into previous chunk
        if merged and (end_sample - start_sample) / sr < 0.3:
            merged[-1] = (merged[-1][0], end_sample)
        else:
            merged.append((start_sample, end_sample))

    # Step 2: Transcribe each interval via DashScope with accurate timestamps
    import time as _time
    all_segments = []
    accumulated_text = []
    total_calls = len(merged)
    _t0 = _time.time()
    _MAX_TOTAL_SEC = 600  # 10 minutes total cap

    for idx, (start_sample, end_sample) in enumerate(merged):
        if _time.time() - _t0 > _MAX_TOTAL_SEC:
            raise TimeoutError(
                f"Cloud API transcription timed out after {_MAX_TOTAL_SEC}s "
                f"({idx}/{total_calls} chunks completed). Try splitting the audio file."
            )
        seg_audio = waveform[start_sample:end_sample]
        seg_start = start_sample / sr
        seg_end = end_sample / sr
        seg_duration = seg_end - seg_start

        # Skip very short segments (likely noise rather than speech)
        if seg_duration < 0.3:
            continue

        # Write segment audio to temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        try:
            sf.write(temp_wav, seg_audio, sr)

            # Compress for smaller payload (mp3 at 32k)
            compressed = _compress_audio_for_upload(temp_wav)
            compressed_is_temp = (compressed != temp_wav)
            try:
                text = _transcribe_dashscope_payload(compressed, api_url, api_key, model)
            finally:
                if compressed_is_temp:
                    try:
                        os.unlink(compressed)
                    except OSError:
                        pass
        finally:
            try:
                os.unlink(temp_wav)
            except OSError:
                pass

        if not text or not text.strip():
            continue

        # Apply glossary on the fly
        if glossary:
            for bad, good in glossary.items():
                text = text.replace(bad, good)

        accumulated_text.append(text)

        # Split text into subtitle-friendly segments within this time range
        is_cjk = text_splitter.is_mostly_cjk(text)
        max_len = text_splitter.DEFAULT_MAX_CHARS if is_cjk else text_splitter.DEFAULT_MAX_CHARS_EN  # Phase 1: 引用 text_splitter 共享常量
        sub_texts = text_splitter.split_text_for_subtitles(text, max_len, is_cjk)

        if len(sub_texts) <= 1:
            all_segments.append({"start": seg_start, "end": seg_end, "text": text})
        else:
            # Distribute sub-segments proportionally within this interval
            total_chars = sum(len(t) for t in sub_texts) or 1
            offset = 0.0
            for sub in sub_texts:
                ratio = len(sub) / total_chars
                sub_start = seg_start + offset
                sub_end = seg_start + offset + ratio * seg_duration
                offset += ratio * seg_duration
                all_segments.append({"start": sub_start, "end": sub_end, "text": sub})

    full_text = " ".join(accumulated_text)

    # Refine timestamps with CTC forced alignment if SenseVoice model available
    refined = _refine_timestamps_with_ctc(audio_path, full_text, sr, all_segments)
    if refined:
        all_segments = refined

    # Fallback: if chunking produced no segments but we have text, create single segment
    if not all_segments and full_text.strip():
        all_segments = [{"start": 0.0, "end": audio_duration, "text": full_text}]

    if not all_segments:
        raise RuntimeError("Cloud API returned empty transcription (no speech detected).")

    srt_content = _segments_to_srt(all_segments)
    return srt_content, all_segments


# ── Response parsing ────────────────────────────────────────────────


def _extract_segments(result: dict) -> list:
    """Extract time-aligned segments from Whisper verbose_json response."""
    segments_raw = result.get("segments", [])
    if not segments_raw:
        return []

    # Traditional Chinese → Simplified Chinese
    try:
        from zhconv import convert as _zhconv_convert
    except ImportError:
        _zhconv_convert = None

    out = []
    for s in segments_raw:
        text = s["text"].strip()
        if _zhconv_convert:
            try:
                text = _zhconv_convert(text, "zh-cn")
            except Exception:
                pass
        out.append({"start": s["start"], "end": s["end"], "text": text})
    return out


# ── HTTP helpers (curl / requests) ────────────────────────────────────


def _post_multipart_via_curl(endpoint: str, headers: dict, form_fields: dict, audio_path: str) -> dict:
    """Send multipart form data with file upload via curl (bypasses Python OpenSSL issues)."""
    header_args = []
    for k, v in headers.items():
        header_args += ["-H", f"{k}: {v}"]

    cmd = [
        "curl", "-s", "--max-time", "300",
        "-X", "POST",
        endpoint,
    ] + header_args

    for key, value in form_fields.items():
        cmd += ["-F", f"{key}={value}"]

    cmd += ["-F", f"file=@{audio_path}"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=310)

        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or "curl process failed"
            raise RuntimeError(f"Cloud API request failed: {err}")

        stdout = (proc.stdout or "").strip()
        if not stdout:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                f"Cloud API returned empty response. Stderr: {stderr[:200]}" if stderr
                else "Cloud API returned empty response."
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Cloud API request timed out after 300 seconds.")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        if "401" in stdout or "Unauthorized" in stdout:
            raise PermissionError("Cloud API authentication failed. Check your API key.")
        raise RuntimeError(f"Cloud API response error: {stdout[:500]}")


# ── SRT formatting ──────────────────────────────────────────────────


def _segments_to_srt(segments: List[dict]) -> str:
    """Convert segments list to SRT format string."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = _fmt_srt_time(seg["start"])
        end_ts = _fmt_srt_time(seg["end"])
        srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{seg['text']}\n")
    return "\n".join(srt_lines)


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── BaseTranscriber Adapter ────────────────────────────────────────

from .engine_base import BaseTranscriber, TranscribeResult, Segment
from .chunk_merger import ChunkMerger


class CloudTranscriber(BaseTranscriber):
    """Adapter wrapping cloud engine as a BaseTranscriber.

    Supports standard Whisper API and DashScope.
    For DashScope with continuous audio, uses overlapping chunks + ChunkMerger
    for more accurate timestamps.
    """

    name = "cloud"
    display_name = "云模型"
    requires_network = True
    is_local = False

    def __init__(self, use_chunk_merger: bool = True):
        self._use_chunk_merger = use_chunk_merger
        self._merger = ChunkMerger()

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        **kwargs,
    ) -> TranscribeResult:
        config = load_engine_config()
        api_url = config["api_url"].rstrip("/")
        api_key = config["api_key"]
        model = (config.get("model") or "").strip()
        request_mode = (config.get("request_mode") or "whisper").lower().strip()

        if not api_url or not api_key:
            raise ValueError("Cloud engine not configured.")

        glossary = kwargs.get("glossary", {})

        if request_mode == "chat_completions":
            # OpenAI-compatible chat completions + input_audio (DashScope / MiMo / etc.)
            if self._use_chunk_merger:
                return self._transcribe_dashscope_merged(
                    audio_path, language, api_url, api_key, model, glossary
                )
            else:
                _, segments = _transcribe_dashscope(
                    audio_path, language, api_url, api_key, model, glossary
                )
        elif request_mode == "whisper":
            _, segments = _transcribe_whisper(
                audio_path, language, api_url, api_key, model, glossary
            )

        typed = [Segment(start=s["start"], end=s["end"], text=s["text"])
                 for s in segments]
        return TranscribeResult(segments=typed)

    def _transcribe_dashscope_merged(
        self,
        audio_path: str,
        language: str,
        api_url: str,
        api_key: str,
        model: str,
        glossary: dict,
    ) -> TranscribeResult:
        """DashScope path with overlapping chunks + ChunkMerger for better timestamps."""
        import librosa
        import soundfile as sf

        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(waveform) / sr

        # Use longer chunks with overlap
        CHUNK_SEC = 20
        OVERLAP_SEC = 10
        chunk_samples = int(CHUNK_SEC * sr)
        overlap_samples = int(OVERLAP_SEC * sr)
        step_samples = chunk_samples - overlap_samples

        if step_samples <= 0:
            step_samples = chunk_samples // 2

        # Build overlapping chunks
        chunks = []
        t = 0
        while t < len(waveform):
            end = min(t + chunk_samples, len(waveform))
            chunks.append((t, end))
            t += step_samples
            if t + overlap_samples >= len(waveform):
                break

        if len(chunks) <= 1:
            # Short audio: no need for merger
            _, segments = _transcribe_dashscope(
                audio_path, language, api_url, api_key, model, glossary
            )
            typed = [Segment(start=s["start"], end=s["end"], text=s["text"])
                     for s in segments]
            return TranscribeResult(segments=typed)

        # Transcribe each overlapping chunk
        chunk_results = []
        for start_sample, end_sample in chunks:
            seg_audio = waveform[start_sample:end_sample]
            seg_duration = (end_sample - start_sample) / sr

            if seg_duration < 1.0:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav = f.name
            try:
                sf.write(temp_wav, seg_audio, sr)
                compressed = _compress_audio_for_upload(temp_wav)
                compressed_is_temp = (compressed != temp_wav)
                try:
                    text = _transcribe_dashscope_payload(compressed, api_url, api_key, model)
                finally:
                    if compressed_is_temp:
                        try:
                            os.unlink(compressed)
                        except OSError:
                            pass
            finally:
                try:
                    os.unlink(temp_wav)
                except OSError:
                    pass

            if not text or not text.strip():
                chunk_results.append(TranscribeResult())
                continue

            # Apply glossary
            if glossary:
                for bad, good in glossary.items():
                    text = text.replace(bad, good)

            # Create a single segment per chunk for merger
            seg_start = start_sample / sr
            seg_end = end_sample / sr
            seg = Segment(start=seg_start, end=seg_end, text=text)
            chunk_results.append(TranscribeResult(segments=[seg]))

        if not chunk_results:
            raise RuntimeError("Cloud API returned empty transcription (no speech detected).")

        # Merge overlapping transcriptions
        merged = self._merger.merge(chunk_results, overlap_duration=OVERLAP_SEC)

        # Build subtitle segments from merged result
        if not merged.segments:
            return merged

        all_segments = []
        for seg in merged.segments:
            is_cjk = text_splitter.is_mostly_cjk(seg.text)
            max_len = text_splitter.DEFAULT_MAX_CHARS if is_cjk else text_splitter.DEFAULT_MAX_CHARS_EN  # Phase 1: 引用 text_splitter 共享常量
            sub_texts = text_splitter.split_text_for_subtitles(seg.text, max_len, is_cjk)

            if len(sub_texts) <= 1:
                all_segments.append(Segment(start=seg.start, end=seg.end, text=seg.text))
            else:
                total_chars = sum(len(t) for t in sub_texts) or 1
                duration = seg.end - seg.start
                offset = 0.0
                for sub in sub_texts:
                    ratio = len(sub) / total_chars
                    sub_start = seg.start + offset
                    sub_end = seg.start + offset + ratio * duration
                    offset += ratio * duration
                    all_segments.append(Segment(start=sub_start, end=sub_end, text=sub))

        # Clean up boundaries
        for i in range(len(all_segments) - 1):
            if all_segments[i].end > all_segments[i + 1].start:
                all_segments[i].end = all_segments[i + 1].start

        if all_segments:
            all_segments[-1].end = min(all_segments[-1].end, audio_duration)

        return TranscribeResult(segments=all_segments)

    def is_available(self):
        config = load_engine_config()
        if config.get("enabled") and config.get("api_url") and config.get("api_key"):
            return True, ""
        return False, "Cloud API not configured"
