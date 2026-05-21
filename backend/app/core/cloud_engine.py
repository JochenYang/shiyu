"""
Cloud transcription engine supporting both:
1. Standard OpenAI Whisper API (POST /audio/transcriptions, multipart file upload)
   - Providers: OpenAI, Groq, DeepSeek, etc.
   - Returns native segment timestamps (verbose_json)
2. DashScope compatible mode (POST /chat/completions, input_audio base64)
   - Provider: DashScope Qwen3-ASR-Flash
   - Returns plain text only; timestamps generated locally via VAD
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

CONFIG_FILE = Path.home() / ".shiyu" / "config.json"

# On Windows, curl (Schannel TLS) avoids OpenSSL SSLEOFError with some servers
_USE_CURL = sys.platform == "win32"

# Max base64-encoded payload size for DashScope (conservative: 9.5MB to stay within 10MB limit)
_DASHSCOPE_MAX_B64_BYTES = int(9.5 * 1024 * 1024)


# ── Config ──────────────────────────────────────────────────────────


def load_engine_config() -> Dict[str, Any]:
    """Load cloud engine config from ~/.shiyu/config.json"""
    if not CONFIG_FILE.exists():
        return {"api_url": "", "api_key": "", "model": "whisper-1", "enabled": False}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("engine", {"api_url": "", "api_key": "", "model": "whisper-1", "enabled": False})
    except (json.JSONDecodeError, IOError):
        return {"api_url": "", "api_key": "", "model": "whisper-1", "enabled": False}


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


# ── Provider detection ──────────────────────────────────────────────


def _is_dashscope_url(api_url: str) -> bool:
    """Detect if the API URL points to DashScope."""
    return "dashscope.aliyuncs.com" in api_url


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


# ── ffmpeg helpers ──────────────────────────────────────────────────


def _find_ffmpeg() -> str:
    """Locate ffmpeg executable, with Windows fallback paths."""
    ffmpeg_exe = os.environ.get("FFMPEG_PATH", "ffmpeg")
    # Quick check if it's in PATH
    try:
        subprocess.run(
            [ffmpeg_exe, "-version"],
            capture_output=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return ffmpeg_exe
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback for common WinGet installation path
    winget_ffmpeg = (
        r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
    )
    if os.path.exists(winget_ffmpeg):
        return winget_ffmpeg
    return ffmpeg_exe  # Let it fail naturally with a clear error


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
    ffmpeg_exe = _find_ffmpeg()
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


def _is_mostly_cjk(text: str) -> bool:
    """Check if text is primarily CJK (Chinese/Japanese/Korean)."""
    cjk = sum(1 for c in text if (
        '\u4e00' <= c <= '\u9fff' or   # CJK Unified
        '\u3040' <= c <= '\u30ff' or   # Hiragana + Katakana
        '\uac00' <= c <= '\ud7af'      # Hangul
    ))
    return cjk > len(text) * 0.3


def _split_text_for_subtitles(text: str, max_len: int = 25, is_cjk: bool = True) -> List[str]:
    """Split text into subtitle-friendly segments using punctuation boundaries.

    Level 1: Split at sentence boundaries (。！？；!?;.)
    Level 2: Split long sentences at commas/colons
    Level 3: Hard-cut fallback (word-aware for English, char-level for CJK)
    """
    # Level 1: Sentence boundary split
    if is_cjk:
        parts = re.split(r"([。！？；!?;.])", text)
    else:
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

    if not final:
        final = [text.strip()] if text.strip() else []

    return final


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
    is_cjk = _is_mostly_cjk(full_text)
    max_len = 25 if is_cjk else 42
    text_segments = _split_text_for_subtitles(full_text, max_len, is_cjk)

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

    Auto-detects DashScope vs standard Whisper-compatible providers.
    For DashScope: compresses audio, gets plain text, generates VAD timestamps.
    For Whisper: uses native verbose_json timestamps.

    Returns:
        Tuple of (srt_content, segments_list)
        Each segment: {"start": float, "end": float, "text": str}
    """
    config = load_engine_config()
    api_url = config["api_url"].rstrip("/")
    api_key = config["api_key"]
    model = config.get("model", "whisper-1")

    if not api_url or not api_key:
        raise ValueError("Cloud engine not configured. Set API URL and key in settings.")

    if _is_dashscope_url(api_url):
        return _transcribe_dashscope(audio_path, language, api_url, api_key, model, glossary or {})
    else:
        return _transcribe_whisper(audio_path, language, api_url, api_key, model, glossary or {})


# ── Standard Whisper API (OpenAI, Groq, DeepSeek, etc.) ─────────────


def _transcribe_whisper(audio_path: str, language: str, api_url: str, api_key: str,
                        model: str, glossary: dict) -> Tuple[str, list]:
    """Transcribe via standard OpenAI Whisper /audio/transcriptions with multipart file upload."""
    endpoint = f"{api_url}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    form_fields = {"model": model, "response_format": "verbose_json"}
    if language and language != "auto":
        form_fields["language"] = language

    if _USE_CURL:
        result = _post_multipart_via_curl(endpoint, headers, form_fields, audio_path)
    else:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(endpoint, headers=headers, files=files, data=form_fields, timeout=300)
            if resp.status_code == 401:
                raise PermissionError("Cloud API authentication failed. Check your API key.")
            if resp.status_code != 200:
                raise RuntimeError(f"Cloud API error ({resp.status_code}): {resp.text[:500]}")
            result = resp.json()

    segments = _extract_segments(result)
    if not segments:
        # API returned text but no segments; generate timestamps via VAD
        text = result.get("text", "") or ""
        if text:
            text = text.strip()
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

    # Build chat completions request body
    # IMPORTANT: Must include a text instruction alongside the audio.
    # Without one, the model may return empty (no output) for non-speech audio
    # like songs, music, or rap.
    body = {
        "model": model or "qwen3-asr-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:{mime};base64,{audio_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请将音频中的语音转写为文字，包括歌曲歌词。"
                    }
                ]
            }
        ],
        "stream": False,
    }

    # Normalize URL and build endpoint
    base = _normalize_base_url(api_url)
    endpoint = f"{base}/chat/completions"

    auth_header = f"Bearer {api_key}"

    headers = {"Authorization": auth_header, "Content-Type": "application/json"}
    resp = requests.post(endpoint, headers=headers, json=body, timeout=300)
    if resp.status_code == 401:
        raise PermissionError("Cloud API authentication failed. Check your API key.")
    if resp.status_code != 200:
        raise RuntimeError(f"Cloud API error ({resp.status_code}): {resp.text[:500]}")
    result = resp.json()

    # Check for error response
    if "error" in result:
        err_msg = result["error"].get("message", str(result["error"]))
        raise RuntimeError(f"DashScope API error: {err_msg}")

    # Parse chat completions response
    _log.debug(f"[cloud_engine v3] Raw response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    try:
        content = result["choices"][0]["message"]["content"]
        _log.info(f"[cloud_engine v3] content type={type(content).__name__}, value={repr(content)[:200]}")
    except (KeyError, IndexError):
        _log.error(f"[cloud_engine v3] Unexpected response: {json.dumps(result, ensure_ascii=False)[:1000]}")
        raise RuntimeError(
            f"Unexpected DashScope response format: "
            f"{json.dumps(result, ensure_ascii=False)[:500]}"
        )

    if content is None or (isinstance(content, str) and not content.strip()):
        return ""  # Return empty string — caller's loop handles via `if not text: continue`

    try:
        text = content.strip() if isinstance(content, str) else str(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse DashScope transcription content: {e}")
    return text


def _transcribe_dashscope(audio_path: str, language: str, api_url: str, api_key: str,
                          model: str, glossary: dict) -> Tuple[str, list]:
    """Transcribe via DashScope with accurate timestamps using VAD pre-chunking.

    1. VAD on original audio → detect speech intervals (exact start/end)
    2. For each interval: extract audio → compress → send to DashScope → get text
    3. Each segment's timestamp = the VAD interval time range (accurate!)
    4. Long text within a segment is split by punctuation and distributed proportionally
    """
    import librosa
    import soundfile as sf

    # Step 1: Load audio and run VAD
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_duration = len(waveform) / sr

    intervals = librosa.effects.split(waveform, top_db=35, frame_length=2048, hop_length=512)

    # If no speech detected, treat entire audio as one segment
    if len(intervals) == 0:
        intervals = [(0, len(waveform))]

    # Merge nearby intervals (gap < 0.5s of silence)
    gap_threshold = int(0.5 * sr)
    merged = []
    for start, end in intervals:
        if merged and start - merged[-1][1] < gap_threshold:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Fallback: fixed-duration chunking for continuous audio (music/rap/etc.)
    # When VAD finds no silence gaps, split into fixed 12-second chunks
    FIXED_CHUNK_SEC = 12
    if audio_duration > 30 and len(merged) < 3:
        chunk_size = int(FIXED_CHUNK_SEC * sr)
        merged = [(i, min(i + chunk_size, len(waveform)))
                  for i in range(0, len(waveform), chunk_size)]

    # Step 2: Transcribe each interval and build segments with accurate timestamps
    all_segments = []
    accumulated_text = []
    total_calls = len(merged)

    for idx, (start_sample, end_sample) in enumerate(merged):
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
        is_cjk = _is_mostly_cjk(text)
        max_len = 25 if is_cjk else 42
        sub_texts = _split_text_for_subtitles(text, max_len, is_cjk)

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

    # Fallback: if VAD produced no segments but we have text, create single segment
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
    if segments_raw:
        return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in segments_raw]
    return []


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
    except subprocess.TimeoutExpired:
        raise RuntimeError("Cloud API request timed out after 300 seconds.")

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

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        if "401" in stdout or "Unauthorized" in stdout:
            raise PermissionError("Cloud API authentication failed. Check your API key.")
        raise RuntimeError(f"Cloud API response error: {stdout[:500]}")


def _post_json_via_curl(endpoint: str, auth_header: str, json_body: dict) -> dict:
    """POST JSON body via curl using stdin pipe (avoids Windows file-reference issues)."""
    body_str = json.dumps(json_body, ensure_ascii=False)

    cmd = [
        "curl", "-s", "--max-time", "300",
        "-X", "POST",
        endpoint,
        "-H", f"Authorization: {auth_header}",
        "-H", "Content-Type: application/json",
        "--data-binary", "@-",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, input=body_str, timeout=310)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Cloud API request timed out after 300 seconds.")

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or "curl process failed"
        raise RuntimeError(f"Cloud API request failed: {err}")

    stdout = (proc.stdout or "").strip()
    if not stdout:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"Cloud API returned empty response (rc={proc.returncode})"
            + (f". stderr: {stderr[:300]}" if stderr else "")
        )

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
    display_name = "Cloud API"
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
        model = config.get("model", "whisper-1")

        if not api_url or not api_key:
            raise ValueError("Cloud engine not configured.")

        glossary = kwargs.get("glossary", {})

        if _is_dashscope_url(api_url):
            if self._use_chunk_merger:
                return self._transcribe_dashscope_merged(
                    audio_path, language, api_url, api_key, model, glossary
                )
            else:
                _, segments = _transcribe_dashscope(
                    audio_path, language, api_url, api_key, model, glossary
                )
        else:
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
            is_cjk = _is_mostly_cjk(seg.text)
            max_len = 25 if is_cjk else 42
            sub_texts = _split_text_for_subtitles(seg.text, max_len, is_cjk)

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
