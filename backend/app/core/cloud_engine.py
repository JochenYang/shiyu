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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import requests

from .audio_processor import find_ffmpeg
from . import text_splitter

CONFIG_FILE = Path.home() / ".shiyu" / "config.json"

# ── 内置云端 provider 清单 (v2.5) ────────────────────────────────
#
# 用户只需选 provider + 填 KEY; base_url / request_mode / 默认模型
# 全部预填 (可改)。这五家实测都在 OpenAI 兼容协议上提供 ASR 模型:
#   * whisper 模式 = POST {base}/audio/transcriptions (multipart)
#   * chat 模式    = POST {base}/chat/completions + input_audio(base64)
# 豆包/腾讯/讯飞的语音是独立签名或 WebSocket 协议, 不在此清单,
# 用「自定义」入口 (若该厂商有兼容网关) 或等待专用适配器。
#
# auth_header 仅 MiMo 需要 (认证头是 api-key: <key> 而非 Bearer);
# 其余留空表示默认 Authorization: Bearer <key>。
BUILTIN_PROVIDERS = [
    {
        "id": "openai",
        "name": "OpenAI",
        "api_url": "https://api.openai.com/v1",
        "request_mode": "whisper",
        "models": ["whisper-1", "gpt-4o-transcribe"],
        "auth_header": None,
        "note": "支持 25MB 音频; whisper-1 有 verbose_json 时间戳",
    },
    {
        "id": "groq",
        "name": "Groq",
        "api_url": "https://api.groq.com/openai/v1",
        "request_mode": "whisper",
        "models": ["whisper-large-v3-turbo", "whisper-large-v3"],
        "auth_header": None,
        "note": "10MB 上限; 速度快, 有免费额度",
    },
    {
        "id": "siliconflow",
        "name": "SiliconFlow 硅基流动",
        "api_url": "https://api.siliconflow.cn/v1",
        "request_mode": "whisper",
        "models": ["FunAudioLLM/SenseVoiceSmall"],
        "auth_header": None,
        "note": "10MB 上限; 国内直连, SenseVoice 中文效果好",
    },
    {
        "id": "dashscope",
        "name": "阿里云 DashScope (Qwen-ASR)",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "request_mode": "chat_completions",
        "models": ["qwen3-asr-flash"],
        "auth_header": None,
        "note": "单次 5 分钟音频; base64 走 chat 接口, 无时间戳",
    },
    {
        "id": "mimo",
        "name": "小米 MiMo",
        "api_url": "https://api.xiaomimimo.com/v1",
        "request_mode": "chat_completions",
        "models": ["mimo-v2.5-asr"],
        "auth_header": {"name": "api-key", "value_template": "{api_key}"},
        "note": "认证头特殊 (api-key), 已自动配置",
    },
    {
        "id": "doubao",
        "name": "豆包录音文件识别（火山引擎）",
        "api_url": "https://openspeech.bytedance.com/api/v3/auc/bigmodel",
        # 豆包不用 model 字段区分版本: 模型版本由请求头 X-Api-Resource-Id
        # 表达 (官方文档: 1.0=volc.bigasr.auc, 2.0=volc.seedasr.auc);
        # 请求体的 model_name 固定为协议常量 bigmodel。
        # 这里把两个版本列为可选"模型", 后端把它映射成 resource_id。
        "request_mode": "doubao",
        "models": [
            {"id": "doubao-asr-2.0", "name": "录音文件识别 2.0",
             "resource_id": "volc.seedasr.auc",
             "note": "准确率更高, 推荐"},
            {"id": "doubao-asr-1.0", "name": "录音文件识别 1.0",
             "resource_id": "volc.bigasr.auc",
             "note": "兼容旧开通服务"},
        ],
        "default_model_id": "doubao-asr-2.0",
        "auth_header": None,
        "note": "逐词毫秒时间戳; 60s 块提交即计费, 只需语音服务 KEY",
    },
    {
        "id": "custom",
        "name": "自定义",
        "api_url": "",
        "request_mode": "whisper",
        "models": [],
        "auth_header": None,
        "note": "任何 OpenAI 兼容端点; 高级项可改认证头",
    },
]

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
    """Load the ACTIVE cloud engine config from ~/.shiyu/config.json.

    支持多配置并存 (v2.4):
      * ``engines``: 配置列表, 每项 {id, api_url, api_key, model, enabled,
        request_mode, display_name, auth_header}
      * ``active_engine``: 当前激活配置的 id
      * ``engine``: 历史单配置字段, 保留为激活项的镜像 (旧代码/旧前端兼容)

    读取顺序: active_engine 命中 engines 列表 → 该项; 否则回落到
    历史 ``engine`` 字段; 都没有 → 默认空配置。返回激活项的**全部**
    字段 (不只已知键) —— 白名单式读取会把 ``request_mode`` 等
    静默丢掉, 而转录路由正靠它决定走 whisper 还是 chat_completions。
    """
    defaults = {"api_url": "", "api_key": "", "model": "", "enabled": False}
    if not CONFIG_FILE.exists():
        return dict(defaults)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return dict(defaults)

    engines = data.get("engines")
    active_id = data.get("active_engine")
    if isinstance(engines, list) and active_id:
        for item in engines:
            if isinstance(item, dict) and item.get("id") == active_id:
                result = dict(defaults)
                result.update(_infer_provider(item))
                return result

    # 回落: 历史 engine 字段 (旧配置文件没有 engines 列表)
    cfg = data.get("engine")
    if not isinstance(cfg, dict):
        return dict(defaults)
    result = dict(defaults)
    result.update(_infer_provider(cfg))
    return result

def _infer_provider(item: Dict[str, Any]) -> Dict[str, Any]:
    """补出配置里缺失的 provider 字段。

    早期版本的配置只有 api_url 没有 provider, 而引擎路由靠 provider
    判断: 豆包缺了它就会被当成通用 OpenAI 适配器, 每次调用都失败。
    这里按 api_url 的主机名反查内置清单补上。

    已知内置 provider 时连 request_mode 一并纠正 —— 否则豆包仍会以
    chat/completions 发出请求。custom 不动, 它的模式由用户决定。
    """
    if item.get("provider"):
        return item
    api_url = (item.get("api_url") or "").strip()
    if not api_url:
        return item

    def host(url: str) -> str:
        # 比主机名而不是整串: 用户可能补了 /v1 之类的后缀
        return url.split("//", 1)[-1].split("/", 1)[0].lower()

    for provider in BUILTIN_PROVIDERS:
        if provider.get("id") == "custom":
            continue
        base = (provider.get("api_url") or "").strip()
        if base and host(base) == host(api_url):
            filled = dict(item)
            filled["provider"] = provider["id"]
            if provider.get("request_mode"):
                filled["request_mode"] = provider["request_mode"]
            return filled
    return item


def list_engine_configs() -> List[Dict[str, Any]]:
    """全部云端配置 (含未启用的), 每项带 id。

    首次迁移: 磁盘上只有历史 ``engine`` 字段时, 把它包装成列表第一项,
    不改写磁盘 (写入时才落 engines), 保证只读路径无副作用。
    """
    if not CONFIG_FILE.exists():
        return []
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    engines = data.get("engines")
    items: List[Dict[str, Any]] = []
    if isinstance(engines, list):
        for item in engines:
            if isinstance(item, dict) and item.get("id"):
                items.append(_infer_provider(dict(item)))
    if not items:
        cfg = data.get("engine")
        if isinstance(cfg, dict) and (cfg.get("api_url") or cfg.get("api_key") or cfg.get("model")):
            legacy = _infer_provider(dict(cfg))
            legacy["id"] = "cloud"
            items.append(legacy)
    return items


def _new_engine_id() -> str:
    import uuid

    return f"cloud-{uuid.uuid4().hex[:8]}"


def save_engine_configs(items: List[Dict[str, Any]], active_id: Optional[str] = None) -> None:
    """整体写入云端配置列表与激活项, 并把激活项镜像到历史 ``engine`` 字段。

    旧代码 (transcribe_cloud 等) 只读 ``engine``; 镜像保证它们无需改动
    也始终读到激活配置。镜像只复制已知字段, 不携带 id。
    """
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}

    clean: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        if not entry.get("id"):
            entry["id"] = _new_engine_id()
        clean.append(entry)

    data["engines"] = clean
    if active_id is None and clean:
        active_id = clean[0]["id"]
    if active_id and not any(e.get("id") == active_id for e in clean):
        active_id = clean[0]["id"] if clean else None
    data["active_engine"] = active_id

    # 镜像激活项到历史 engine 字段 (旧读取路径兼容)
    mirror: Dict[str, Any] = {}
    if active_id:
        for e in clean:
            if e.get("id") == active_id:
                mirror = {k: v for k, v in e.items() if k != "id"}
                break
    if mirror:
        data["engine"] = mirror

    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(CONFIG_FILE)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass  # Windows may not support chmod

def save_engine_config(
    api_url: str,
    api_key: str,
    model: str,
    enabled: bool,
    request_mode: Optional[str] = None,
    display_name: Optional[str] = None,
) -> None:
    """Save cloud engine config to ~/.shiyu/config.json

    只覆盖表单传回的字段, 保留磁盘上已有的其它字段
    (``request_mode``/``display_name``/``auth_header`` 等)。
    整个字典直接赋值会把用户在界面上看不到的字段全部抹掉。

    ``request_mode`` 与 ``display_name`` 传 None 时保留原值;
    传空字符串时同样保留原值 (避免旧前端把配置清空)。
    """
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    existing = data.get("engine")
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update({"api_url": api_url, "model": model, "enabled": enabled})
    # 空 key 不覆盖已有密钥, 与 LLM 修正配置保持一致
    merged["api_key"] = api_key if api_key else merged.get("api_key", "")
    if request_mode:
        merged["request_mode"] = request_mode
    if display_name:
        merged["display_name"] = display_name
    data["engine"] = merged
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
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    existing = data.get("llm_correction")
    existing = existing if isinstance(existing, dict) else {}
    resolved_key = api_key if api_key else existing.get("api_key", "")
    merged = dict(existing)
    merged.update({
        "enabled": enabled,
        "provider": provider,
        "api_url": api_url,
        "api_key": resolved_key,
        "model": model,
    })
    data["llm_correction"] = merged
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
            # 自定义 glossary 由 pipeline 层统一应用 (避免引擎层双重替换)
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

        # 自定义 glossary 由 pipeline 层统一应用 (避免引擎层双重替换)
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


class CloudTranscriber(BaseTranscriber):
    """Adapter wrapping a cloud engine config as a BaseTranscriber.

    支持多配置并存 (v2.4): 每个配置注册为一个引擎实例,
    引擎名 = ``cloud`` (激活项) 或 ``cloud:<id>`` (其余),
    ``transcribe`` 读绑定配置而不是全局激活项。
    """

    name = "cloud"
    display_name = "云模型"
    requires_network = True
    is_local = False

    def __init__(self, **kwargs):
        self.engine_id = kwargs.get("engine_id")  # None → 跟随全局激活项
        # Backward-compat: accept legacy use_chunk_merger kwarg, but ignore
        # Cloud engines return full transcripts with timestamps built-in.

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        **kwargs,
    ) -> TranscribeResult:
        config = self._bound_config()
        # 从绑定配置里取出本次调用要用的字段。
        # (这几行曾被一次编辑吞掉, 导致 transcribe 一调用就 NameError:
        #  name 'api_url' is not defined)
        api_url = (config.get("api_url") or "").rstrip("/")
        api_key = config.get("api_key") or ""
        model = (config.get("model") or "").strip()
        request_mode = (config.get("request_mode") or "whisper").lower().strip()

        if not api_url or not api_key:
            raise ValueError("Cloud engine not configured.")

        glossary = kwargs.get("glossary", {})

        if request_mode == "chat_completions":
            # OpenAI-compatible chat completions + input_audio (DashScope / MiMo / Volcengine proxy)
            _, segments = _transcribe_dashscope(
                audio_path, language, api_url, api_key, model, glossary
            )
        elif request_mode == "whisper":
            _, segments = _transcribe_whisper(
                audio_path, language, api_url, api_key, model, glossary
            )
        else:
            raise ValueError(f"Unknown request_mode: {request_mode!r}")

        typed = [Segment(start=s["start"], end=s["end"], text=s["text"])
                 for s in segments]
        return TranscribeResult(segments=typed)

    def _bound_config(self) -> Dict[str, Any]:
        """本实例绑定的配置: 有 engine_id 时取该项, 否则取全局激活项。"""
        if self.engine_id:
            for item in list_engine_configs():
                if item.get("id") == self.engine_id:
                    result = {"api_url": "", "api_key": "", "model": "", "enabled": False}
                    result.update(item)
                    return result
            raise ValueError(f"Cloud engine config not found: {self.engine_id}")
        return load_engine_config()

    def is_available(self) -> Tuple[bool, str]:
        """Check if cloud API is configured."""
        config = self._bound_config()
        if config.get("enabled") and config.get("api_url") and config.get("api_key"):
            return True, ""
        return False, "Cloud API not configured (api_url + api_key required)"
