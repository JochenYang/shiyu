"""
Subtitle Assistant - FastAPI Backend Service.
Provides REST API for audio/video transcription and subtitle generation.
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from app.core.config_loader import load_config
from app.core.model_downloader import (
    ensure_model, get_download_status, get_model_dir,
    ensure_punctuation_model, get_punctuation_status,
    get_all_status,
)
from app.core.cloud_engine import load_engine_config, save_engine_config
from app.core.engine_base import BaseTranscriber, TranscribeResult, Segment
from app.core.pipeline import PipelineContext, SubtitlePipeline

logger = logging.getLogger(__name__)


# ── Engine Registry ─────────────────────────────────────────────────
# Maps engine_mode string → transcriber instance
_transcriber_registry: Dict[str, BaseTranscriber] = {}
# 注册失败的引擎列表 (PyInstaller 漏打包等), 在 /engines 返回给前端
_engines_failed: List[Tuple[str, str]] = []  # [(name, reason), ...]  type: ignore[name-defined]  # noqa: F821

# Global state (backward compat)
config = None

# LLM correction status (set by _apply_llm_correction, returned in API response)
# Values: "success" | "failed" | "skipped"
_llm_status: str = "skipped"
_llm_corrected: bool = False  # backward-compat boolean


def _register_default_engines():
    """Register all available ASR engines in the global registry.

    本地引擎: Zipformer-CTC-zh / SenseVoice-Small / Paraformer-large,
    另注册 Cloud (OpenAI 兼容 API). 前端引擎选择器按名字取用.
    """
    global _transcriber_registry, config
    global _engines_failed
    _transcriber_registry = {}
    _engines_failed = []  # 记录注册失败的引擎 (PyInstaller 缺模块时兜底)

    # 1. Local Zipformer (sherpa-onnx, 中文 SOTA)
    try:
        from app.core.zipformer_engine import ZipformerTranscriber
        engines_cfg = (config or {}).get("engines", {}) if config else {}
        if engines_cfg.get("zipformer", {}).get("enabled", True):  # default enabled
            zf_kwargs = ZipformerTranscriber.from_config(engines_cfg.get("zipformer", {}))
            # from_config 返回实例, 提取 kwargs
            from app.core.zipformer_engine import ZipformerEngine
            init_kwargs = ZipformerEngine.init_kwargs_from_config(engines_cfg.get("zipformer", {}))
            register_engine(ZipformerTranscriber, **init_kwargs)
            logger.info("[Engine Registry] Zipformer registered (sherpa-onnx)")
        else:
            logger.info("[Engine Registry] Zipformer disabled (set engines.zipformer.enabled=true in config.yaml)")
    except ImportError as e:
        _engines_failed.append(("zipformer_local", f"import failed: {e}"))
        logger.warning("[Engine Registry] Zipformer unavailable: %s (PyInstaller 可能漏打包 zipformer_engine)", e)
    except Exception as e:
        _engines_failed.append(("zipformer_local", f"load failed: {e}"))
        logger.warning("[Engine Registry] Zipformer load failed: %s", e)

    # 2. Local SenseVoice (多语言备选 zh/en/ja/ko/yue 5 语言)
    try:
        from app.core.sensevoice_engine import SenseVoiceTranscriber
        if config:
            register_engine(SenseVoiceTranscriber,
                model_path=config["model"]["onnx_path"],
                tokens_path=config["model"]["tokens_path"],
                device=config["inference"].get("device", "cpu"),
                audio_cfg=config.get("audio", {}))
        else:
            register_engine(SenseVoiceTranscriber)
    except ImportError:
        logger.info("[Engine Registry] SenseVoice engine unavailable (import failed)")

    # 3. Cloud API (OpenAI 兼容通用, 用户自己填 endpoint + key)
    try:
        from app.core.cloud_engine import CloudTranscriber
        register_engine(CloudTranscriber)
    except ImportError:
        logger.info("[Engine Registry] Cloud engine unavailable (import failed)")

    # 4. Local Paraformer (v2.1 主走, sherpa-onnx Paraformer-large int8, CIF 边界)
    try:
        from app.core.paraformer_engine import ParaformerTranscriber
        from app.core.paraformer_engine import ParaformerEngine
        engines_cfg = (config or {}).get("engines", {}) if config else {}
        if engines_cfg.get("paraformer", {}).get("enabled", True):  # default enabled
            init_kwargs = ParaformerEngine.init_kwargs_from_config(engines_cfg.get("paraformer", {}))
            register_engine(ParaformerTranscriber, **init_kwargs)
            logger.info("[Engine Registry] Paraformer registered (v2.1 primary, sherpa-onnx CIF)")
    except ImportError as e:
        _engines_failed.append(("paraformer_local", f"import failed: {e}"))
        logger.warning("[Engine Registry] Paraformer unavailable: %s (PyInstaller 可能漏打包 paraformer_engine)", e)
    except Exception as e:
        _engines_failed.append(("paraformer_local", f"load failed: {e}"))
        logger.warning("[Engine Registry] Paraformer load failed: %s", e)

    # Bcut (必剪) and Jianying (剪映) free ASR engines were removed
    # in May 2026 because their reverse-engineered API endpoints became
    # permanently unreachable (DNS resolution failure).


def register_engine(transcriber_cls, **init_kwargs):
    """Register a transcriber class into the global registry.

    Creates instance lazily on first use or eagerly if 'name' matches.
    """
    name = transcriber_cls.name
    if name in _transcriber_registry:
        return  # Already registered
    _transcriber_registry[name] = {
        "cls": transcriber_cls,
        "kwargs": init_kwargs,
        "instance": None,
    }


def get_engine(name: str) -> BaseTranscriber:
    """Get or create a transcriber instance by name.

    Falls back to 'sensevoice_local' for backward-compatible 'local',
    and 'cloud' for backward-compatible 'cloud'.
    """
    # Backward compat aliases
    alias_map = {"local": "sensevoice_local"}
    resolved = alias_map.get(name, name)

    entry = _transcriber_registry.get(resolved)
    if entry is None:
        raise HTTPException(400, f"Unknown engine: {name}")

    if entry["instance"] is None:
        entry["instance"] = entry["cls"](**entry["kwargs"])
    return entry["instance"]


def get_available_engines() -> list:
    """Return list of registered engine metadata and availability."""
    results = []
    for name, entry in _transcriber_registry.items():
        inst = entry["instance"]
        if inst is None:
            # Lazy-instantiate — always try, even with no kwargs
            try:
                inst = entry["cls"](**entry["kwargs"])
            except Exception as e:
                inst = None
        if inst is None:
            # Fall back to class-level metadata if no instance
            cls = entry["cls"]
            results.append({
                "name": name,
                "display_name": getattr(cls, "display_name", name),
                "requires_network": getattr(cls, "requires_network", False),
                "is_local": getattr(cls, "is_local", True),
                "available": False,
                "reason": "cannot instantiate (missing config or model)",
            })
            continue
        try:
            avail, reason = inst.is_available()
        except Exception as e:
            avail, reason = False, f"is_available() raised: {e}"
        results.append({
            "name": name,
            "display_name": inst.display_name if hasattr(inst, "display_name") else name,
            "requires_network": inst.requires_network if hasattr(inst, "requires_network") else False,
            "is_local": inst.is_local if hasattr(inst, "is_local") else True,
            "available": avail,
            "reason": reason,
        })
    return results


def _resolve_engines_list(config=None, engine_mode: str = "local") -> list:
    """解析本次转录要使用的 ASR 引擎列表.

    行为:
      1. engine_mode='cloud' → ['cloud'] (OpenAI 兼容云端 API)
      2. engine_mode=zipformer_local/sensevoice_local/paraformer_local → [该引擎]
      3. 其他值 (默认 'local') → ['zipformer_local'],
         若 config["engines"]["sensevoice"]["enabled"] == True 则追加 'sensevoice_local'

    Returns:
        引擎名列表, e.g. ["zipformer_local"] 或 ["zipformer_local", "sensevoice_local"]
    """
    if engine_mode == "cloud":
        return ["cloud"]
    known_local = {"zipformer_local", "sensevoice_local", "paraformer_local"}
    if engine_mode in known_local:
        return [engine_mode]

    engines = ["zipformer_local"]
    if not config:
        return engines
    engines_cfg = config.get("engines", {}) or {}
    if engines_cfg.get("sensevoice", {}).get("enabled", False):
        engines.append("sensevoice_local")
    return engines


# v2.0: _apply_llm_correction removed (LLM layer dropped, owner decision #2)
# Module-level _llm_status / _llm_corrected flags kept for UI backward-compat
# (always "skipped" / False now)


def _extract_audio_for_transcribe(file_path: Path, audio_config: dict = None) -> str:
    """Extract audio from video if needed; ensure we have a 16kHz WAV.

    Args:
        file_path: Path to the upload or local file.
        audio_config: Optional audio config dict.

    Returns:
        Path to the audio file (extracted WAV for video, original for audio).
    """
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    is_video = file_path.suffix.lower() in video_exts

    if is_video:
        from app.core.audio_processor import extract_audio_from_video
        return extract_audio_from_video(str(file_path))
    return str(file_path)


# ── VAD 断句 Segmentation ─────────────────────────────────────────────


# v2.0: _apply_vad_sentence_segmentation removed (clause_processor 模块已删, VAD 走 pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config and register engines. Shutdown: cleanup."""
    global config

    # Load config
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)

    # Register all engines
    _register_default_engines()

    # Try to init local SenseVoice
    model_status = ensure_model()
    logger.info("[SubtitleAssistant] Model status: %s", model_status['status'])

    # Pre-download Silero-VAD model (background, lazy at first call)
    try:
        from app.core.vad_detector import ensure_model as ensure_vad_model
        vad_status = ensure_vad_model()
        if vad_status["status"] == "downloading":
            logger.info("[SubtitleAssistant] Silero-VAD downloading in background...")
        elif vad_status["status"] == "ready":
            logger.info("[SubtitleAssistant] Silero-VAD model ready")
    except Exception:
        pass  # Non-blocking; VAD will download on first use

    # Paraformer/Zipformer 模型由 sherpa-onnx 自带 downloader 管理, 不经过 model_downloader

    # Pre-download Punctuation model (always check; NN engine auto-fallback if missing)
    try:
        punc_status = ensure_punctuation_model()
        logger.info("[SubtitleAssistant] Punctuation model status: %s", punc_status["status"])
    except Exception as e:
        logger.debug("[SubtitleAssistant] Punctuation ensure failed: %s", e)

    if model_status["status"] == "ready":
        try:
            get_engine("sensevoice_local")
            logger.info("[SubtitleAssistant] Model loaded: %s", config['model']['onnx_path'])
        except Exception as e:
            logger.warning("[SubtitleAssistant] Failed to load model: %s", e)
    else:
        logger.info("[SubtitleAssistant] Model not ready yet: %s", model_status['status'])

    logger.info(
        "[SubtitleAssistant] Server ready at http://%s:%s",
        config['server']['host'], config['server']['port'],
    )

    yield

    logger.info("[SubtitleAssistant] Shutting down...")


app = FastAPI(
    title="时语 Shiyu API",
    description="Local audio/video transcription and subtitle generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Tauri/Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["server"]["cors_origins"] if config else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    language: Optional[str] = "auto"
    format: Optional[str] = "srt"


@app.get("/health")
async def health_check():
    log_dir = os.environ.get("SHIYU_LOG_DIR", "")
    engines = get_available_engines()
    local_ready = any(
        e["name"] in ("zipformer_local", "sensevoice_local", "paraformer_local")
        and e["available"]
        for e in engines
    )
    return {
        "status": "ok",
        "model_loaded": local_ready,
        "log_dir": log_dir,
        "engines": engines,
        "model_status": "ready" if local_ready else get_download_status().get("status", "unknown"),
    }


@app.get("/model/status")
async def model_status():
    """Get model download/loading status. Auto-load engine when download completes."""
    download_status = get_download_status()

    # Auto-initialize SenseVoiceTranscriber when model becomes ready
    if download_status["status"] == "ready":
        try:
            get_engine("sensevoice_local")
            logger.info("[SubtitleAssistant] Model loaded after download: %s", config['model']['onnx_path'])
        except Exception as e:
            if "Unknown engine" not in str(e):
                logger.warning("[SubtitleAssistant] Failed to load model after download: %s", e)
                download_status["status"] = "error"
                download_status["error"] = str(e)

    local_entry = _transcriber_registry.get("sensevoice_local")
    engine_ready = local_entry and local_entry["instance"] is not None
    model_path = get_model_dir() / "model.onnx"
    model_size = model_path.stat().st_size if model_path.exists() else 0

    return {
        "status": download_status["status"],
        "progress": download_status["progress"],
        "error": download_status.get("error", ""),
        "engine_loaded": engine_ready,
        "model_path": str(model_path),
        "model_size_mb": round(model_size / (1024 * 1024), 1),
    }


@app.get("/logs/open")
async def open_logs_folder():
    import os
    import sys
    log_dir = os.environ.get("SHIYU_LOG_DIR", "")
    if log_dir and os.path.exists(log_dir):
        try:
            if os.name == 'nt':
                os.startfile(log_dir)
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', log_dir])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', log_dir])
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}
    return {"status": "error", "msg": "Directory not found"}


@app.get("/engines")
async def list_engines():
    """Return list of registered ASR engines with availability status."""
    return {
        "engines": get_available_engines(),
        "failed": _engines_failed,  # 注册失败的引擎 (PyInstaller 漏打包等)
    }


@app.post("/model/download")
async def trigger_model_download():
    """Trigger or retry model download. Resets error state and starts fresh download."""
    status = ensure_model()
    return status


@app.post("/model/download/zipformer")
async def trigger_zipformer_download():
    """Trigger Zipformer-CTC-zh model download.

    Non-blocking. Returns current status. Use /model/status/all to monitor.
    """
    # Zipformer 模型目前由 sherpa-onnx 单独管理 (不经过 model_downloader)
    return {"status": "ready", "progress": 100, "error": "",
            "note": "Zipformer 模型请用 sherpa-onnx 1.10+ 自带的 model downloader"}


@app.post("/model/download/punctuation")
async def trigger_punctuation_download():
    """Trigger PunctuationCtTransformer model download.

    Non-blocking. Returns current status.
    """
    status = ensure_punctuation_model()
    return status


@app.get("/model/status/all")
async def get_all_model_status():
    """统一查询所有模型 (SenseVoice / Zipformer / Punctuation / VAD) 的下载状态。

    Returns:
        {
            "sensevoice": {"status": "...", "progress": ..., "error": "..."},
            "zipformer": {...},
            "punctuation": {...},
            "vad": {...}
        }
    """
    return get_all_status()


@app.get("/model/open")
async def open_model_folder():
    """Open the model directory in file explorer."""
    import os
    import sys
    model_dir = get_model_dir()
    if model_dir.exists():
        try:
            if os.name == 'nt':
                os.startfile(str(model_dir))
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', str(model_dir)])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', str(model_dir)])
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}
    return {"status": "error", "msg": "Model directory not found"}


@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    output_format: str = Form("srt"),
    glossary: str = Form("{}"),
    engine_mode: str = Form("local"),
    subtitle_offset_sec: float = Form(0.0),
):
    """Transcribe audio/video file and return subtitle content.

    Args:
        file: Audio or video file
        language: Target language (auto/zh/en/ja/ko/yue)
        output_format: srt / lrc / ass
        engine_mode: Engine identifier (local/zipformer_local/sensevoice_local/paraformer_local/cloud)
        subtitle_offset_sec: 主人前端滑块, 正=字幕提前, 负=字幕推后

    Returns:
        Plain text subtitle content
    """
    fmt = output_format.lower()
    if fmt not in ("srt", "lrc", "ass"):
        raise HTTPException(400, f"Unsupported format: {output_format}")

    suffix = Path(file.filename).suffix or ".tmp"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / f"input{suffix}"

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            custom_glossary = json.loads(glossary)
        except json.JSONDecodeError as exc:
            logger.warning("[Transcribe] invalid glossary JSON: %s", exc)
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_glossary", "detail": str(exc)},
            )

        audio_path = _extract_audio_for_transcribe(
            tmp_path, config.get("audio") if config else None,
        )

        ctx = PipelineContext(
            audio_path=audio_path,
            audio_cfg=config.get("audio") if config else None,
            engines=_resolve_engines_list(config, engine_mode),
            language=language,
            glossary=custom_glossary,
            output_format=fmt,
            title=file.filename or "Subtitle",
            enable_deterministic_correction=True,
            # v2.0: LLM correction removed (owner decision #2)
            # enable_llm_correction=False,
            subtitle_offset_sec=subtitle_offset_sec,  # 主人前端滑块传过来
        )
        ctx = SubtitlePipeline(ctx).run()

        if not ctx.engines_results:
            raise HTTPException(
                500, "No ASR engine available or all engines failed to transcribe"
            )

        return PlainTextResponse(ctx.output_content)

    except PermissionError as e:
        raise HTTPException(401, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/transcribe/json")
async def transcribe_json(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    glossary: str = Form("{}"),
    engine_mode: str = Form("local"),
    subtitle_offset_sec: float = Form(0.0),
):
    """Transcribe and return raw JSON with segments."""
    suffix = Path(file.filename).suffix or ".tmp"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / f"input{suffix}"

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            custom_glossary = json.loads(glossary)
        except json.JSONDecodeError as exc:
            logger.warning("[Transcribe JSON] invalid glossary JSON: %s", exc)
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_glossary", "detail": str(exc)},
            )

        audio_path = _extract_audio_for_transcribe(
            tmp_path, config.get("audio") if config else None,
        )

        ctx = PipelineContext(
            audio_path=audio_path,
            audio_cfg=config.get("audio") if config else None,
            engines=_resolve_engines_list(config, engine_mode),
            language=language,
            glossary=custom_glossary,
            output_format="srt",
            enable_deterministic_correction=True,
            # v2.0: LLM correction removed (owner decision #2)
            # enable_llm_correction=False,
            subtitle_offset_sec=subtitle_offset_sec,  # 主人前端滑块传过来
        )
        ctx = SubtitlePipeline(ctx).run()

        if not ctx.engines_results:
            raise HTTPException(
                500, "No ASR engine available or all engines failed to transcribe"
            )

        return {
            "filename": file.filename,
            "language": language,
            "duration": ctx.duration,
            "segments": ctx.final_segments or [],
            # v2.0: LLM 字段保留 (UI 仍读取, 永远 "skipped")
            "llm_status": "skipped",
            "llm_corrected": False,
        }

    except PermissionError as e:
        raise HTTPException(401, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/transcribe/local")
async def transcribe_local(
    path: str = Form(...),
    language: str = Form("auto"),
    output_format: str = Form("srt"),
    glossary: str = Form("{}"),
    engine_mode: str = Form("local"),
    subtitle_offset_sec: float = Form(0.0),
):
    """Transcribe a local file by path (for Tauri drag-drop, no upload needed).

    Args:
        path: Absolute path to the audio/video file on disk
        language: Target language (auto/zh/en/ja/ko/yue)
        output_format: srt / lrc / ass
        engine_mode: Engine identifier (local/zipformer_local/sensevoice_local/paraformer_local/cloud)

    Returns:
        JSON with subtitle content and segments for preview
    """
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(400, f"File not found: {path}")

    fmt = output_format.lower()
    if fmt not in ("srt", "lrc", "ass"):
        raise HTTPException(400, f"Unsupported format: {output_format}")

    try:
        try:
            custom_glossary = json.loads(glossary)
        except json.JSONDecodeError as exc:
            logger.warning("[Transcribe Local] invalid glossary JSON: %s", exc)
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_glossary", "detail": str(exc)},
            )

        audio_path = _extract_audio_for_transcribe(
            file_path, config.get("audio") if config else None,
        )

        ctx = PipelineContext(
            audio_path=audio_path,
            audio_cfg=config.get("audio") if config else None,
            engines=_resolve_engines_list(config, engine_mode),
            language=language,
            glossary=custom_glossary,
            output_format=fmt,
            title=file_path.name,
            enable_deterministic_correction=True,
            # v2.0: LLM correction removed (owner decision #2)
            # enable_llm_correction=False,
            subtitle_offset_sec=subtitle_offset_sec,  # 主人前端滑块传过来
        )
        ctx = SubtitlePipeline(ctx).run()

        if not ctx.engines_results:
            raise HTTPException(
                500, "No ASR engine available or all engines failed to transcribe"
            )

        return {
            "content": ctx.output_content,
            "segments": ctx.final_segments or [],
            "filename": file_path.name,
            "duration": ctx.duration,
            "language": language,
            # v2.0: LLM 字段保留 (UI 仍读取, 永远 "skipped")
            "llm_status": "skipped",
            "llm_corrected": False,
        }

    except PermissionError as e:
        raise HTTPException(401, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@app.post("/settings/log-level")
async def set_log_level(level: str = Form("info")):
    """Change backend log level dynamically at runtime."""
    import logging
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    lvl = level_map.get(level.lower())
    if lvl is None:
        raise HTTPException(400, f"Unsupported log level: {level}")
    
    # Set root logger and all uvicorn loggers
    logging.getLogger().setLevel(lvl)
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(name).setLevel(lvl)
    
    return {"status": "ok", "level": level}


@app.get("/settings/engine")
async def get_engine_settings():
    """Return cloud engine config (without api_key plaintext)."""
    cfg = load_engine_config()
    return {
        "api_url": cfg.get("api_url", ""),
        "model": cfg.get("model", "whisper-1"),
        "has_key": bool(cfg.get("api_key")),
        "enabled": bool(cfg.get("enabled", False)),
    }


@app.post("/settings/engine")
async def set_engine_settings(
    api_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("whisper-1"),
    enabled: str = Form("false"),
):
    """Save cloud engine configuration."""
    save_engine_config(
        api_url=api_url,
        api_key=api_key,
        model=model,
        enabled=enabled.lower() == "true",
    )
    return {"status": "ok"}


@app.get("/settings/llm-correction")
async def get_llm_correction_settings():
    """返回 LLM 修正配置（不含 api_key 明文）。"""
    from app.core.cloud_engine import load_llm_correction_config_masked
    return load_llm_correction_config_masked()


@app.post("/settings/llm-correction")
async def set_llm_correction_settings(
    enabled: str = Form("false"),
    provider: str = Form("openai"),
    api_url: str = Form(""),
    api_key: str = Form(""),
    model: str = Form(""),
):
    """保存 LLM 修正配置。"""
    from app.core.cloud_engine import save_llm_correction_config
    save_llm_correction_config(
        enabled=enabled.lower() == "true",
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        model=model,
    )
    return {"status": "ok"}


@app.post("/transcribe/correct")
async def correct_transcription(body: dict):
    """Apply determinisic + optional LLM correction to transcription segments.

    Accepts JSON: {"segments": [...], "engine_mode": "local", "subtitle_offset_sec": -2.0}
    Optionally accepts "audio_path" for VAD-based sentence segmentation.

    Two-layer architecture (delegated to SubtitlePipeline):
      Layer 1: Deterministic correction (always runs, zero cost)
      Layer 2: Optional LLM enhancement (on top of Layer 1 result)
      Stage 2: Optional VAD-based sentence segmentation (if audio_path provided)

    Does NOT re-transcribe — only fixes text recognition errors.

    Three-red-lines principle:
    - Don't change timestamp (start/end fields)
    - Don't add/remove segments
    - Don't rewrite semantics, only fix transcription errors
    """
    segments = body.get("segments", [])
    audio_path: Optional[str] = body.get("audio_path")
    subtitle_offset_sec = float(body.get("subtitle_offset_sec", 0.0))
    if not segments:
        return {"segments": [], "llm_corrected": False}

    ctx = PipelineContext(
        skip_asr=True,
        audio_path=audio_path or "",
        # 深拷贝入参: pipeline 的 glossary/deterministic 会修改段文本,
        # 若与 body 共享 dict 引用, 后续 changed 比较会因入参已被改而漏报
        raw_segments=[dict(s) for s in segments],
        engines_results={},  # /correct endpoint doesn't run ASR
        enable_deterministic_correction=True,
        enable_itn=True,
        enable_punctuation=False,
        output_format="srt",
        subtitle_offset_sec=subtitle_offset_sec,
    )

    try:
        ctx = SubtitlePipeline(ctx).run()
    except Exception as e:
        logger.warning("[Correct] Pipeline failed, fallback to original: %s", e)
        return {"segments": segments, "llm_corrected": False}

    # v2.0: segment_merger + llm_guardrail removed
    # 直接用 corrected_segments 当 final
    final_source = ctx.final_segments or segments
    final = final_source

    # 计算是否实际发生了文本变化 (deterministic only — v2.0 无 LLM)
    final_min_len = min(len(final), len(segments))
    changed = any(
        s.get("text") != o.get("text")
        for s, o in zip(final[:final_min_len], segments[:final_min_len])
    )

    logger.info("[Pipeline Correct] %d segments → %d corrected",
                len(segments), len(final))

    return {"segments": final, "llm_corrected": False, "llm_status": "skipped", "changed": changed}


if __name__ == "__main__":
    import uvicorn
    server_cfg = config["server"] if config else {"host": "127.0.0.1", "port": 11235}
    uvicorn.run(app, host=server_cfg["host"], port=server_cfg["port"])
