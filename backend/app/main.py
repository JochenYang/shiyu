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
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from app.core.config_loader import load_config
from app.core import model_catalog, model_download
from app.core.cloud_engine import load_engine_config, save_engine_config
from app.core.engine_base import BaseTranscriber
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
    """按模型清单注册本地引擎, 另加 Cloud (OpenAI 兼容 API)。

    本地引擎全部走 sherpa_engine 的同一推理路径; 引擎可用性取决于模型清单
    中对应文件是否齐备, 未下载的模型不会出现在可选列表中。
    """
    global _transcriber_registry
    global _engines_failed
    _transcriber_registry = {}
    _engines_failed = []  # 记录注册失败的引擎 (PyInstaller 缺模块时兜底)

    from app.core.sherpa_engine import SherpaOfflineEngine

    engines_cfg = (config or {}).get("engines", {}) if config else {}
    for spec in model_catalog.CATALOG["models"]:
        model_id = spec["id"]
        engine_name = spec["engine_name"]
        try:
            cfg = engines_cfg.get(engine_name, {}) or {}
            if not cfg.get("enabled", True):
                logger.info("[Engine Registry] %s 已禁用 (config.yaml)", engine_name)
                continue
            instance = SherpaOfflineEngine.from_config(model_id, cfg)
            _transcriber_registry[engine_name] = {
                "cls": SherpaOfflineEngine,
                "kwargs": {"model_id": model_id, "num_threads": cfg.get("num_threads", 4),
                           "ctc_delay_sec": cfg.get("ctc_delay_sec", 0.15)},
                "instance": instance,
            }
            logger.info("[Engine Registry] %s 已注册 (%s)", engine_name, spec["name"])
        except Exception as e:
            _engines_failed.append((engine_name, f"load failed: {e}"))
            logger.warning("[Engine Registry] %s 注册失败: %s", engine_name, e)

    # Cloud API (OpenAI 兼容通用): 多配置并存, 每个已配置项注册一个实例。
    # 激活项用 "cloud" (旧路由 engine_mode='cloud' 继续命中), 其余用 "cloud:<id>"。
    try:
        from app.core.cloud_engine import (
            CloudTranscriber,
            list_engine_configs,
            load_engine_config,
        )

        active_id = load_engine_config().get("id")
        items = list_engine_configs()
        if not items:
            # 一个配置都没有时也注册默认实例, 保证 "cloud" 引擎名始终存在
            register_engine(CloudTranscriber)
        for item in items:
            eid = item.get("id")
            if not eid:
                continue
            engine_name = "cloud" if eid == active_id else f"cloud:{eid}"
            display = item.get("display_name") or item.get("model") or "云模型"
            try:
                instance = CloudTranscriber(engine_id=eid)
                _transcriber_registry[engine_name] = {
                    "cls": CloudTranscriber,
                    "kwargs": {"engine_id": eid},
                    "instance": instance,
                }
                # 实例级 display_name: 下拉里同时显示多个云配置时能区分
                try:
                    instance.display_name = f"云模型 · {display}"
                except Exception:
                    pass
                logger.info("[Engine Registry] %s 已注册 (%s)", engine_name, display)
            except Exception as e:
                _engines_failed.append((engine_name, f"load failed: {e}"))
                logger.warning("[Engine Registry] %s 注册失败: %s", engine_name, e)
    except ImportError:
        logger.info("[Engine Registry] Cloud engine unavailable (import failed)")

def register_engine(transcriber_cls, **init_kwargs):
    """Register a transcriber class into the global registry."""
    name = transcriber_cls.name
    if name in _transcriber_registry:
        return  # Already registered
    _transcriber_registry[name] = {
        "cls": transcriber_cls,
        "kwargs": init_kwargs,
        "instance": None,
    }


def get_engine(name: str) -> BaseTranscriber:
    """Get or create a transcriber instance by name."""
    entry = _transcriber_registry.get(name)
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
            except Exception:
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
    """解析本次转录要使用的 ASR 引擎列表。

    行为:
      1. ``engine_mode='cloud'`` → ``['cloud']`` (OpenAI 兼容云端 API)
      2. ``engine_mode`` 命中已注册的本地引擎 → ``[该引擎]``
      3. 其他值 (默认 'local') → 清单默认模型对应的引擎

    Returns:
        引擎名列表, 例如 ``['zipformer_local']``。
    """
    if engine_mode == "cloud":
        return ["cloud"]
    if engine_mode in _transcriber_registry:
        return [engine_mode]

    default_engine = model_catalog.model_spec(model_catalog.DEFAULT_MODEL_ID)["engine_name"]
    if default_engine in _transcriber_registry:
        return [default_engine]
    # 默认模型尚未注册 (未下载): 退回第一个已注册的本地引擎, 没有则空列表
    return [name for name in _transcriber_registry if name != "cloud"]


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

    # Register all engines (按模型清单)
    _register_default_engines()

    # 只报告已安装的模型, 绝不在启动时自动下载
    for entry in model_catalog.availability():
        if entry["installed"]:
            logger.info("[SubtitleAssistant] 模型就绪: %s", entry["id"])
        else:
            logger.info("[SubtitleAssistant] 模型未安装: %s (%s)", entry["id"], entry["reason"])

    # 端口以实际绑定为准: run_server.py 可用 --host/--port 覆盖配置,
    # 若此处只读配置, 日志会报出与真实监听地址不符的端口, 排查时误导。
    logger.info(
        "[SubtitleAssistant] Server ready at http://%s:%s",
        os.environ.get("SHIYU_BOUND_HOST", config['server']['host']),
        os.environ.get("SHIYU_BOUND_PORT", str(config['server']['port'])),
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
        e["name"] != "cloud" and e["available"]
        for e in engines
    )
    return {
        "status": "ok",
        "model_loaded": local_ready,
        "log_dir": log_dir,
        "engines": engines,
        "model_status": "ready" if local_ready else "not_installed",
    }


@app.get("/logs/open")
async def open_logs_folder():
    """在文件管理器中打开日志目录。"""
    import os
    import sys

    log_dir = os.environ.get("SHIYU_LOG_DIR", "")
    if log_dir and os.path.exists(log_dir):
        try:
            if os.name == "nt":
                os.startfile(log_dir)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", log_dir])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", log_dir])
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}
    return {"status": "error", "msg": "Directory not found"}


@app.get("/engines")
async def list_engines():
    """返回已注册 ASR 引擎及其可用性。"""
    return {
        "engines": get_available_engines(),
        "failed": _engines_failed,  # 注册失败的引擎 (PyInstaller 漏打包等)
    }


@app.get("/models")
async def list_models():
    """列出模型清单与安装状态 (只读, 绝不触发下载)。

    Returns:
        ``{"models": [...], "defaultModelID": str, "job": {...}|null, "jobs": [...]}``
    """
    return {
        "models": model_catalog.availability(),
        "defaultModelID": model_catalog.default_model_id(),
        "job": model_download.active_job(),
        "jobs": model_download.get_jobs(),
    }


@app.post("/models/{model_id}/default")
async def set_default_model(model_id: str):
    """设置默认识别模型 (持久化到用户配置)。

    与「云端设为当前使用」对应: 本地模型的默认项决定引擎下拉的初始
    选中与 engine_mode 缺省值。未安装的模型也允许设 (下载后即生效),
    但未知 id 与对齐器会被拒绝。
    """
    try:
        model_catalog.set_default_model(model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "ok", "defaultModelID": model_id}

@app.post("/models/{model_id}/install")
async def install_model(model_id: str, source: str = Form("auto")):
    """开始下载并安装模型 (非阻塞, 返回任务状态)。

    Args:
        model_id: 清单中的模型 id。
        source: ``auto`` (国内镜像优先) 或 ``official`` (仅官方源)。
    """
    try:
        return model_download.start_install(model_id, source)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/models/{model_id}/remove")
async def remove_model(model_id: str):
    """移除模型 (删除其全部文件, 需显式调用)。"""
    try:
        return model_download.start_remove(model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/models/jobs/{job_id}/cancel")
async def cancel_model_job(job_id: str):
    """取消一个运行中的下载任务 (残余保留, 重试可继续)。"""
    try:
        return model_download.cancel_job(job_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/models/jobs")
async def list_model_jobs():
    """列出所有下载任务状态 (新任务在前)。"""
    return {"jobs": model_download.get_jobs(), "job": model_download.active_job()}


@app.get("/model/open")
async def open_model_folder():
    """Open the model directory in file explorer."""
    import os
    import sys

    model_dir = model_catalog.model_root()
    if model_dir.exists():
        try:
            if os.name == "nt":
                os.startfile(str(model_dir))
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", str(model_dir)])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(model_dir)])
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
        engine_mode: 引擎标识 (local/zipformer_local/sensevoice_local/firered_local/cloud)
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
        engine_mode: 引擎标识 (local/zipformer_local/sensevoice_local/firered_local/cloud)

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
        "request_mode": cfg.get("request_mode", "whisper"),
        "display_name": cfg.get("display_name", ""),
    }


@app.post("/settings/engine")
async def set_engine_settings(
    api_url: str = Form(...),
    # 默认空串而不是必填: 前端用掩码回显密钥, 用户不改动时会传空值;
    # 空值保留磁盘上的旧密钥, 不该因此报 422。
    api_key: str = Form(""),
    model: str = Form("whisper-1"),
    enabled: str = Form("false"),
    request_mode: str = Form(""),
    display_name: str = Form(""),
):
    """Save cloud engine configuration.

    ``request_mode`` 与 ``display_name`` 可省略 (旧前端仍能工作);
    省略时保留磁盘上的原值, 不把已有配置抹掉。
    """
    save_engine_config(
        api_url=api_url,
        api_key=api_key,
        model=model,
        enabled=enabled.lower() == "true",
        request_mode=request_mode or None,
        display_name=display_name or None,
    )
    return {"status": "ok"}


# ── 云端多配置 (v2.4): engines 列表 + active_engine ─────────────────


def _masked_engine(item: Dict[str, Any]) -> Dict[str, Any]:
    """配置项脱敏: 不回传 api_key 明文, 用 has_key 标记是否存在。"""
    out = {k: v for k, v in item.items() if k != "api_key"}
    out["has_key"] = bool(item.get("api_key"))
    return out


@app.get("/settings/engines")
async def list_engine_settings():
    """全部云端配置 (脱敏) 与激活 id。"""
    from app.core.cloud_engine import list_engine_configs, load_engine_config

    items = list_engine_configs()
    active = load_engine_config().get("id")
    return {"engines": [_masked_engine(i) for i in items], "active": active}


@app.post("/settings/engines")
async def save_engine_settings(body: dict):
    """整体保存云端配置列表与激活项。

    Body::
        {"engines": [{id?, api_url, api_key?, model, enabled,
                      request_mode?, display_name?, auth_header?}, ...],
         "active": "<id>"}

    ``api_key`` 传空字符串保留该项磁盘上的旧密钥 (前端掩码回显)。
    """
    from app.core.cloud_engine import list_engine_configs, save_engine_configs

    items_in = body.get("engines")
    if not isinstance(items_in, list):
        raise HTTPException(422, "engines 必须是列表")

    # 旧密钥保护: 前端掩码回显, 空 api_key 表示用户没改
    old_by_id = {i.get("id"): i for i in list_engine_configs() if i.get("id")}
    clean: list = []
    for raw in items_in:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        eid = entry.get("id")
        key = (entry.get("api_key") or "").strip()
        if not key and eid and eid in old_by_id:
            entry["api_key"] = old_by_id[eid].get("api_key", "")
        else:
            entry["api_key"] = key
        clean.append(entry)

    active = body.get("active")
    save_engine_configs(clean, active)

    # 配置变了, 云端引擎注册表要重建 (激活项/实例绑定都可能变)
    _register_default_engines()
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
