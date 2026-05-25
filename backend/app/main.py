"""
Subtitle Assistant - FastAPI Backend Service.
Provides REST API for audio/video transcription and subtitle generation.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.core.config_loader import load_config
from app.core.audio_processor import preprocess_audio
from app.core.model_downloader import ensure_model, get_download_status, get_model_dir
from app.core.cloud_engine import load_engine_config, save_engine_config
from app.core.engine_base import BaseTranscriber, TranscribeResult, Segment


# ── Engine Registry ─────────────────────────────────────────────────
# Maps engine_mode string → transcriber instance
_transcriber_registry: Dict[str, BaseTranscriber] = {}

# Global state (backward compat)
config = None

# LLM correction applied flag (set by _apply_llm_correction, returned in API response)
_llm_corrected: bool = False


def _register_default_engines():
    """Register all available ASR engines in the global registry."""
    global _transcriber_registry, config
    _transcriber_registry = {}

    # 1. Local SenseVoice
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
        pass

    # 2. Cloud API
    try:
        from app.core.cloud_engine import CloudTranscriber
        register_engine(CloudTranscriber)
    except ImportError:
        pass

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
        inst = entry["instance"] or entry["cls"]()
        avail, reason = inst.is_available()
        results.append({
            "name": name,
            "display_name": inst.display_name if hasattr(inst, "display_name") else name,
            "requires_network": inst.requires_network if hasattr(inst, "requires_network") else False,
            "is_local": inst.is_local if hasattr(inst, "is_local") else True,
            "available": avail,
            "reason": reason,
        })
    return results


def _apply_llm_correction(segments: list) -> list:
    """If LLM correction is enabled, correct text recognition errors in segments.

    Sets global flag _llm_corrected for UI feedback.
    Three-red-lines principle:
    - Don't change timestamp (start/end fields)
    - Don't add/remove segments
    - Don't rewrite semantics, only fix transcription errors
    """
    global _llm_corrected
    _llm_corrected = False
    from app.core.cloud_engine import load_llm_correction_config
    cfg = load_llm_correction_config()
    if not cfg.get("enabled"):
        return segments
    try:
        from app.core.llm_corrector import create_corrector, CorrectionConfig
        corr_config = CorrectionConfig(
            enabled=True,
            provider=cfg.get("provider", "openai"),
            api_url=cfg.get("api_url", ""),
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", ""),
        )
        corrector = create_corrector(corr_config)
        if corrector:
            print(f"[LLM Correction] Correcting {len(segments)} segments via {corr_config.provider}")
            corrected = corrector.correct(segments)
            # Sort by start time
            corrected.sort(key=lambda s: s.get("start", 0))
            # Only mark corrected if text actually changed
            _llm_corrected = any(
                seg["text"] != orig["text"]
                for seg, orig in zip(corrected, segments)
            )
            if _llm_corrected:
                print(f"[LLM Correction] Done — {sum(1 for s,o in zip(corrected,segments) if s['text']!=o['text'])}/{len(segments)} segments changed")
            else:
                print(f"[LLM Correction] Done — no changes (correction returned original)")
            return corrected
    except Exception as e:
        print(f"[LLM Correction] Failed: {e}")
    return segments


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
    print(f"[SubtitleAssistant] Model status: {model_status['status']}")

    if model_status["status"] == "ready":
        try:
            get_engine("sensevoice_local")
            print(f"[SubtitleAssistant] Model loaded: {config['model']['onnx_path']}")
        except Exception as e:
            print(f"[SubtitleAssistant] Failed to load model: {e}")
    else:
        print(f"[SubtitleAssistant] Model not ready yet: {model_status['status']}")

    print(f"[SubtitleAssistant] Server ready at http://{config['server']['host']}:{config['server']['port']}")

    yield

    print("[SubtitleAssistant] Shutting down...")


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
    local_ready = any(e["name"] == "sensevoice_local" and e["available"] for e in engines)
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
            print(f"[SubtitleAssistant] Model loaded after download: {config['model']['onnx_path']}")
        except Exception as e:
            if "Unknown engine" not in str(e):
                print(f"[SubtitleAssistant] Failed to load model after download: {e}")
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
    return {"engines": get_available_engines()}


@app.post("/model/download")
async def trigger_model_download():
    """Trigger or retry model download. Resets error state and starts fresh download."""
    status = ensure_model()
    return status


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
):
    """Transcribe audio/video file and return subtitle content.

    Args:
        file: Audio or video file
        language: Target language (auto/zh/en/ja/ko/yue)
        output_format: srt / lrc / ass
        engine_mode: Engine identifier (local, cloud)

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
        except json.JSONDecodeError:
            custom_glossary = {}

        # Resolve engine and transcribe
        transcriber = get_engine(engine_mode)
        audio_path = _extract_audio_for_transcribe(tmp_path, config.get("audio") if config else None)
        result = transcriber.transcribe(audio_path, language=language, glossary=custom_glossary)

        # Format output
        segments = result.to_segments_list()
        from app.core.subtitle_formatter import (
            generate_srt, generate_lrc, generate_ass,
            segments_from_transcription,
        )
        srt_segments = segments_from_transcription(segments)

        if fmt == "srt":
            content = generate_srt(srt_segments)
        elif fmt == "lrc":
            content = generate_lrc(srt_segments)
        else:
            content = generate_ass(srt_segments, title=file.filename)

        return PlainTextResponse(content)

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
        except json.JSONDecodeError:
            custom_glossary = {}

        # Resolve engine and transcribe
        transcriber = get_engine(engine_mode)
        audio_path = _extract_audio_for_transcribe(tmp_path, config.get("audio") if config else None)
        result = transcriber.transcribe(audio_path, language=language, glossary=custom_glossary)

        segments = result.to_segments_list()

        # Calculate actual duration for reference
        import librosa
        try:
            total_duration = librosa.get_duration(path=audio_path)
        except Exception:
            total_duration = segments[-1]["end"] if segments else 0

        return {"filename": file.filename, "language": language, "duration": total_duration, "segments": segments}

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
):
    """Transcribe a local file by path (for Tauri drag-drop, no upload needed).

    Args:
        path: Absolute path to the audio/video file on disk
        language: Target language (auto/zh/en/ja/ko/yue)
        output_format: srt / lrc / ass
        engine_mode: Engine identifier (local, cloud)

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
        except json.JSONDecodeError:
            custom_glossary = {}

        # Resolve engine and transcribe
        transcriber = get_engine(engine_mode)
        audio_path = _extract_audio_for_transcribe(file_path, config.get("audio") if config else None)
        result = transcriber.transcribe(audio_path, language=language, glossary=custom_glossary)

        segments = result.to_segments_list()

        # Calculate actual duration for reference
        import librosa
        try:
            total_duration = librosa.get_duration(path=audio_path)
        except Exception:
            total_duration = segments[-1]["end"] if segments else 0

        # Format output if requested
        from app.core.subtitle_formatter import (
            generate_srt, generate_lrc, generate_ass,
            segments_from_transcription,
        )
        srt_segments = segments_from_transcription(segments)

        if fmt == "srt":
            content = generate_srt(srt_segments)
        elif fmt == "lrc":
            content = generate_lrc(srt_segments)
        else:
            content = generate_ass(srt_segments, title=file_path.name)

        return {
            "content": content,
            "segments": segments,
            "filename": file_path.name,
            "duration": total_duration,
            "language": language,
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

    Accepts JSON: {"segments": [...], "engine_mode": "local"}

    Two-layer architecture:
      Layer 1: Deterministic correction (always runs, zero cost)
      Layer 2: Optional LLM enhancement (on top of Layer 1 result)

    Does NOT re-transcribe — only fixes text recognition errors.

    Three-red-lines principle:
    - Don't change timestamp (start/end fields)
    - Don't add/remove segments
    - Don't rewrite semantics, only fix transcription errors
    """
    segments = body.get("segments", [])
    if not segments:
        return {"segments": [], "llm_corrected": False}

    # ── Layer 1: Deterministic correction (always runs) ──
    from app.core.asr_corrector import deterministic_correct
    layer1 = deterministic_correct(segments)
    layer1.sort(key=lambda s: s.get("start", 0))

    # ── Layer 2: Optional LLM enhancement ──
    from app.core.cloud_engine import load_llm_correction_config
    cfg = load_llm_correction_config()

    if not cfg.get("enabled"):
        # LLM disabled → use Layer 1 result directly
        from app.core.segment_merger import split_long_segments
        final = split_long_segments(layer1)
        changed = any(
            s["text"] != o["text"] for s, o in zip(layer1, segments)
        )
        print(f"[Deterministic] Corrected {len(layer1)} segments (LLM disabled)")
        return {"segments": final, "llm_corrected": changed}

    from app.core.llm_corrector import create_corrector, CorrectionConfig
    corr_config = CorrectionConfig(
        enabled=True,
        provider=cfg.get("provider", "openai"),
        api_url=cfg.get("api_url", ""),
        api_key=cfg.get("api_key", ""),
        model=cfg.get("model", ""),
    )
    corrector = create_corrector(corr_config)
    if not corrector:
        from app.core.segment_merger import split_long_segments
        final = split_long_segments(layer1)
        changed = any(
            s["text"] != o["text"] for s, o in zip(layer1, segments)
        )
        print(f"[Deterministic] Corrected {len(layer1)} segments (LLM config missing)")
        return {"segments": final, "llm_corrected": changed}

    print(f"[LLM Enhancement] Enhancing {len(layer1)} segments via {corr_config.provider}")
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        # LLM corrects on top of Layer 1 result (already clean text)
        llm_result = await loop.run_in_executor(None, corrector.correct, layer1)
    except Exception as e:
        print(f"[LLM Enhancement] Failed: {e}, falling back to deterministic result")
        from app.core.segment_merger import split_long_segments
        final = split_long_segments(layer1)
        changed = any(
            s["text"] != o["text"] for s, o in zip(layer1, segments)
        )
        return {"segments": final, "llm_corrected": changed}

    llm_result.sort(key=lambda s: s.get("start", 0))

    # Check if LLM improved anything beyond deterministic correction
    llm_improved = any(
        s["text"] != o["text"] for s, o in zip(llm_result, layer1)
    )
    if llm_improved:
        print(f"[LLM Enhancement] Done — improvements applied")
    else:
        print(f"[LLM Enhancement] Done — no additional changes")

    # Use LLM result if it improved text, otherwise fall back to Layer 1
    final_source = llm_result if llm_improved else layer1

    # Phase 3: Split any segments that are still too long
    from app.core.segment_merger import split_long_segments
    final = split_long_segments(final_source)
    if len(final) != len(final_source):
        print(f"[Segment Splitter] Split {len(final_source)} → {len(final)} segments")

    changed = any(
        s["text"] != o["text"] for s, o in zip(final, segments)
    )
    return {"segments": final, "llm_corrected": changed}


if __name__ == "__main__":
    import uvicorn
    server_cfg = config["server"] if config else {"host": "127.0.0.1", "port": 11235}
    uvicorn.run(app, host=server_cfg["host"], port=server_cfg["port"])
