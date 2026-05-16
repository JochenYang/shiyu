"""
Subtitle Assistant - FastAPI Backend Service.
Provides REST API for audio/video transcription and subtitle generation.
"""
import os
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.core.config_loader import load_config
from app.core.audio_processor import preprocess_audio
from app.core.sensevoice_engine import SenseVoiceEngine
from app.core.subtitle_formatter import (
    generate_srt, generate_lrc, generate_ass,
    segments_from_transcription, SubtitleSegment
)


# Global state
config = None
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config and model. Shutdown: cleanup."""
    global config, engine
    
    # Load config
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    
    # Init ONNX engine
    model_cfg = config["model"]
    inference_cfg = config["inference"]
    engine = SenseVoiceEngine(
        model_path=model_cfg["onnx_path"],
        tokens_path=model_cfg["tokens_path"],
        device=inference_cfg.get("device", "cpu")
    )
    
    print(f"[SubtitleAssistant] Model loaded: {model_cfg['onnx_path']}")
    print(f"[SubtitleAssistant] Server ready at http://{config['server']['host']}:{config['server']['port']}")
    
    yield
    
    # Cleanup
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
    max_line_length: Optional[int] = 40
    max_line_count: Optional[int] = 2


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    output_format: str = Form("srt"),
    max_line_length: int = Form(40),
    max_line_count: int = Form(2)
):
    """Transcribe audio/video file and return subtitle content.
    
    Args:
        file: Audio or video file
        language: Target language (auto/zh/en/ja/ko/yue)
        output_format: srt / lrc / ass
        max_line_length: Max chars per subtitle line
        max_line_count: Max lines per subtitle event
    
    Returns:
        Plain text subtitle content
    """
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    
    # Validate format
    fmt = output_format.lower()
    if fmt not in ("srt", "lrc", "ass"):
        raise HTTPException(400, f"Unsupported format: {output_format}")
    
    # Save uploaded file to temp
    suffix = Path(file.filename).suffix or ".tmp"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / f"input{suffix}"
    
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Check if video
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
        is_video = tmp_path.suffix.lower() in video_exts
        
        # Preprocess audio
        audio_cfg = config["audio"]
        features = preprocess_audio(str(tmp_path), is_video=is_video, config=audio_cfg)
        
        # Get audio duration for timestamp estimation
        import librosa
        if is_video:
            from app.core.audio_processor import extract_audio_from_video
            audio_path = extract_audio_from_video(str(tmp_path))
        else:
            audio_path = str(tmp_path)
        
        waveform, sr = librosa.load(audio_path, sr=audio_cfg["sample_rate"])
        duration_sec = len(waveform) / sr
        
        # Run inference
        segments_raw = engine.transcribe_with_timestamps(
            features, 
            audio_duration_sec=duration_sec,
            language=language
        )
        
        # Convert to subtitle segments
        segments = segments_from_transcription(segments_raw)
        
        # Generate subtitle
        if fmt == "srt":
            content = generate_srt(segments, max_line_length, max_line_count)
        elif fmt == "lrc":
            content = generate_lrc(segments)
        else:  # ass
            content = generate_ass(segments, title=file.filename)
        
        return PlainTextResponse(content)
    
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/transcribe/json")
async def transcribe_json(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe and return raw JSON with segments."""
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    
    suffix = Path(file.filename).suffix or ".tmp"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / f"input{suffix}"
    
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
        is_video = tmp_path.suffix.lower() in video_exts
        
        audio_cfg = config["audio"]
        features = preprocess_audio(str(tmp_path), is_video=is_video, config=audio_cfg)
        
        import librosa
        if is_video:
            from app.core.audio_processor import extract_audio_from_video
            audio_path = extract_audio_from_video(str(tmp_path))
        else:
            audio_path = str(tmp_path)
        
        waveform, sr = librosa.load(audio_path, sr=audio_cfg["sample_rate"])
        duration_sec = len(waveform) / sr
        
        segments = engine.transcribe_with_timestamps(
            features, duration_sec, language
        )
        
        return {
            "filename": file.filename,
            "language": language,
            "duration": duration_sec,
            "segments": segments
        }
    
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    server_cfg = config["server"] if config else {"host": "127.0.0.1", "port": 11235}
    uvicorn.run(app, host=server_cfg["host"], port=server_cfg["port"])