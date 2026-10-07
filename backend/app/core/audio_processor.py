"""
音频抽取 — 从视频/音频文件得到可供识别的 16kHz 单声道波形。

职责只限于「拿到音频」: fbank / LFR / CMVN 特征由 sherpa-onnx 内部完成,
本模块不再自己算特征 (旧的 SenseVoice 自研 ONNX 路径已移除)。
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
def _run_hidden(cmd, **kwargs):
    """Run subprocess without flashing a console window on Windows."""
    opts = dict(kwargs)
    if sys.platform == "win32":
        opts.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
    subprocess.run(cmd, **opts)


def find_ffmpeg() -> str:
    """Locate ffmpeg executable.

    Search order:
      1. backend/bin/ffmpeg (or .exe on Windows) — bundled with app
      2. ~/.shiyu/bin/ffmpeg — auto-downloaded by ffmpeg_downloader.py
      3. FFMPEG_PATH environment variable
      4. System PATH
      5. Windows: known WinGet installation path (fallback)
      6. Trigger background download to ~/.shiyu/bin/
    Returns the path string, or 'ffmpeg' as last resort.
    """
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    # 1. Bundled binary (next to backend/ in backend/bin/)
    backend_dir = Path(__file__).resolve().parent.parent.parent
    bundled = backend_dir / "bin" / exe_name
    if bundled.is_file():
        return str(bundled)

    # 2. User data dir (~/.shiyu/bin/) — auto-downloaded
    shiyu_bin = Path.home() / ".shiyu" / "bin" / exe_name
    if shiyu_bin.is_file():
        return str(shiyu_bin)

    # 3. Environment variable override
    env_ffmpeg = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg:
        try:
            subprocess.run(
                [env_ffmpeg, "-version"],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            return env_ffmpeg
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 4. System PATH
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 5. Windows: WinGet fallback
    if sys.platform == "win32":
        winget_path = (
            r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
        )
        if os.path.exists(winget_path):
            return winget_path

    # 6. Trigger background download to ~/.shiyu/bin/
    try:
        from app.core.ffmpeg_downloader import ensure_ffmpeg
        ensure_ffmpeg()
        # Return the expected path even if download hasn't finished yet
        return str(shiyu_bin)
    except Exception:
        pass

    return "ffmpeg"  # Let it fail naturally


def extract_audio_from_video(video_path: str, output_wav: str = None, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg. Returns path to wav file.

    Raises:
        RuntimeError: if ffmpeg is not found, the conversion fails, or the
            output wav file is empty. The error message includes the ffmpeg
            stderr (truncated) so the user can diagnose codec/container issues
            without grep-ing logs.
    """
    if output_wav is None:
        output_wav = tempfile.mktemp(suffix=".wav")

    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe == "ffmpeg":
        # Heuristic: the literal string "ffmpeg" is returned when no bundled
        # binary, no env var, no PATH match, and no WinGet fallback found one.
        # We don't know whether the next subprocess.run() will actually find
        # one in PATH, so we let it run and surface a real error if it fails.
        # But we do emit a hint so the user knows fallback was used.
        print("[audio] Note: using system ffmpeg from PATH (no bundled binary found)")

    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate),
        output_wav
    ]
    try:
        result = subprocess.run(
            cmd, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg executable not found at '{ffmpeg_exe}'. "
            f"Install ffmpeg or set FFMPEG_PATH env var. "
            f"Underlying error: {e}"
        )

    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        # ffmpeg prints status to stderr; trim to last 800 chars for readability
        stderr_tail = stderr.strip()[-800:] if stderr.strip() else "(no stderr)"
        raise RuntimeError(
            f"ffmpeg failed to extract audio from video. "
            f"ffmpeg={ffmpeg_exe}, video={video_path}, output={output_wav}, "
            f"returncode={result.returncode}. "
            f"ffmpeg stderr (last 800 chars): {stderr_tail}"
        )

    # Verify the output file is non-empty (ffmpeg can exit 0 but produce a
    # zero-byte file if the input has no audio track).
    if not os.path.exists(output_wav):
        raise RuntimeError(
            f"ffmpeg reported success but output wav does not exist: {output_wav}. "
            f"Check that the input video has an audio track."
        )
    if os.path.getsize(output_wav) < 1024:  # <1 KB is almost certainly empty WAV
        raise RuntimeError(
            f"ffmpeg produced a suspiciously small wav file ({os.path.getsize(output_wav)} bytes). "
            f"Input video may have no audio track or use an unsupported codec. "
            f"Output: {output_wav}"
        )

    return output_wav




