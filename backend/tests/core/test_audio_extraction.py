"""Tests for extract_audio_from_video error reporting.

Verifies that ffmpeg failures surface actionable error messages
instead of bubbling up a raw subprocess.CalledProcessError.
"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from app.core.audio_processor import extract_audio_from_video


def _make_completed_proc(returncode, stderr_text=""):
    """Build a fake subprocess.CompletedProcess with given returncode."""
    p = MagicMock()
    p.returncode = returncode
    p.stdout = b""
    p.stderr = stderr_text.encode("utf-8")
    return p


def test_extract_audio_raises_runtime_error_on_ffmpeg_failure():
    """ffmpeg non-zero exit should raise RuntimeError with stderr included."""
    fake_stderr = "ffmpeg version 6.0 ... Invalid data found when processing input"
    with patch("app.core.audio_processor.subprocess.run",
               return_value=_make_completed_proc(1, fake_stderr)):
        with pytest.raises(RuntimeError) as exc_info:
            extract_audio_from_video("/tmp/nonexistent.mp4")

    msg = str(exc_info.value)
    # Must include the ffmpeg stderr so user can diagnose
    assert "ffmpeg failed" in msg
    assert "Invalid data found" in msg
    assert "returncode=1" in msg


def test_extract_audio_raises_when_output_wav_missing():
    """ffmpeg exits 0 but no output file → RuntimeError."""
    with patch("app.core.audio_processor.subprocess.run",
               return_value=_make_completed_proc(0, "ok")):
        with pytest.raises(RuntimeError) as exc_info:
            extract_audio_from_video("/tmp/video.mp4", output_wav="/tmp/never_created.wav")
    msg = str(exc_info.value)
    assert "output wav does not exist" in msg
    assert "audio track" in msg  # diagnostic hint


def test_extract_audio_raises_when_output_too_small():
    """ffmpeg produces a 100-byte file → likely empty → RuntimeError."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"\x00" * 100)
        tiny_wav = f.name
    try:
        with patch("app.core.audio_processor.subprocess.run",
                   return_value=_make_completed_proc(0, "ok")):
            with pytest.raises(RuntimeError) as exc_info:
                extract_audio_from_video("/tmp/video.mp4", output_wav=tiny_wav)
        msg = str(exc_info.value)
        assert "suspiciously small" in msg
        assert "no audio track" in msg
    finally:
        os.unlink(tiny_wav)


def test_extract_audio_raises_when_ffmpeg_not_found():
    """FileNotFoundError from subprocess → RuntimeError with installation hint."""
    with patch("app.core.audio_processor.subprocess.run",
               side_effect=FileNotFoundError("No such file: 'ffmpeg'")):
        with pytest.raises(RuntimeError) as exc_info:
            extract_audio_from_video("/tmp/video.mp4")

    msg = str(exc_info.value)
    assert "ffmpeg executable not found" in msg
    assert "FFMPEG_PATH" in msg  # tells user how to fix
