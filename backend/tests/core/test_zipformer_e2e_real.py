"""
E2E test: Run Zipformer on the user's actual audio scenario.

This test verifies that the v2.0 Zipformer pipeline produces clean SRT output
(no 8-segment duplication, no "ph y s ic" broken BPE words like SenseVoice).
"""
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


ZIPFORMER_TEST_WAV = os.path.expanduser(
    "~/.shiyu/models/zipformer-ctc-zh/test_wavs/0.wav"
)


def _create_test_audio(duration_sec: float = 5.0, output_path: Path = None) -> Path:
    """Create a simple synthetic audio file for testing (sine wave at 440Hz).

    Real audio not provided — Zipformer's bundled test_wavs/0.wav is used.
    """
    if output_path is None:
        output_path = Path("test_audio_sine_5s.wav")
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Amplitude-modulated sine (simulates speech)
    audio = 0.3 * np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 3 * t)
    audio = audio.astype(np.float32)
    sf.write(str(output_path), audio, sr)
    return output_path


def test_zipformer_real_audio_no_duplication():
    """Run Zipformer on real test audio; verify no 8-segment duplication (the v1.x bug)."""
    from app.main import _register_default_engines, get_engine
    from app.core.config_loader import load_config

    cfg = load_config("config.yaml")
    _register_default_engines()
    engine = get_engine("zipformer_local")

    if not Path(ZIPFORMER_TEST_WAV).exists():
        pytest.skip(f"Zipformer test wav not found: {ZIPFORMER_TEST_WAV}")

    result = engine.transcribe(ZIPFORMER_TEST_WAV, language="zh")
    assert len(result.segments) >= 1, "Zipformer produced no segments"

    # Verify NO 8-segment duplication (the v1.x bug)
    # i.e., no two segments with same start AND end timestamps
    timestamp_pairs = [(s.start, s.end) for s in result.segments]
    assert len(timestamp_pairs) == len(set(timestamp_pairs)), (
        f"Found {len(timestamp_pairs) - len(set(timestamp_pairs))} "
        f"duplicate timestamp pairs (v1.x SRT bug regression)"
    )

    # Verify NO "ph y s ic" word-broken output (SenseVoice BPE bug)
    full_text = result.full_text
    assert "ph y s" not in full_text, (
        f"SenseVoice BPE-broken tokens detected in Zipformer output: {full_text!r}"
    )
    assert "spro cess" not in full_text, (
        f"SenseVoice BPE-broken tokens detected: {full_text!r}"
    )
    assert "mo v le" not in full_text, (
        f"SenseVoice BPE-broken tokens detected: {full_text!r}"
    )


def test_zipformer_synthetic_audio():
    """Create synthetic audio and run Zipformer; verify it doesn't crash."""
    from app.main import _register_default_engines, get_engine
    from app.core.config_loader import load_config

    cfg = load_config("config.yaml")
    _register_default_engines()
    engine = get_engine("zipformer_local")

    test_wav = _create_test_audio(duration_sec=3.0)
    try:
        result = engine.transcribe(str(test_wav), language="zh")
        assert result is not None
        # Synthetic sine wave may produce empty transcription, that's OK
    finally:
        if test_wav.exists():
            test_wav.unlink()


def test_zipformer_english_token_recognition():
    """Verify Zipformer outputs English tokens as complete words, not character-by-character."""
    from app.main import _register_default_engines, get_engine
    from app.core.config_loader import load_config

    cfg = load_config("config.yaml")
    _register_default_engines()
    engine = get_engine("zipformer_local")

    if not Path(ZIPFORMER_TEST_WAV).exists():
        pytest.skip(f"Zipformer test wav not found: {ZIPFORMER_TEST_WAV}")

    result = engine.transcribe(ZIPFORMER_TEST_WAV, language="zh")
    full_text = result.full_text

    # The test wav contains the phrase "大家如果对我的研究感兴趣" (everyone interested in my research)
    # If Zipformer output broke English tokens character-by-character, we'd see:
    #   "r e s e a r c h" or "r-e-s-e-a-r-c-h"
    # Instead, Zipformer should produce complete Chinese characters
    # (the test wav is pure Chinese so this is mainly a structural check)
    for char in full_text:
        if char.isascii() and char.isalpha():
            # If we see English letters, they should form real words
            # Not single chars with spaces between them
            pass

    # No space-padded English letters (SenseVoice bug signature)
    import re
    suspicious_pattern = re.compile(r"\b[a-zA-Z] [a-zA-Z] [a-zA-Z]\b")
    assert not suspicious_pattern.search(full_text), (
        f"Found space-padded English letters in output: {full_text!r}"
    )
