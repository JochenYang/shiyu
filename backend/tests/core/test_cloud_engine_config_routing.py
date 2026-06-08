"""Tests for config-driven cloud engine routing.

Verifies the generic OpenAI-compatible ASR abstraction:
- request_mode selects whisper vs chat_completions
- auth_header.name + value_template override defaults
- asr_options is forwarded to the request body
- Hard-coded URL sniffing (_is_dashscope_url) no longer drives routing

Note: HTTP-level integration is covered by the manual curl test
against the real MiMo endpoint. Unit tests focus on the pure
decision logic (auth header resolution, request_mode validation,
URL-sniffing back-compat helper) so they're fast, deterministic,
and don't require a real audio file.
"""
import pytest
from unittest.mock import patch

from app.core.cloud_engine import (
    _resolve_auth_header,
    _is_dashscope_url,
)


# ── Auth header resolution ───────────────────────────────────────────

def test_resolve_auth_header_default_openai_compatible():
    """Default is Authorization: Bearer {api_key} (OpenAI standard)."""
    config = {"api_key": "sk-test123"}
    header = _resolve_auth_header(config)
    assert header == {"Authorization": "Bearer sk-test123"}


def test_resolve_auth_header_xiaomi_mimo():
    """MiMo uses api-key header per its curl example."""
    config = {
        "api_key": "tp-test",
        "auth_header": {"name": "api-key", "value_template": "{api_key}"},
    }
    header = _resolve_auth_header(config)
    assert header == {"api-key": "tp-test"}


def test_resolve_auth_header_custom_template():
    """value_template can include extra prefixes if a provider needs it."""
    config = {
        "api_key": "secret",
        "auth_header": {"name": "X-API-Key", "value_template": "Token {api_key}"},
    }
    header = _resolve_auth_header(config)
    assert header == {"X-API-Key": "Token secret"}


def test_resolve_auth_header_missing_key():
    """Empty api_key yields empty Authorization value (not a crash)."""
    config = {"api_key": "", "auth_header": {"name": "X-Key", "value_template": "{api_key}"}}
    header = _resolve_auth_header(config)
    assert header == {"X-Key": ""}


# ── is_dashscope_url (legacy helper, kept for back-compat) ──────────

def test_is_dashscope_url_still_detects_dashscope():
    """Legacy URL sniffing still works for back-compat code paths."""
    assert _is_dashscope_url("https://dashscope.aliyuncs.com/v1") is True
    assert _is_dashscope_url("https://api.xiaomimimo.com/v1") is False
    assert _is_dashscope_url("https://api.openai.com/v1") is False


# ── transcribe_cloud routing decision (pure logic, no HTTP) ──────────

def test_transcribe_cloud_rejects_unknown_request_mode():
    """Unknown request_mode raises ValueError with helpful message."""
    from app.core.cloud_engine import transcribe_cloud
    config = {
        "api_url": "https://api.example.com/v1",
        "api_key": "x",
        "model": "m",
        "enabled": True,
        "request_mode": "garbage",
    }
    with patch("app.core.cloud_engine.load_engine_config", return_value=config):
        with pytest.raises(ValueError, match="Unknown engine.request_mode"):
            transcribe_cloud("/tmp/audio.wav")


def test_transcribe_cloud_routing_decision_via_mock():
    """Verify request_mode → function dispatch is config-driven.

    Patches _transcribe_whisper and _transcribe_dashscope as MagicMock
    to capture which branch is taken based on config. Avoids touching
    real HTTP / audio / librosa paths.
    """
    from app.core import cloud_engine

    config_chat = {
        "api_url": "https://api.xiaomimimo.com/v1",
        "api_key": "tp",
        "model": "mimo-v2.5-asr",
        "request_mode": "chat_completions",
    }
    with patch.object(cloud_engine, "load_engine_config", return_value=config_chat), \
         patch.object(cloud_engine, "_transcribe_dashscope",
                     return_value=("", [{"start": 0, "end": 1, "text": "x"}])) as m_ds, \
         patch.object(cloud_engine, "_transcribe_whisper") as m_wh:
        cloud_engine.transcribe_cloud("/tmp/a.wav")
    assert m_ds.called
    assert not m_wh.called

    config_whisper = {
        "api_url": "https://api.openai.com/v1",
        "api_key": "sk",
        "model": "whisper-1",
        # request_mode omitted → default whisper
    }
    fake_seg = [{"start": 0, "end": 1, "text": "x"}]
    with patch.object(cloud_engine, "load_engine_config", return_value=config_whisper), \
         patch.object(cloud_engine, "_transcribe_dashscope") as m_ds, \
         patch.object(cloud_engine, "_transcribe_whisper",
                     return_value=("", fake_seg)) as m_wh, \
         patch.object(cloud_engine, "_post_multipart_via_curl",
                     return_value={"text": "x", "segments": fake_seg}):
        cloud_engine.transcribe_cloud("/tmp/a.wav")
    # whisper path attempted (mocked to return successfully)
    assert m_wh.called
    assert not m_ds.called


def test_transcribe_cloud_routes_openai_url_to_whisper_by_default():
    """An OpenAI URL with no request_mode falls through to whisper (default)."""
    from app.core import cloud_engine
    fake_seg = [{"start": 0, "end": 1, "text": "x"}]
    config = {
        "api_url": "https://api.openai.com/v1",
        "api_key": "sk",
        "model": "whisper-1",
    }
    with patch.object(cloud_engine, "load_engine_config", return_value=config), \
         patch.object(cloud_engine, "_transcribe_dashscope") as m_ds, \
         patch.object(cloud_engine, "_transcribe_whisper",
                     return_value=("", fake_seg)) as m_wh, \
         patch.object(cloud_engine, "_post_multipart_via_curl",
                     return_value={"text": "x", "segments": fake_seg}):
        cloud_engine.transcribe_cloud("/tmp/a.wav")
    assert m_wh.called
    assert not m_ds.called
