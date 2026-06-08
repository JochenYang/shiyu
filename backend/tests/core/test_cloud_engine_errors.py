"""Tests for cloud_engine._friendly_http_error.

Verifies that HTML gateway pages, JSON error envelopes, and unknown
content types are all converted to actionable user-facing messages.
"""
import json
from app.core.cloud_engine import _friendly_http_error


class _FakeResponse:
    def __init__(self, status_code, content_type, text):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.text = text

    def json(self):
        return json.loads(self.text)


def test_html_404_returns_url_hint():
    """HTML 404 from openresty/nginx should not be dumped to the user."""
    html = (
        "<html><head><title>404 Not Found</title></head>"
        "<body><center><h1>404 Not Found</h1></center>"
        "<hr><center>openresty</center></body></html>"
    )
    resp = _FakeResponse(404, "text/html; charset=utf-8", html)
    err = _friendly_http_error(resp, "https://token-plan-cn.xiaomimimo.com/v1/audio/transcriptions", "Whisper")
    msg = str(err)
    assert "404" in msg
    assert "Whisper" in msg
    assert "openresty" in msg or "404" in msg  # server hint captured
    assert "https://token-plan-cn.xiaomimimo.com" in msg
    # MUST NOT contain raw HTML tags
    assert "<html>" not in msg
    assert "<body>" not in msg


def test_json_error_envelope_parsed():
    """Standard OpenAI-style error JSON should be parsed into a readable message."""
    body = json.dumps({"error": {"message": "Invalid API key", "type": "auth"}})
    resp = _FakeResponse(401, "application/json", body)
    err = _friendly_http_error(resp, "https://api.example.com/v1/audio/transcriptions", "Whisper")
    msg = str(err)
    assert "401" in msg
    assert "Invalid API key" in msg


def test_json_string_error_envelope():
    """Some providers return 'error' as a string instead of a dict."""
    body = json.dumps({"error": "rate limit exceeded"})
    resp = _FakeResponse(429, "application/json", body)
    err = _friendly_http_error(resp, "https://api.example.com/v1/audio/transcriptions", "Whisper")
    msg = str(err)
    assert "429" in msg
    assert "rate limit" in msg


def test_malformed_json_returns_body_snippet():
    resp = _FakeResponse(500, "application/json", "{not valid json")
    err = _friendly_http_error(resp, "https://api.example.com/v1", "Whisper")
    msg = str(err)
    assert "500" in msg
    assert "malformed JSON" in msg or "Body:" in msg


def test_unknown_content_type_returns_body():
    resp = _FakeResponse(502, "text/plain", "Bad Gateway from upstream")
    err = _friendly_http_error(resp, "https://api.example.com/v1", "Whisper")
    msg = str(err)
    assert "502" in msg
    assert "Bad Gateway" in msg


def test_endpoint_always_in_message():
    """All error messages must include the endpoint so user can verify config."""
    html = "<html>404</html>"
    resp = _FakeResponse(404, "text/html", html)
    err = _friendly_http_error(
        resp,
        "https://token-plan-cn.xiaomimimo.com/v1/audio/transcriptions",
        "Whisper",
    )
    assert "https://token-plan-cn.xiaomimimo.com/v1/audio/transcriptions" in str(err)
