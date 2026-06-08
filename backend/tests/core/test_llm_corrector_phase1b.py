"""
Phase 1B 测试: llm_corrector 性能/可靠性调优

覆盖:
  1. DEFAULT_BATCH_SIZE 降为 20
  2. DEFAULT_MAX_WORKERS 默认 5, env LLM_MAX_WORKERS 可覆盖
  3. DEFAULT_TIMEOUT_SEC 默认 90, env LLM_TIMEOUT_SEC 可覆盖
  4. DEFAULT_RETRIES == 1 (timeout/network 立即重试一次)
  5. _call_llm 一次成功直接返回, 第二次不重试
"""
import os
import time
from unittest.mock import patch, MagicMock

import pytest
import requests

from app.core import llm_corrector
from app.core.llm_corrector import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_RETRIES,
    OpenAICorrector,
    CorrectionConfig,
)


class TestPhase1BConstants:
    """调优常量验证."""

    def test_default_batch_size_is_20(self):
        """Phase 1B: 批次大小 50 -> 20."""
        assert DEFAULT_BATCH_SIZE == 20

    def test_default_max_workers_is_5(self):
        """Phase 1B: 并发 worker 3 -> 5."""
        assert DEFAULT_MAX_WORKERS == 5

    def test_default_timeout_sec_is_90(self):
        """Phase 1B: 超时 180s -> 90s."""
        assert DEFAULT_TIMEOUT_SEC == 90

    def test_default_retries_is_1(self):
        """Phase 1B: 默认 1 次立即重试 (timeout/network)."""
        assert DEFAULT_RETRIES == 1

    def test_env_override_max_workers(self):
        os.environ["LLM_MAX_WORKERS"] = "12"
        try:
            # Re-evaluate module-level int(os.environ.get(...)) value
            import importlib
            importlib.reload(llm_corrector)
            assert llm_corrector.DEFAULT_MAX_WORKERS == 12
        finally:
            os.environ.pop("LLM_MAX_WORKERS", None)
            importlib.reload(llm_corrector)

    def test_env_override_timeout(self):
        os.environ["LLM_TIMEOUT_SEC"] = "45"
        try:
            import importlib
            importlib.reload(llm_corrector)
            assert llm_corrector.DEFAULT_TIMEOUT_SEC == 45
        finally:
            os.environ.pop("LLM_TIMEOUT_SEC", None)
            importlib.reload(llm_corrector)


class TestPhase1BRetryBehavior:
    """_call_llm 重试行为: timeout/network 重试, HTTP 4xx/5xx 不重试."""

    def _make_corrector(self):
        cfg = CorrectionConfig(
            enabled=True, provider="openai",
            api_url="https://example.com/v1/chat/completions",
            api_key="k", model="m",
        )
        return OpenAICorrector(cfg)

    def _ok_response(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"content": "你好世界"}}]
        }
        return resp

    def test_success_path_no_retry(self):
        """一次成功: 只调用 1 次, 不会触发重试."""
        c = self._make_corrector()
        with patch.object(requests, "post", return_value=self._ok_response()) as m:
            text = c._call_llm("https://x", "k", "m", "prompt")
        assert text == "你好世界"
        assert m.call_count == 1

    def test_timeout_retries_once(self):
        """第一次 timeout, 第二次成功 — 总调用 2 次."""
        c = self._make_corrector()
        with patch.object(
            requests, "post",
            side_effect=[requests.exceptions.Timeout("first"), self._ok_response()],
        ) as m:
            text = c._call_llm("https://x", "k", "m", "prompt")
        assert text == "你好世界"
        assert m.call_count == 2, f"应重试 1 次, 总计 2 次, 实际 {m.call_count}"

    def test_http_error_no_retry(self):
        """HTTP 4xx/5xx 不重试."""
        c = self._make_corrector()
        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=500)
        )
        with patch.object(requests, "post", return_value=bad_resp) as m:
            text = c._call_llm("https://x", "k", "m", "prompt")
        assert text is None
        assert m.call_count == 1, f"HTTP 错误不应重试, 实际 {m.call_count} 次"


class TestPhase1BBatchExecution:
    """_batch_correct 实际使用新批次大小切分 + 进度日志."""

    def test_uses_default_batch_size_20(self):
        """30 个 segments 应被切分为 2 个 batch (20 + 10)."""
        c = self._make_corrector_for_batch()
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"line{i}"}
            for i in range(30)
        ]
        # Mock _process_single_batch: 返回与输入同长度的列表
        with patch.object(
            OpenAICorrector, "_process_single_batch",
            side_effect=lambda *a, **kw: list(reversed(kw.get("original_texts", a[-1]))) if kw.get("original_texts") else list(reversed(a[-1])),
        ) as m:
            # 由于签名是 _process_single_batch(api_url, api_key, model, prompt_text, original_texts)
            # 我们用简单 mock
            c.correct(segments)
        # 30 segments / batch_size 20 = 2 batches
        assert m.call_count == 2, f"应切为 2 个 batch, 实际 {m.call_count}"


    def _make_corrector_for_batch(self):
        cfg = CorrectionConfig(
            enabled=True, provider="openai",
            api_url="https://example.com/v1/chat/completions",
            api_key="k", model="m",
        )
        return OpenAICorrector(cfg)
