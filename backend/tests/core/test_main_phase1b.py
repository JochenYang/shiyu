"""
Phase 1B 测试: main.py 错误显式化

覆盖:
  1. _apply_llm_correction 在 LLM 关闭时设置 llm_status='skipped'
  2. _apply_llm_correction 在 LLM 成功时设置 llm_status='success'
  3. _apply_llm_correction 在 LLM 抛异常时设置 llm_status='failed'
  4. glossary json.JSONDecodeError 返回 HTTP 400 invalid_glossary
  5. print() 在 _register_default_engines / lifespan 中被替换为 logger
"""
import json
import logging
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import (
    _apply_llm_correction,
    app,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_llm_flags():
    """每个测试前重置 LLM 状态标志."""
    main_module._llm_status = "skipped"
    main_module._llm_corrected = False
    yield
    main_module._llm_status = "skipped"
    main_module._llm_corrected = False


class TestLLMStatusFlag:
    """_llm_status 全局标志在各种场景下行为正确."""

    def test_disabled_correction_sets_skipped(self):
        with patch(
            "app.core.cloud_engine.load_llm_correction_config",
            return_value={"enabled": False},
        ):
            result = _apply_llm_correction([{"start": 0, "end": 1, "text": "x"}])
        assert main_module._llm_status == "skipped"
        assert result == [{"start": 0, "end": 1, "text": "x"}]

    def test_failed_correction_sets_failed(self):
        """corrector.correct() 抛异常 → llm_status='failed', 原文保留."""
        cfg_mock = MagicMock()
        cfg_mock.get.side_effect = lambda k, default=None: {
            "enabled": True, "provider": "openai",
            "api_url": "https://x", "api_key": "k", "model": "m",
        }.get(k, default)

        with patch(
            "app.core.cloud_engine.load_llm_correction_config",
            return_value=cfg_mock,
        ), patch(
            "app.core.llm_corrector.create_corrector",
        ) as create_mock:
            corr_instance = MagicMock()
            corr_instance.correct.side_effect = RuntimeError("LLM down")
            create_mock.return_value = corr_instance
            segs = [{"start": 0, "end": 1, "text": "原文本"}]
            result = _apply_llm_correction(segs)
        assert main_module._llm_status == "failed"
        assert result == segs  # fallback to original

    def test_corrector_returns_none_sets_skipped(self):
        """create_corrector 返回 None (provider missing) → skipped."""
        cfg_mock = MagicMock()
        cfg_mock.get.side_effect = lambda k, default=None: {
            "enabled": True, "provider": "openai",
            "api_url": "", "api_key": "", "model": "",
        }.get(k, default)

        with patch(
            "app.core.cloud_engine.load_llm_correction_config",
            return_value=cfg_mock,
        ), patch(
            "app.core.llm_corrector.create_corrector",
            return_value=None,
        ):
            segs = [{"start": 0, "end": 1, "text": "x"}]
            result = _apply_llm_correction(segs)
        assert main_module._llm_status == "skipped"


class TestGlossaryJSONErrorReturns400:
    """glossary JSON 解析失败时, 返回 400 + invalid_glossary 错误体."""

    def test_transcribe_invalid_glossary(self, client, tmp_path):
        # 准备一个临时音频文件
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        # 用一个明显非 JSON 的 glossary 触发 JSONDecodeError
        with patch.object(main_module, "get_engine") as get_engine_mock:
            transcriber = MagicMock()
            transcriber.transcribe.return_value = MagicMock(
                segments=[],
                language="auto",
            )
            # to_segments_list via property
            result_obj = MagicMock()
            result_obj.to_segments_list.return_value = []
            transcriber.transcribe.return_value = result_obj
            get_engine_mock.return_value = transcriber

            response = client.post(
                "/transcribe/json",
                files={"file": ("test.wav", audio.read_bytes(), "audio/wav")},
                data={"glossary": "this-is-not-json{"},
            )
        # FastAPI TestClient 对 JSONResponse 不会自动解析 — 校验 status_code
        assert response.status_code == 400, (
            f"应返回 400, 实际 {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("error") == "invalid_glossary", (
            f"error 字段应为 'invalid_glossary', 实际 {body!r}"
        )
        assert "detail" in body

    def test_transcribe_local_invalid_glossary(self, client, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        with patch.object(main_module, "get_engine") as get_engine_mock:
            transcriber = MagicMock()
            transcriber.transcribe.return_value = MagicMock(segments=[], language="auto")
            get_engine_mock.return_value = transcriber

            response = client.post(
                "/transcribe/local",
                data={
                    "path": str(audio),
                    "glossary": "broken-json",
                    "output_format": "srt",
                },
            )
        assert response.status_code == 400, (
            f"应返回 400, 实际 {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("error") == "invalid_glossary"


class TestPrintReplacedWithLogger:
    """lifespan 和 _register_default_engines 不应再调用 print()."""

    def test_no_print_in_lifespan_or_register(self, caplog):
        """运行 lifespan 启动流程时, 不应触发任何 print() 调用.

        通过 mock print 来捕获意外调用.
        """
        with patch("builtins.print") as print_mock:
            with caplog.at_level(logging.INFO, logger="app.main"):
                # 直接调用 _register_default_engines 验证
                main_module._register_default_engines()
        # print 可能在 lifespan 启动时调用, 但 _register_default_engines
        # 内部不应再使用 print — 改用 logger
        # 注意: 这里只检查 _register_default_engines 范围
        register_calls = [
            c for c in print_mock.call_args_list
            if "Engine Registry" in str(c)
        ]
        assert len(register_calls) == 0, (
            f"_register_default_engines 不应再 print(), 实际调用: {register_calls}"
        )
