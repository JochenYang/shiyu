"""
P1 引擎路由测试 — main.py 的引擎解析、端点路由与 /health 就绪状态.

覆盖:
  1. _resolve_engines_list: cloud / 单引擎 / 默认 local 各模式
  2. 三个 transcribe 端点: engine_mode 传入 PipelineContext, 确定性修正默认开启
  3. /health: 任一本地引擎可用即 local_ready=True
"""
import asyncio
import io

import pytest
from fastapi import UploadFile


def _make_config(sensevoice_enabled: bool = False) -> dict:
    """构造最小 config 字典."""
    return {
        "engines": {
            "sensevoice": {"enabled": sensevoice_enabled},
        },
    }


class TestResolveEnginesList:
    """_resolve_engines_list 按 engine_mode 解析引擎列表."""

    def test_cloud_mode_returns_cloud_only(self):
        from app.main import _resolve_engines_list
        assert _resolve_engines_list(_make_config(), "cloud") == ["cloud"]

    def test_specific_engine_mode_returns_that_engine(self):
        from app.main import _resolve_engines_list
        for engine in ("zipformer_local", "sensevoice_local", "paraformer_local"):
            assert _resolve_engines_list(_make_config(), engine) == [engine]

    def test_default_local_mode_uses_zipformer(self):
        from app.main import _resolve_engines_list
        assert _resolve_engines_list(_make_config(), "local") == ["zipformer_local"]

    def test_unknown_mode_falls_back_to_default(self):
        from app.main import _resolve_engines_list
        assert _resolve_engines_list(_make_config(), "some_future_engine") == ["zipformer_local"]

    def test_local_mode_appends_sensevoice_when_enabled(self):
        from app.main import _resolve_engines_list
        cfg = _make_config(sensevoice_enabled=True)
        assert _resolve_engines_list(cfg, "local") == ["zipformer_local", "sensevoice_local"]

    def test_no_config_returns_zipformer_default(self):
        from app.main import _resolve_engines_list
        assert _resolve_engines_list(None, "local") == ["zipformer_local"]


class TestHealthLocalReady:
    """/health 的 model_loaded (local_ready 判断) 反映任一本地引擎可用."""

    def _patch_engines(self, monkeypatch, ready_local=(), cloud_ready=False):
        """monkeypatch get_available_engines, 固定本地/云端引擎可用性."""
        import app.main as main
        engines = []
        for name in ("zipformer_local", "sensevoice_local", "paraformer_local"):
            engines.append({"name": name, "available": name in ready_local})
        engines.append({"name": "cloud", "available": cloud_ready})
        monkeypatch.setattr(main, "get_available_engines", lambda: engines)
        # 避免 local_ready=False 时触碰真实下载状态
        monkeypatch.setattr(main, "get_download_status", lambda: {"status": "unknown"})
        return main

    def test_local_ready_true_when_zipformer_available(self, monkeypatch):
        main = self._patch_engines(monkeypatch, ready_local=("zipformer_local",))
        result = asyncio.run(main.health_check())
        assert result["model_loaded"] is True

    def test_local_ready_true_when_sensevoice_available(self, monkeypatch):
        main = self._patch_engines(monkeypatch, ready_local=("sensevoice_local",))
        result = asyncio.run(main.health_check())
        assert result["model_loaded"] is True

    def test_local_ready_true_when_paraformer_available(self, monkeypatch):
        main = self._patch_engines(monkeypatch, ready_local=("paraformer_local",))
        result = asyncio.run(main.health_check())
        assert result["model_loaded"] is True

    def test_local_ready_false_when_only_cloud_available(self, monkeypatch):
        main = self._patch_engines(monkeypatch, cloud_ready=True)
        result = asyncio.run(main.health_check())
        assert result["model_loaded"] is False

    def test_local_ready_false_when_no_local_engine_available(self, monkeypatch):
        main = self._patch_engines(monkeypatch)
        result = asyncio.run(main.health_check())
        assert result["model_loaded"] is False


class TestTranscribeEndpoints:
    """三个 transcribe 端点把 engine_mode 传入 PipelineContext."""

    @pytest.fixture
    def capture_pipeline(self, monkeypatch):
        """替换 SubtitlePipeline, 捕获构造时的 PipelineContext."""
        import app.main as main
        captured = {}

        class FakePipeline:
            def __init__(self, ctx):
                captured["ctx"] = ctx
                self.ctx = ctx

            def run(self):
                self.ctx.output_content = "captured"
                # 模拟 ASR 完成: 端点会在 engines_results 为空时报 500
                self.ctx.engines_results = {"mock_engine": []}
                return self.ctx

        monkeypatch.setattr(main, "SubtitlePipeline", FakePipeline)
        monkeypatch.setattr(main, "_extract_audio_for_transcribe",
                            lambda p, cfg=None: str(p))
        return captured

    @staticmethod
    def _upload(filename: str = "input.mp4") -> UploadFile:
        return UploadFile(file=io.BytesIO(b"fake"), filename=filename)

    def test_transcribe_file_uses_cloud_engine_mode(self, capture_pipeline):
        import app.main as main
        asyncio.run(main.transcribe_file(
            file=self._upload(),
            language="auto",
            output_format="srt",
            glossary="{}",
            engine_mode="cloud",
            subtitle_offset_sec=0.0,
        ))
        assert capture_pipeline["ctx"].engines == ["cloud"]
        assert capture_pipeline["ctx"].enable_deterministic_correction is True

    def test_transcribe_file_uses_zipformer_engine_mode(self, capture_pipeline):
        import app.main as main
        asyncio.run(main.transcribe_file(
            file=self._upload(),
            language="auto",
            output_format="srt",
            glossary="{}",
            engine_mode="zipformer_local",
            subtitle_offset_sec=0.0,
        ))
        assert capture_pipeline["ctx"].engines == ["zipformer_local"]
        assert capture_pipeline["ctx"].enable_deterministic_correction is True

    def test_transcribe_file_defaults_to_local(self, capture_pipeline):
        import app.main as main
        asyncio.run(main.transcribe_file(
            file=self._upload(),
            language="auto",
            output_format="srt",
            glossary="{}",
            subtitle_offset_sec=0.0,
        ))
        assert capture_pipeline["ctx"].engines == ["zipformer_local"]

    def test_transcribe_json_uses_sensevoice_engine_mode(self, capture_pipeline):
        import app.main as main
        asyncio.run(main.transcribe_json(
            file=self._upload(),
            language="auto",
            glossary="{}",
            engine_mode="sensevoice_local",
            subtitle_offset_sec=0.0,
        ))
        assert capture_pipeline["ctx"].engines == ["sensevoice_local"]
        assert capture_pipeline["ctx"].enable_deterministic_correction is True

    def test_transcribe_local_uses_paraformer_engine_mode(self, capture_pipeline, tmp_path):
        import app.main as main
        media = tmp_path / "input.wav"
        media.write_bytes(b"fake")
        asyncio.run(main.transcribe_local(
            path=str(media),
            language="auto",
            output_format="srt",
            glossary="{}",
            engine_mode="paraformer_local",
            subtitle_offset_sec=0.0,
        ))
        assert capture_pipeline["ctx"].engines == ["paraformer_local"]
        assert capture_pipeline["ctx"].enable_deterministic_correction is True
