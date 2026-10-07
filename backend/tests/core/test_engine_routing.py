"""
引擎路由测试 — main.py 的引擎解析、端点路由与 /health 就绪状态。

覆盖:
  1. _resolve_engines_list: cloud / 已注册本地引擎 / 默认引擎 / 未注册回退
  2. 三个 transcribe 端点: engine_mode 传入 PipelineContext, 确定性修正默认开启
  3. /health: 任一本地引擎可用即 local_ready=True (云端不算)
"""
import asyncio
import io

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

def _register(monkeypatch, names=("zipformer_local", "sensevoice_local", "firered_local")):
    """把 main 的引擎注册表替换成给定的一组引擎名。"""
    import app.main as main

    registry = {name: {"cls": object, "kwargs": {}, "instance": object()} for name in names}
    monkeypatch.setattr(main, "_transcriber_registry", registry)
    return main


class TestResolveEnginesList:
    """_resolve_engines_list 按 engine_mode 解析引擎列表。"""

    def test_cloud_mode_returns_cloud_only(self, monkeypatch):
        main = _register(monkeypatch)
        assert main._resolve_engines_list(None, "cloud") == ["cloud"]

    def test_registered_engine_mode_returns_that_engine(self, monkeypatch):
        main = _register(monkeypatch)
        for engine in ("zipformer_local", "sensevoice_local", "firered_local"):
            assert main._resolve_engines_list(None, engine) == [engine]

    def test_default_mode_uses_catalog_default_engine(self, monkeypatch):
        """默认模式用清单默认模型对应的引擎, 而不是硬编码的引擎名。"""
        from app.core import model_catalog

        main = _register(monkeypatch)
        expected = model_catalog.model_spec(model_catalog.DEFAULT_MODEL_ID)["engine_name"]
        assert main._resolve_engines_list(None, "local") == [expected]

    def test_unknown_mode_falls_back_to_catalog_default(self, monkeypatch):
        main = _register(monkeypatch)
        assert main._resolve_engines_list(None, "some_future_engine") == ["zipformer_local"]

    def test_unregistered_default_model_falls_back_to_available_local(self, monkeypatch):
        """默认模型未下载时退回已注册的本地引擎, 而不是返回空列表。"""
        main = _register(monkeypatch, names=("sensevoice_local",))
        assert main._resolve_engines_list(None, "local") == ["sensevoice_local"]

    def test_no_local_engine_registered_returns_empty(self, monkeypatch):
        """一个本地模型都没下载时返回空列表, 让端点明确报错。"""
        main = _register(monkeypatch, names=())
        assert main._resolve_engines_list(None, "local") == []

    def test_cloud_never_leaks_into_local_fallback(self, monkeypatch):
        main = _register(monkeypatch, names=("cloud",))
        assert main._resolve_engines_list(None, "local") == []


class TestHealthLocalReady:
    """/health 的 model_loaded (local_ready 判断) 反映任一本地引擎可用。"""

    def _patch_engines(self, monkeypatch, ready_local=(), cloud_ready=False):
        """monkeypatch get_available_engines, 固定本地/云端引擎可用性。"""
        import app.main as main

        engines = []
        for name in ("zipformer_local", "sensevoice_local", "firered_local"):
            engines.append({"name": name, "available": name in ready_local})
        engines.append({"name": "cloud", "available": cloud_ready})
        monkeypatch.setattr(main, "get_available_engines", lambda: engines)
        return main

    @pytest.mark.parametrize(
        "engine", ["zipformer_local", "sensevoice_local", "firered_local"]
    )
    def test_local_ready_true_for_each_local_engine(self, monkeypatch, engine):
        main = self._patch_engines(monkeypatch, ready_local=(engine,))
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

    def test_model_status_reports_not_installed(self, monkeypatch):
        """无本地引擎时状态是 not_installed, 不再依赖已删除的下载状态表。"""
        main = self._patch_engines(monkeypatch)
        result = asyncio.run(main.health_check())
        assert result["model_status"] == "not_installed"


class TestTranscribeEndpoints:
    """三个 transcribe 端点把 engine_mode 传入 PipelineContext。"""

    @pytest.fixture
    def capture_pipeline(self, monkeypatch):
        """替换 SubtitlePipeline, 捕获构造时的 PipelineContext。

        同时固定引擎注册表: 路由现在只认已注册的引擎 (模型未下载即未注册),
        否则这些用例会因本机模型未安装而拿到空列表。
        """
        import app.main as main

        captured = {}
        _register(monkeypatch)

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
        monkeypatch.setattr(
            main, "_extract_audio_for_transcribe", lambda p, cfg=None: str(p)
        )
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

    def test_transcribe_file_defaults_to_catalog_default(self, capture_pipeline):
        """不传 engine_mode 时落到清单默认模型的引擎。"""
        from app.core import model_catalog

        import app.main as main

        asyncio.run(main.transcribe_file(
            file=self._upload(),
            language="auto",
            output_format="srt",
            glossary="{}",
            subtitle_offset_sec=0.0,
        ))
        expected = model_catalog.model_spec(model_catalog.DEFAULT_MODEL_ID)["engine_name"]
        assert capture_pipeline["ctx"].engines == [expected]

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

    def test_transcribe_local_uses_firered_engine_mode(self, capture_pipeline, tmp_path):
        import app.main as main

        media = tmp_path / "input.wav"
        media.write_bytes(b"fake")
        asyncio.run(main.transcribe_local(
            path=str(media),
            language="auto",
            output_format="srt",
            glossary="{}",
            engine_mode="firered_local",
            subtitle_offset_sec=0.0,
        ))
        assert capture_pipeline["ctx"].engines == ["firered_local"]
        assert capture_pipeline["ctx"].enable_deterministic_correction is True


class TestEngineSettingsEndpoint:
    """云端设置端点: 空 api_key 必须被接受, 且不得抹掉其它字段。

    回归: api_key 曾是必填 Form(...), 前端用掩码回显密钥、不改动时传空值,
    于是保存直接 422 —— 「空值保留旧密钥」的代码路径根本走不到。

    用 TestClient 走真实 FastAPI 表单解析: 直接调用函数会跳过 Form(...)
    的校验, 就测不出这个 bug。
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import app.main as main
        from app.core import cloud_engine

        monkeypatch.setattr(cloud_engine, "CONFIG_FILE", tmp_path / "config.json")
        return TestClient(main.app), tmp_path / "config.json"

    def test_empty_api_key_is_accepted(self, client):
        http, _ = client
        resp = http.post("/settings/engine", data={
            "api_url": "https://x/v1", "api_key": "",
            "model": "m", "enabled": "true",
        })
        assert resp.status_code == 200, resp.text

    def test_saving_keeps_request_mode_and_existing_key(self, client):
        import json as _json

        http, path = client
        path.write_text(_json.dumps({
            "engine": {"api_key": "sk-old", "request_mode": "chat_completions"}
        }, ensure_ascii=False), encoding="utf-8")

        resp = http.post("/settings/engine", data={
            "api_url": "https://x/v1", "api_key": "",
            "model": "m", "enabled": "true",
        })
        assert resp.status_code == 200, resp.text

        saved = _json.loads(path.read_text(encoding="utf-8"))["engine"]
        assert saved["request_mode"] == "chat_completions"
        assert saved["api_key"] == "sk-old"

    def test_get_returns_request_mode(self, client):
        import json as _json

        http, path = client
        path.write_text(_json.dumps({
            "engine": {"api_url": "u", "model": "m", "enabled": True,
                       "request_mode": "chat_completions", "display_name": "D"}
        }, ensure_ascii=False), encoding="utf-8")

        resp = http.get("/settings/engine")
        assert resp.status_code == 200
        assert resp.json()["request_mode"] == "chat_completions"
        assert resp.json()["display_name"] == "D"


class TestEngineMultiConfig:
    """云端多配置 (v2.4): engines 列表 + active_engine。

    回归背景: 原设计单槽位 (data["engine"]), 用户配第二个云模型会覆盖
    第一个 —— 多个云模型无法同时显示。现列表化, 激活项镜像到 engine
    字段保证旧读取路径 (transcribe_cloud) 不改也能读到激活配置。
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import app.main as main
        from app.core import cloud_engine

        monkeypatch.setattr(cloud_engine, "CONFIG_FILE", tmp_path / "config.json")
        return TestClient(main.app), tmp_path / "config.json"

    def test_save_list_and_activate(self, client):
        import json as _json

        http, path = client
        resp = http.post("/settings/engines", json={
            "engines": [
                {"id": "a", "api_url": "https://a/v1", "api_key": "sk-a",
                 "model": "m-a", "enabled": True, "display_name": "A"},
                {"id": "b", "api_url": "https://b/v1", "api_key": "sk-b",
                 "model": "m-b", "enabled": True, "display_name": "B"},
            ],
            "active": "b",
        })
        assert resp.status_code == 200, resp.text

        data = _json.loads(path.read_text(encoding="utf-8"))
        assert [e["id"] for e in data["engines"]] == ["a", "b"]
        assert data["active_engine"] == "b"
        # 激活项镜像到 engine 字段 (旧读取路径兼容)
        assert data["engine"]["display_name"] == "B"
        assert data["engine"]["api_key"] == "sk-b"
        assert "id" not in data["engine"]

    def test_list_masks_keys_and_reports_active(self, client):
        import json as _json

        http, path = client
        path.write_text(_json.dumps({
            "engines": [
                {"id": "a", "api_url": "https://a/v1", "api_key": "sk-a",
                 "model": "m-a", "enabled": True},
            ],
            "active_engine": "a",
        }), encoding="utf-8")

        data = http.get("/settings/engines").json()
        assert data["active"] == "a"
        assert data["engines"][0]["has_key"] is True
        assert "api_key" not in data["engines"][0]

    def test_empty_key_keeps_existing_secret(self, client):
        import json as _json

        http, path = client
        path.write_text(_json.dumps({
            "engines": [
                {"id": "a", "api_url": "https://a/v1", "api_key": "sk-keep",
                 "model": "m", "enabled": True},
            ],
            "active_engine": "a",
        }), encoding="utf-8")

        resp = http.post("/settings/engines", json={
            "engines": [{"id": "a", "api_url": "https://a/v1", "api_key": "",
                         "model": "m", "enabled": True}],
            "active": "a",
        })
        assert resp.status_code == 200, resp.text
        saved = _json.loads(path.read_text(encoding="utf-8"))["engines"][0]
        assert saved["api_key"] == "sk-keep"

    def test_legacy_engine_field_migrates_on_read(self, client):
        """旧配置 (只有 engine 字段) 读出来是一项, 不丢。"""
        import json as _json

        http, path = client
        path.write_text(_json.dumps({
            "engine": {"api_url": "https://old/v1", "api_key": "sk-old",
                       "model": "m", "enabled": True}
        }), encoding="utf-8")

        data = http.get("/settings/engines").json()
        assert len(data["engines"]) == 1
        assert data["engines"][0]["api_url"] == "https://old/v1"
        assert data["engines"][0]["has_key"] is True

    def test_each_config_registers_as_engine(self, client):
        """多个启用的云配置各注册一个引擎, 下拉里同时可见。"""
        import json as _json

        http, path = client
        path.write_text(_json.dumps({
            "engines": [
                {"id": "a", "api_url": "https://a/v1", "api_key": "k",
                 "model": "m-a", "enabled": True, "display_name": "甲"},
                {"id": "b", "api_url": "https://b/v1", "api_key": "k",
                 "model": "m-b", "enabled": True, "display_name": "乙"},
            ],
            "active_engine": "a",
        }), encoding="utf-8")

        # 触发重建注册表
        http.post("/settings/engines", json={
            "engines": [
                {"id": "a", "api_url": "https://a/v1", "api_key": "k",
                 "model": "m-a", "enabled": True, "display_name": "甲"},
                {"id": "b", "api_url": "https://b/v1", "api_key": "k",
                 "model": "m-b", "enabled": True, "display_name": "乙"},
            ],
            "active": "a",
        })

        engines = http.get("/engines").json()["engines"]
        names = {e["name"] for e in engines}
        assert "cloud" in names            # 激活项用旧名, 旧路由兼容
        assert "cloud:b" in names          # 其余 cloud:<id>
        by_name = {e["name"]: e for e in engines}
        assert by_name["cloud"]["display_name"].endswith("甲")
        assert by_name["cloud:b"]["display_name"].endswith("乙")
