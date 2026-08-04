"""
model_downloader 回归测试 (P0 修复)。

覆盖:
  1. ensure_model() 模型缺失时不再抛 NameError, 返回 downloading 并启动下载
  2. ensure_model() 模型完整时返回 ready
  3. 下载全部失败时状态置为 error
  4. _download_file() 大文件进度更新路径正常
  5. get_paraformer_status() 本地文件检查 (ready / not_ready)
  6. get_all_status() 返回 sensevoice/paraformer/punctuation/vad 四类

所有测试把 ~/.shiyu 目录重定向到 tmp_path, 不触碰真实用户目录。
"""
import threading

import pytest

from app.core import model_downloader as mdl


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """把 ~/.shiyu 与模型目录重定向到 tmp_path, 并重置模块级下载状态."""
    shiyu_dir = tmp_path / "shiyu"
    monkeypatch.setattr(mdl, "get_shiyu_data_dir", lambda: shiyu_dir)
    monkeypatch.setattr(
        mdl, "get_model_dir", lambda: shiyu_dir / "models" / "sensevoice-small"
    )
    monkeypatch.setattr(mdl, "_download_state", {"status": "idle", "progress": 0, "error": ""})
    monkeypatch.setattr(
        mdl, "_punctuation_state", {"status": "idle", "progress": 0, "error": ""}
    )
    return shiyu_dir


class _FakeThread:
    """同步执行 target 的假线程, 避免测试中真实起后台线程."""

    def __init__(self, target=None, daemon=False, name=None):
        self.target = target

    def start(self):
        self.target()


class _FakeResponse:
    """模拟 requests.Response: 2MB content-length, 流式返回 1MB 数据."""

    headers = {"content-length": str(2 * 1024 * 1024)}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for _ in range(128):
            yield b"\x00" * chunk_size


# ─────────────────────────────────────────────────────────────────────
# ensure_model() 状态机
# ─────────────────────────────────────────────────────────────────────

class TestEnsureModel:
    def test_ensure_model_returns_downloading_when_model_missing(
        self, monkeypatch, _isolate_dirs
    ):
        """模型缺失 → 不抛异常 (回归: _download_state 曾未定义导致 NameError), 返回 downloading 并启动下载."""
        monkeypatch.setattr(mdl, "model_exists", lambda: False)
        monkeypatch.setattr(mdl, "_try_migrate_from_bundle", lambda: False)
        started = []
        monkeypatch.setattr(mdl, "_download_worker", lambda: started.append(True))
        monkeypatch.setattr(threading, "Thread", _FakeThread)

        status = mdl.ensure_model()

        assert status == {"status": "downloading", "progress": 0, "error": ""}
        assert started == [True], "应启动后台下载 worker"

    def test_ensure_model_returns_ready_when_model_exists(self, monkeypatch, _isolate_dirs):
        """模型完整 → 返回 ready 状态."""
        monkeypatch.setattr(mdl, "model_exists", lambda: True)

        status = mdl.ensure_model()

        assert status == {"status": "ready", "progress": 100, "error": ""}

    def test_ensure_model_does_not_restart_when_downloading(self, monkeypatch, _isolate_dirs):
        """已处于 downloading → 直接返回当前状态, 不重复启动 worker."""
        monkeypatch.setattr(mdl, "model_exists", lambda: False)
        monkeypatch.setattr(mdl, "_try_migrate_from_bundle", lambda: False)
        monkeypatch.setattr(threading, "Thread", _FakeThread)
        monkeypatch.setattr(
            mdl, "_download_state", {"status": "downloading", "progress": 30, "error": ""}
        )
        started = []
        monkeypatch.setattr(mdl, "_download_worker", lambda: started.append(True))

        status = mdl.ensure_model()

        assert status["status"] == "downloading"
        assert started == [], "已在下载时不应再启动 worker"


# ─────────────────────────────────────────────────────────────────────
# 下载 worker / 文件下载
# ─────────────────────────────────────────────────────────────────────

class TestDownloadWorker:
    def test_download_worker_marks_error_when_all_sources_fail(self, monkeypatch, _isolate_dirs):
        """所有下载源失败 → 状态 error 且带错误信息."""
        monkeypatch.setattr(mdl, "_download_file", lambda *a, **k: False)
        monkeypatch.setattr(mdl, "_extract_onnx_from_github_release", lambda *a, **k: False)

        mdl._download_worker()

        assert mdl._download_state["status"] == "error"
        assert "Failed to download" in mdl._download_state["error"]

    def test_download_worker_marks_ready_on_success(self, monkeypatch, _isolate_dirs):
        """所有文件下载成功 → 状态 ready 且进度 100."""
        def fake_download_file(url, dst, progress_weight=1.0):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"\x00" * 2048)
            return True

        monkeypatch.setattr(mdl, "_download_file", fake_download_file)
        monkeypatch.setattr(mdl, "_extract_onnx_from_github_release", lambda *a, **k: False)

        mdl._download_worker()

        assert mdl._download_state == {"status": "ready", "progress": 100, "error": ""}

    def test_download_file_updates_progress_on_large_file(self, monkeypatch, _isolate_dirs):
        """大文件下载时进度更新不抛异常 (回归: _download_state 曾未定义导致 NameError 返回失败)."""
        monkeypatch.setattr(mdl.requests, "get", lambda *a, **k: _FakeResponse())
        dst = _isolate_dirs / "models" / "sensevoice-small" / "model.onnx"
        dst.parent.mkdir(parents=True, exist_ok=True)

        ok = mdl._download_file("http://fake/model.onnx", dst, progress_weight=0.85)

        assert ok is True
        assert dst.exists()
        assert mdl._download_state["progress"] > 0


# ─────────────────────────────────────────────────────────────────────
# get_paraformer_status()
# ─────────────────────────────────────────────────────────────────────

class TestParaformerStatus:
    def test_paraformer_status_ready_when_model_file_exists(self, _isolate_dirs):
        """model.int8.onnx 存在且 >100MB → ready."""
        model_path = _isolate_dirs / "models" / "paraformer-large" / "model.int8.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            f.truncate(101 * 1024 * 1024)  # 稀疏文件, 只占元数据空间

        status = mdl.get_paraformer_status()

        assert status == {"status": "ready", "progress": 100, "error": ""}

    def test_paraformer_status_not_ready_when_model_missing(self, _isolate_dirs):
        """模型文件不存在 → not_ready."""
        status = mdl.get_paraformer_status()

        assert status == {"status": "not_ready", "progress": 0, "error": ""}

    def test_paraformer_status_not_ready_when_model_too_small(self, _isolate_dirs):
        """模型文件存在但 <100MB → not_ready."""
        model_path = _isolate_dirs / "models" / "paraformer-large" / "model.int8.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"\x00" * 1024)

        status = mdl.get_paraformer_status()

        assert status["status"] == "not_ready"


# ─────────────────────────────────────────────────────────────────────
# get_all_status()
# ─────────────────────────────────────────────────────────────────────

class TestGetAllStatus:
    def test_get_all_status_includes_four_model_kinds(self, monkeypatch, _isolate_dirs):
        """get_all_status 返回 sensevoice/paraformer/punctuation/vad 四类 (回归: 曾缺 sensevoice/paraformer)."""
        monkeypatch.setattr(
            mdl, "get_download_status",
            lambda: {"status": "downloading", "progress": 42, "error": ""},
        )
        monkeypatch.setattr(
            mdl, "get_punctuation_status",
            lambda: {"status": "ready", "progress": 100, "error": ""},
        )
        monkeypatch.setattr(
            "app.core.vad_detector.get_download_status",
            lambda: {"status": "ready", "progress": 100, "error": ""},
        )

        result = mdl.get_all_status()

        assert set(result.keys()) == {"sensevoice", "paraformer", "punctuation", "vad"}
        assert result["sensevoice"]["status"] == "downloading"
        assert result["paraformer"]["status"] == "not_ready"
        assert result["punctuation"]["status"] == "ready"
        assert result["vad"]["status"] == "ready"
