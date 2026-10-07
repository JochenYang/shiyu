"""
模型清单与下载层测试。

覆盖:
  1. 清单完整性: 每个模型都有 repository / pinned revision / 每文件 SHA-256
  2. 未知模型 id 与未知下载源一律拒绝
  3. 路径安全: 符号链接与越界目录被拒
  4. transfer(): 断点续传、服务器忽略 Range、校验失败不留残骸、取消保留 .part
  5. install(): 多镜像回退、磁盘不足提前报错
  6. remove(): 删除模型目录但保留模型根目录

所有测试把模型根目录重定向到 tmp_path, 不触碰真实用户目录, 也不联网。
"""
import hashlib
import io
import threading
import time
from pathlib import Path

import pytest

from app.core import model_catalog, model_download


# ── 测试替身 ─────────────────────────────────────────────────────────


class FakeResponse(io.BytesIO):
    """最小 HTTP 响应替身: 支持 status / headers / read。"""

    def __init__(self, data: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _force_symlink(monkeypatch, path: Path) -> None:
    """把指定路径伪装成符号链接。

    Windows 非特权进程无法创建符号链接 (WinError 1314), 因此这里让
    ``Path.is_symlink`` 对该路径返回 True, 以确定性地验证拒绝逻辑本身。
    真实的符号链接行为另有一条测试, 在系统不允许时跳过。
    """
    real = Path.is_symlink
    target = Path(path)

    def fake(self):
        if self == target:
            return True
        return real(self)

    monkeypatch.setattr(Path, "is_symlink", fake)


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    """把模型根目录指向 tmp_path。"""
    root = tmp_path / "models"
    monkeypatch.setenv("SHIYU_MODEL_ROOT", str(root))
    return root


@pytest.fixture
def tiny_catalog(monkeypatch):
    """把清单替换成一个只有一个 6 字节文件的测试模型。"""
    payload = b"abcdef"
    spec = {
        "id": "test-model",
        "name": "Test Model",
        "engine": "zipformer",
        "directory": "test-model",
        "repository": "example/test-model",
        "revision": "pinned-rev",
        "description": "",
        "languages": ["zh"],
        "files": {"weights.bin": len(payload)},
        "sha256": {"weights.bin": hashlib.sha256(payload).hexdigest()},
    }
    monkeypatch.setattr(
        model_catalog,
        "CATALOG",
        {"defaultModelID": "test-model", "models": [spec]},
    )
    return spec, payload



# ── 清单 ─────────────────────────────────────────────────────────────


class TestCatalog:
    def test_every_model_is_pinned_with_hashes(self):
        """每个模型都必须有 pinned revision 与逐文件 SHA-256 (清单式管理的前提)。"""
        for spec in model_catalog.CATALOG["models"]:
            assert spec["repository"], spec["id"]
            # revision 必须是 40 位十六进制 commit, 不接受 main/master 这类浮动引用
            assert len(spec["revision"]) == 40, spec["id"]
            assert all(c in "0123456789abcdef" for c in spec["revision"]), spec["id"]
            assert set(spec["files"]) == set(spec["sha256"]), spec["id"]
            for name, size in spec["files"].items():
                assert isinstance(size, int) and size > 0, (spec["id"], name)
                digest = spec["sha256"][name]
                assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (spec["id"], name)

    def test_directories_are_unique_and_default_exists(self):
        ids = [m["id"] for m in model_catalog.CATALOG["models"]]
        assert len(set(ids)) == len(ids)
        directories = [m["directory"] for m in model_catalog.CATALOG["models"]]
        assert len(set(directories)) == len(directories)
        assert model_catalog.DEFAULT_MODEL_ID in ids

    def test_unknown_model_id_is_rejected(self):
        """未知 id 绝不回退到默认模型。"""
        for bad in ("../../etc/passwd", "", None, "no-such-model"):
            with pytest.raises(ValueError):
                model_catalog.model_spec(bad)

    def test_unknown_download_source_is_rejected(self):
        with pytest.raises(ValueError):
            model_download.sources("ftp://evil")
        assert model_download.sources("auto")
        assert model_download.sources("official")

    def test_missing_files_reports_size_mismatch(self, model_root, tiny_catalog):
        """文件缺失或大小不符都算未安装。"""
        spec, payload = tiny_catalog
        assert model_catalog.is_installed("test-model") is False

        directory = model_root / spec["directory"]
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(payload[:-1])  # 少一个字节
        assert model_catalog.is_installed("test-model") is False
        assert "weights.bin" in model_catalog.missing_files("test-model")

        (directory / "weights.bin").write_bytes(payload)
        assert model_catalog.is_installed("test-model") is True
        assert model_catalog.missing_files("test-model") == {}

    def test_resolve_rejects_incomplete_model(self, model_root, tiny_catalog):
        with pytest.raises(ValueError, match="未安装"):
            model_catalog.resolve_model("test-model")


# ── 路径安全 ─────────────────────────────────────────────────────────


class TestPathSafety:
    def test_symlinked_model_directory_is_refused(self, model_root, tiny_catalog, monkeypatch):
        """模型目录是符号链接时拒绝解析 (避免写到别处)。"""
        spec, _ = tiny_catalog
        model_root.mkdir(parents=True, exist_ok=True)
        _force_symlink(monkeypatch, model_root / spec["directory"])
        with pytest.raises(ValueError, match="符号链接"):
            model_catalog.model_directory("test-model")

    def test_symlinked_model_directory_is_refused_for_real(self, model_root, tiny_catalog):
        """真实符号链接同样被拒 (系统不允许创建链接时跳过)。"""
        spec, _ = tiny_catalog
        model_root.mkdir(parents=True, exist_ok=True)
        elsewhere = model_root.parent / "elsewhere"
        elsewhere.mkdir()
        link = model_root / spec["directory"]
        try:
            link.symlink_to(elsewhere, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("当前系统不允许创建符号链接 (需要管理员权限)")
        with pytest.raises(ValueError, match="符号链接"):
            model_catalog.model_directory("test-model")

    def test_install_refuses_symlinked_root(self, model_root, tiny_catalog, monkeypatch):
        """模型根目录本身是符号链接时拒绝安装。"""
        model_root.parent.mkdir(parents=True, exist_ok=True)
        _force_symlink(monkeypatch, model_root)
        with pytest.raises(ValueError, match="符号链接"):
            model_download.install("test-model")


# ── transfer() ───────────────────────────────────────────────────────


class TestTransfer:
    def test_resumes_from_partial_and_publishes_verified_bytes(self, tmp_path):
        """已有 .part 残余时带 Range 续传, 校验通过才发布。"""
        target = tmp_path / "weights.bin"
        target.with_name("weights.bin.part").write_bytes(b"abc")
        seen_headers = []

        def opener(request, timeout):
            seen_headers.append(request.get_header("Range"))
            return FakeResponse(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})

        model_download.transfer(
            "https://example.test/f", target, 6, hashlib.sha256(b"abcdef").hexdigest(),
            lambda: False, lambda *_: None, opener,
        )
        assert target.read_bytes() == b"abcdef"
        assert seen_headers == ["bytes=3-"]
        assert not target.with_name("weights.bin.part").exists()

    def test_server_ignoring_range_restarts_cleanly(self, tmp_path):
        """服务器返回 200 (忽略 Range) 时丢弃残余重下, 不拼接错位数据。"""
        target = tmp_path / "weights.bin"
        target.with_name("weights.bin.part").write_bytes(b"XXX")
        model_download.transfer(
            "https://example.test/f", target, 6, hashlib.sha256(b"abcdef").hexdigest(),
            lambda: False, lambda *_: None, lambda *a, **k: FakeResponse(b"abcdef"),
        )
        assert target.read_bytes() == b"abcdef"

    def test_wrong_content_range_is_rejected(self, tmp_path):
        """206 的 Content-Range 与请求偏移不符时拒绝, 避免错位拼接。"""
        target = tmp_path / "weights.bin"
        target.with_name("weights.bin.part").write_bytes(b"abc")
        with pytest.raises(ValueError, match="续传范围"):
            model_download.transfer(
                "https://example.test/f", target, 6, hashlib.sha256(b"abcdef").hexdigest(),
                lambda: False, lambda *_: None,
                lambda *a, **k: FakeResponse(b"def", status=206, headers={"Content-Range": "bytes 0-2/6"}),
            )

    def test_bad_hash_never_publishes_and_leaves_no_corrupt_file(self, tmp_path):
        """校验失败: 目标文件不存在, 损坏的 .part 被删除。"""
        target = tmp_path / "weights.bin"
        with pytest.raises(ValueError, match="校验失败"):
            model_download.transfer(
                "https://example.test/f", target, 6, hashlib.sha256(b"abcdef").hexdigest(),
                lambda: False, lambda *_: None, lambda *a, **k: FakeResponse(b"XXXXXX"),
            )
        assert not target.exists()
        assert not target.with_name("weights.bin.part").exists()

    def test_cancel_keeps_partial_for_resume(self, tmp_path):
        """取消时保留 .part 且不发布目标文件。"""
        target = tmp_path / "weights.bin"
        target.with_name("weights.bin.part").write_bytes(b"ab")
        with pytest.raises(model_download.Cancelled):
            model_download.transfer(
                "https://example.test/f", target, 6, "unused",
                lambda: True, lambda *_: None,
            )
        assert not target.exists()
        assert target.with_name("weights.bin.part").read_bytes() == b"ab"

    def test_already_complete_file_is_not_redownloaded(self, tmp_path):
        """已存在且校验通过的文件不重新下载 (重复安装是幂等的)。"""
        payload = b"abcdef"
        target = tmp_path / "weights.bin"
        target.write_bytes(payload)

        def opener(*a, **k):
            raise AssertionError("不应发起网络请求")

        model_download.transfer(
            "https://example.test/f", target, len(payload),
            hashlib.sha256(payload).hexdigest(), lambda: False, lambda *_: None, opener,
        )

    def test_corrupt_existing_file_is_replaced(self, tmp_path):
        """已存在但内容损坏的文件被重下替换, 不会被当成已安装。"""
        target = tmp_path / "weights.bin"
        target.write_bytes(b"corrupt")
        model_download.transfer(
            "https://example.test/f", target, 6, hashlib.sha256(b"abcdef").hexdigest(),
            lambda: False, lambda *_: None, lambda *a, **k: FakeResponse(b"abcdef"),
        )
        assert target.read_bytes() == b"abcdef"

    def test_linked_destination_is_refused(self, tmp_path, monkeypatch):
        """目标文件是符号链接时拒绝写入 (不覆盖链接指向的文件)。"""
        real = tmp_path / "real.bin"
        real.write_bytes(b"keep")
        target = tmp_path / "weights.bin"
        _force_symlink(monkeypatch, target)
        with pytest.raises(ValueError, match="符号链接"):
            model_download.transfer(
                "https://example.test/f", target, 4, "unused",
                lambda: False, lambda *_: None, lambda *a, **k: FakeResponse(b"keep"),
            )
        assert real.read_bytes() == b"keep"



# ── install() / remove() ─────────────────────────────────────────────


class TestInstallRemove:
    def test_install_falls_back_to_second_mirror(self, model_root, tiny_catalog):
        """首个镜像失败时换下一个镜像, 校验值不变。"""
        spec, payload = tiny_catalog
        attempted = []

        def opener(request, timeout):
            url = request.full_url
            attempted.append(url)
            if "hf-mirror.com" in url:
                raise OSError("镜像不可达")
            return FakeResponse(payload)

        model_download.install("test-model", cancel=lambda: False, opener=opener)
        assert (model_root / spec["directory"] / "weights.bin").read_bytes() == payload
        assert any("hf-mirror.com" in u for u in attempted)
        assert any("huggingface.co" in u for u in attempted)
        # pinned revision 必须出现在下载地址里, 不能落到 main
        assert all("/resolve/pinned-rev/" in u for u in attempted)

    def test_install_raises_when_all_mirrors_fail(self, model_root, tiny_catalog):
        """所有镜像都失败时抛出异常, 不静默成功。"""
        def opener(*a, **k):
            raise OSError("network down")

        with pytest.raises(OSError):
            model_download.install("test-model", cancel=lambda: False, opener=opener)

    def test_install_reports_insufficient_disk_space(self, model_root, tiny_catalog, monkeypatch):
        """磁盘余量不足时提前报错, 不下载到一半才失败。"""
        monkeypatch.setattr(
            model_download.shutil, "disk_usage",
            lambda _path: type("U", (), {"free": 1024})(),
        )
        with pytest.raises(ValueError, match="磁盘空间不足"):
            model_download.install("test-model", cancel=lambda: False, opener=lambda *a, **k: None)

    def test_install_is_idempotent_when_already_complete(self, model_root, tiny_catalog):
        """模型已完整时重复安装不报错 (文件校验通过即跳过)。"""
        spec, payload = tiny_catalog
        directory = model_root / spec["directory"]
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(payload)

        def opener(*a, **k):
            raise AssertionError("不应发起网络请求")

        model_download.install("test-model", cancel=lambda: False, opener=opener)

    def test_remove_deletes_model_but_keeps_root(self, model_root, tiny_catalog):
        spec, payload = tiny_catalog
        directory = model_root / spec["directory"]
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(payload)

        model_download.remove("test-model")

        assert not directory.exists()
        assert model_root.exists()
        assert model_catalog.is_installed("test-model") is False

    def test_remove_refuses_symlinked_directory(self, model_root, tiny_catalog, monkeypatch):
        """模型目录是符号链接时拒绝删除, 避免 rmtree 误删链接目标。"""
        spec, _ = tiny_catalog
        model_root.mkdir(parents=True, exist_ok=True)
        elsewhere = model_root.parent / "keep-me"
        elsewhere.mkdir()
        (elsewhere / "important.txt").write_text("keep")
        _force_symlink(monkeypatch, model_root / spec["directory"])

        with pytest.raises(ValueError, match="符号链接"):
            model_download.remove("test-model")
        assert (elsewhere / "important.txt").read_text() == "keep"


    def test_remove_unknown_model_is_rejected(self, model_root):
        with pytest.raises(ValueError):
            model_download.remove("no-such-model")


def _never_opener():
    """测试用 opener: 立即失败, 让后台任务快速结束且不触网。"""
    def opener(*args, **kwargs):
        raise OSError("测试环境不联网")
    return opener

# ── 后台任务与取消 ────────────────────────────────────────────────


class TestJobControl:
    """后台下载任务的状态与取消。

    回归: cancel_job 曾经只把 stage 改成 cancelling 就返回, 从未置位下载线程
    的取消事件, 下载会一路跑完 —— 用户以为已取消, 实际仍在下载。
    """

    def _blocking_opener(self, payload: bytes, released: "threading.Event"):
        """返回一个在首块后阻塞的 opener, 直到测试释放它 —— 让取消有确定的窗口。"""
        class BlockingResponse(io.BytesIO):
            def __init__(self):
                super().__init__(payload)
                self.status = 200
                self.headers = {}
                self._first = True

            def read(self, size=-1):
                if self._first:
                    self._first = False
                    return super().read(min(size, 1024))
                # 卡住直到测试放行 (或超时兜底), 期间下载线程会轮询取消标志
                released.wait(timeout=10)
                return b""

        return lambda *a, **k: BlockingResponse()

    def test_cancel_stops_download_and_keeps_partial(self, model_root, tiny_catalog, monkeypatch):
        spec, payload = tiny_catalog
        # 把文件放大到能分多块下载
        big = payload * 100000
        monkeypatch.setitem(spec["files"], "weights.bin", len(big))
        monkeypatch.setitem(spec["sha256"], "weights.bin", hashlib.sha256(big).hexdigest())

        released = threading.Event()
        job = model_download.start_install("test-model", opener=self._blocking_opener(big, released))
        job_id = job["jobID"]

        # 等下载真正开始 (首块已写入 .part) 再取消
        for _ in range(100):
            time.sleep(0.02)
            if list(model_root.rglob("*.part")):
                break
        model_download.cancel_job(job_id)
        released.set()  # 放行阻塞的读, 让下载线程观察到取消

        deadline = time.time() + 5
        while time.time() < deadline:
            current = next(j for j in model_download.get_jobs() if j["jobID"] == job_id)
            if current["status"] != "running":
                break
            time.sleep(0.05)

        assert current["status"] == "cancelled", current
        # 未完成: 目标文件不应存在, 残余保留供续传
        assert not (model_root / spec["directory"] / "weights.bin").exists()
        assert (model_root / spec["directory"] / "weights.bin.part").exists()

    def test_job_dicts_are_json_serializable(self, model_root, tiny_catalog):
        """任务 dict 不得携带锁/事件等不可序列化对象 (否则 API 返回 500)。"""
        import json

        model_download.start_install("test-model", opener=_never_opener())
        for job in model_download.get_jobs():
            json.dumps(job)  # 不抛异常即通过
        active = model_download.active_job()
        if active is not None:
            json.dumps(active)

    def test_cancel_unknown_job_is_rejected(self, model_root):
        with pytest.raises(ValueError):
            model_download.cancel_job("no-such-job")

    def test_duplicate_install_returns_running_job(self, model_root, tiny_catalog, monkeypatch):
        """同一模型已在下载时不重复启动新任务。"""
        spec, payload = tiny_catalog
        big = payload * 100000
        monkeypatch.setitem(spec["files"], "weights.bin", len(big))
        monkeypatch.setitem(spec["sha256"], "weights.bin", hashlib.sha256(big).hexdigest())
        released = threading.Event()
        opener = self._blocking_opener(big, released)

        try:
            first = model_download.start_install("test-model", opener=opener)
            second = model_download.start_install("test-model", opener=opener)
            assert first["jobID"] == second["jobID"]
            assert second["status"] == "running"
        finally:
            released.set()
            for job in model_download.get_jobs():
                if job["status"] == "running":
                    model_download.cancel_job(job["jobID"])

    def test_remove_refuses_while_downloading(self, model_root, tiny_catalog, monkeypatch):
        """正在下载的模型不能直接移除 (先取消)。"""
        spec, payload = tiny_catalog
        big = payload * 100000
        monkeypatch.setitem(spec["files"], "weights.bin", len(big))
        monkeypatch.setitem(spec["sha256"], "weights.bin", hashlib.sha256(big).hexdigest())
        released = threading.Event()
        job = model_download.start_install("test-model", opener=self._blocking_opener(big, released))
        try:
            with pytest.raises(ValueError, match="正在下载"):
                model_download.start_remove("test-model")
        finally:
            released.set()
            model_download.cancel_job(job["jobID"])


    def test_unknown_model_install_is_rejected(self, model_root):
        with pytest.raises(ValueError):
            model_download.start_install("no-such-model")

    def test_unknown_source_install_is_rejected(self, model_root, tiny_catalog):
        with pytest.raises(ValueError, match="下载源"):
            model_download.start_install("test-model", "ftp://evil")
