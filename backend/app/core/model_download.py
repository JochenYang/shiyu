"""
模型下载与移除 — 固定校验值、断点续传、原子发布。

设计参照 SubPop 的 ``probes/model_download.py``:
  * 只发布完整且校验通过的字节; 失败或取消的残余留在 ``.part`` 供续传
  * HTTP Range 续传; 服务器忽略 Range (返回 200) 时干净重下, 不拼接错位数据
  * 下载前检查磁盘余量, 避免下一半才发现空间不足
  * 多镜像共用同一份上游 SHA-256 (镜像只是加速通道, 不改变校验值)
  * 移除是显式操作, 与安装对称

每个下载任务是后台线程, 进度通过模块级任务表暴露给 API/前端。
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import stat
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.core import model_catalog
from app.core.model_catalog import model_base, model_root, model_spec

logger = logging.getLogger(__name__)

# 镜像列表: 国内镜像优先, 官方源兜底。所有条目使用同一份上游 SHA-256。
SOURCES: Dict[str, List[tuple]] = {
    "auto": [
        ("国内镜像 HF-Mirror", "https://hf-mirror.com"),
        ("Hugging Face 官方", "https://huggingface.co"),
    ],
    "official": [("Hugging Face 官方", "https://huggingface.co")],
}

DOWNLOAD_TIMEOUT = 20          # 单次 HTTP 连接超时 (秒)
CHUNK_SIZE = 256 * 1024        # 读块大小
RESERVE_BYTES = 256 * 1024 * 1024  # 下载前预留的磁盘余量
PROGRESS_INTERVAL = 0.15       # 进度上报节流 (秒)


class Cancelled(Exception):
    """任务被用户取消 (残余保留在 .part, 重试可继续)。"""


def sources(value: str) -> List[tuple]:
    """把下载源标识解析为镜像列表, 未知名拒绝。"""
    if not isinstance(value, str) or value not in SOURCES:
        raise ValueError(f"未知的下载源: {value!r}")
    return SOURCES[value]


def _sha256_file(path: Path, cancel: Callable[[], bool]) -> str:
    """流式计算文件 SHA-256 (不整文件读入内存)。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel():
                raise Cancelled()
            block = f.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, digest: str, cancel: Callable[[], bool]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    return _sha256_file(path, cancel) == digest


def transfer(
    url: str,
    path: Path,
    size: int,
    digest: str,
    cancel: Callable[[], bool],
    progress: Callable[[int, str], None],
    opener=urllib.request.urlopen,
) -> None:
    """下载单个文件并原子发布。

    只有「大小一致 + SHA-256 匹配」的字节才会替换目标文件。任何失败或取消
    都保留 ``.part`` 残余, 下次调用从断点继续。

    Args:
        url: 下载地址 (含 pinned revision)。
        path: 目标文件路径。
        size: 期望字节数。
        digest: 期望 SHA-256 (小写十六进制)。
        cancel: 返回 True 时中止并保留残余。
        progress: progress(已完成字节数, 阶段) 回调。
        opener: 可注入的 URL 打开器 (测试用)。
    """
    partial = path.with_name(path.name + ".part")
    # 嵌套路径 (如 qwen3-asr 的 tokenizer/merges.txt) 需要先建父目录:
    # 模型根目录已存在, 但清单里的子目录不会自动创建。
    partial.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or partial.is_symlink():
        raise ValueError("目标文件是符号链接, 已拒绝")
    # 已存在且校验通过 → 无需下载
    if path.exists():
        if path.stat().st_size == size and _verify(path, digest, cancel):
            if partial.exists():
                partial.unlink()
            progress(size, "checking")
            return
        # 大小不符或校验失败: 删除不可信文件, 重新下载
        path.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    if cancel():
        raise Cancelled()

    if offset < size:
        headers = {"User-Agent": "Shiyu/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with opener(urllib.request.Request(url, headers=headers), timeout=DOWNLOAD_TIMEOUT) as response:
            status = getattr(response, "status", 200)
            if offset and status != 206:
                # 服务器忽略 Range: 丢弃残余, 从 0 重下, 避免拼接错位
                logger.info("[ModelDownload] 服务器忽略 Range, 从零重下: %s", url)
                offset = 0
            if status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("下载服务器返回了错误的续传范围")
            with partial.open("ab" if offset else "wb") as out:
                progress(offset, "resuming" if offset else "downloading")
                while True:
                    if cancel():
                        raise Cancelled()
                    block = response.read(CHUNK_SIZE)
                    if not block:
                        break
                    offset += len(block)
                    if offset > size:
                        raise ValueError("下载内容大于预期大小, 已中止")
                    out.write(block)
                    progress(offset, "downloading")

    progress(offset, "checking")
    if offset != size:
        raise ValueError(f"下载未完成 ({offset}/{size} 字节), 重试可继续")
    if not _verify(partial, digest, cancel):
        partial.unlink()
        raise ValueError("文件校验失败, 已删除损坏文件, 请重试下载")
    partial.replace(path)


def _is_reparse_point(path: Path) -> bool:
    """路径是否为符号链接或 Windows junction (重解析点)。

    ``Path.is_symlink()`` 对 Windows junction 返回 False, 而 ``.resolve()`` 会
    把 junction 静默展开为目标 —— 所以只检查 ``is_symlink()`` 会漏掉 junction,
    模型就被写到链接指向的地方去了。这里同时看重解析点属性。
    """
    if path.is_symlink():
        return True
    try:
        return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, OSError):
        # 非 Windows 或路径不存在: 没有重解析点概念
        return False


def _safe_directory(base: Path, name: str) -> Path:
    """在模型根目录下创建/取得子目录, 拒绝链接与越界。"""
    if _is_reparse_point(base):
        raise ValueError(f"模型根目录是符号链接或重解析点, 已拒绝: {base}")
    base.mkdir(parents=True, exist_ok=True)
    # 再次确认: mkdir 之后目录可能才被换成链接
    if _is_reparse_point(base):
        raise ValueError(f"模型根目录是链接或重解析点, 已拒绝: {base}")
    path = base / name
    if _is_reparse_point(path) or not path.resolve().is_relative_to(base.resolve()) or path.resolve() == base.resolve():
        raise ValueError(f"模型目录越界或是符号链接, 已拒绝: {path}")
    path.mkdir(exist_ok=True)
    return path

def _require_space(base: Path, needed: int) -> None:
    """下载前检查磁盘余量, 不足则明确报错而不是中途失败。"""
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free
    if free < needed + RESERVE_BYTES:
        raise ValueError(
            f"磁盘空间不足: 需要约 {needed / 1024 / 1024:.0f} MB, "
            f"可用 {free / 1024 / 1024:.0f} MB, 请腾出空间后重试"
        )


def install(
    model_id: str,
    root: Optional[Path] = None,
    cancel: Callable[[], bool] = lambda: False,
    emit: Callable[[Dict], None] = lambda _state: None,
    opener=urllib.request.urlopen,
    source: str = "auto",
) -> None:
    """下载并安装一个模型的全部文件。

    仅当所有文件都通过校验后才算成功; 任一步失败都会抛出异常, 已完成的
    文件保留 (重试时跳过已校验文件)。
    """
    endpoints = sources(source)
    spec = model_spec(model_id)
    # 不能用 root.resolve(): 它会静默展开 junction/符号链接, 让下面的链接
    # 检查失效。这里保留原路径, 由 _safe_directory 负责拒绝链接。
    base = model_root() if root is None else Path(root).expanduser().absolute()

    missing = sum(
        size
        for name, size in spec["files"].items()
        if not (base / spec["directory"] / name).is_file()
    )
    _require_space(base, missing)

    total = sum(spec["files"].values())
    done = 0
    started = time.monotonic()
    network = 0
    state = {"network": 0, "last_emit": 0.0}

    def run_files(target_dir: Path, files: Dict[str, int], digests: Dict[str, str], attempts_for) -> None:
        """按文件下载; attempts_for(name) 给出该文件的 (源名称, URL) 候选列表。"""
        nonlocal done
        for name, size in files.items():
            prior = 0

            def progress(value: int, stage: str) -> None:
                nonlocal prior, network
                if stage in ("downloading", "resuming"):
                    network += max(0, value - prior)
                prior = value
                now = time.monotonic()
                if now - state["last_emit"] >= PROGRESS_INTERVAL or stage == "checking":
                    emit({
                        "stage": stage,
                        "completedBytes": done + value,
                        "totalBytes": total,
                        "progress": (done + value) / total if total else 1.0,
                        "bytesPerSecond": network / max(now - started, 1e-6),
                        "file": name,
                    })
                    state["last_emit"] = now

            emit({
                "stage": "checking",
                "completedBytes": done,
                "totalBytes": total,
                "progress": done / total if total else 0.0,
                "file": name,
            })
            attempts = attempts_for(name)
            for index, (source_name, url) in enumerate(attempts):
                emit({
                    "source": source_name,
                    "stage": "connecting" if index == 0 else "switching-source",
                    "completedBytes": done,
                    "totalBytes": total,
                    "progress": done / total if total else 0.0,
                    "file": name,
                })
                try:
                    transfer(
                        url,
                        target_dir / name,
                        size,
                        digests[name],
                        cancel,
                        progress,
                        opener,
                    )
                    break
                except Cancelled:
                    raise
                except (OSError, ValueError):
                    # 当前镜像失败: 换下一个; 已是最后一个则把原因抛给调用方
                    if index + 1 == len(attempts):
                        raise
                    if cancel():
                        raise Cancelled()
            done += size

    model_dir = _safe_directory(base, spec["directory"])
    run_files(
        model_dir,
        spec["files"],
        spec["sha256"],
        lambda name: [
            (source_name, f"{endpoint}/{spec['repository']}/resolve/{spec['revision']}/{name}")
            for source_name, endpoint in endpoints
        ],
    )


    emit({
        "stage": "ready",
        "completedBytes": total,
        "totalBytes": total,
        "progress": 1.0,
        "bytesPerSecond": network / max(time.monotonic() - started, 1e-6),
    })


def remove(model_id: str, root: Optional[Path] = None) -> None:
    """移除一个模型的全部文件 (仅该模型的目录, 不动模型库其他内容)。

    复用 ``model_directory`` 的校验: 目录名必须是单个目录名、不得越界、
    不得是链接 —— 否则 ``rmtree`` 可能删掉整个模型库或链接目标。
    """
    base = model_base(root)
    path = model_catalog.model_directory(model_id, root)
    if _is_reparse_point(path) or _is_reparse_point(base):
        raise ValueError("模型目录或其根目录是符号链接, 已拒绝删除")
    if path == base or not path.is_relative_to(base):
        raise ValueError("模型目录越出模型根目录, 已拒绝删除")
    if path.exists():
        shutil.rmtree(path)
        logger.info("[ModelDownload] 已移除模型 %s: %s", model_id, path)


# ── 后台任务管理 (供 API 调用) ───────────────────────────────────────

_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()
MAX_JOBS = 20


def _prune_jobs() -> None:
    """只保留最近 MAX_JOBS 个已结束任务, 避免任务表无限增长。"""
    finished = [
        job_id for job_id, job in _jobs.items()
        if job["status"] in ("ready", "failed", "cancelled")
    ]
    order = sorted(
        finished,
        key=lambda job_id: _jobs[job_id]["startedAt"],
    )
    for job_id in order[:-MAX_JOBS]:
        _jobs.pop(job_id, None)

def _public(job: Dict) -> Dict:
    """去掉仅供内部使用的键 (取消事件不可序列化, 也不该下发给前端)。"""
    return {key: value for key, value in job.items() if key != "_cancel"}


def start_install(model_id: str, source: str = "auto", opener=urllib.request.urlopen) -> Dict:
    """启动后台安装任务, 立刻返回任务状态 (非阻塞)。

    Args:
        model_id: 清单中的模型 id。
        source: ``auto`` (国内镜像优先) 或 ``official`` (仅官方源)。
        opener: 可注入的 URL 打开器 (测试用)。
    """
    spec = model_spec(model_id)          # 未知 id 在此拒绝
    sources(source)                      # 未知下载源在此拒绝

    with _jobs_lock:
        for job in _jobs.values():
            if job["modelID"] == model_id and job["status"] == "running":
                return _public(job)
        job_id = str(uuid.uuid4())
        job = {
            "jobID": job_id,
            "modelID": model_id,
            "modelName": spec["name"],
            "operation": "install",
            "status": "running",
            "stage": "connecting",
            "progress": 0.0,
            "completedBytes": 0,
            "totalBytes": sum(spec["files"].values()),
            "error": "",
            "startedAt": time.time(),
        }
        _jobs[job_id] = job
        _prune_jobs()

    cancel_flag = threading.Event()
    with _jobs_lock:
        _jobs[job_id]["_cancel"] = cancel_flag

    def emit(value: Dict) -> None:
        with _jobs_lock:
            target = _jobs.get(job_id)
            if target is not None:
                target.update(value)

    def worker() -> None:
        try:
            install(
                model_id,
                cancel=cancel_flag.is_set,
                emit=emit,
                source=source,
                opener=opener,
            )
            emit({"status": "ready", "stage": "ready", "progress": 1.0})
        except Cancelled:
            emit({
                "status": "cancelled",
                "stage": "cancelled",
                "error": "已取消下载, 重试可继续",
            })
        except urllib.error.URLError as e:
            logger.warning("[ModelDownload] 网络失败 %s: %s", model_id, e)
            emit({
                "status": "failed",
                "stage": "failed",
                "error": "无法连接模型下载服务器, 请检查网络后重试",
                "detail": str(e)[:500],
            })
        except TimeoutError as e:
            emit({
                "status": "failed",
                "stage": "failed",
                "error": "下载连接超时, 重试可继续下载",
                "detail": str(e)[:500],
            })
        except OSError as e:
            logger.warning("[ModelDownload] 文件失败 %s: %s", model_id, e)
            emit({
                "status": "failed",
                "stage": "failed",
                "error": "模型文件读写失败, 请检查磁盘空间和文件权限",
                "detail": str(e)[:500],
            })
        except Exception as e:  # 校验失败 / 未知错误
            logger.warning("[ModelDownload] 安装失败 %s: %s", model_id, e)
            emit({
                "status": "failed",
                "stage": "failed",
                "error": str(e)[:300],
                "detail": str(e)[:500],
            })

    thread = threading.Thread(
        target=worker, daemon=True, name=f"model-install-{model_id}"
    )
    thread.start()
    return _public(job)


def cancel_job(job_id: str) -> Dict:
    """请求取消一个运行中的下载任务。

    置位任务自己的取消事件 —— 下载线程在读写块之间检查它, 取消后已下载的
    ``.part`` 残余保留, 重试时从断点继续。
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise ValueError("未找到该下载任务")
        if job["status"] != "running":
            return _public(job)
        flag = job.get("_cancel")
        job["stage"] = "cancelling"
        snapshot = _public(job)
    if flag is not None:
        flag.set()
    return snapshot


def get_jobs() -> List[Dict]:
    """返回所有任务状态 (新任务在前)。"""
    with _jobs_lock:
        return sorted(
            (_public(job) for job in _jobs.values()),
            key=lambda job: job["startedAt"],
            reverse=True,
        )


def active_job() -> Optional[Dict]:
    """返回当前运行中的任务 (一次只跑一个模型下载)。"""
    with _jobs_lock:
        for job in _jobs.values():
            if job["status"] == "running":
                return _public(job)
    return None


def start_remove(model_id: str) -> Dict:
    """移除模型 (同步执行, 体积删除很快)。

    只要该模型还在下载中就不允许移除 —— 删除正在写入的 ``.part`` 会在
    Windows 上直接报 PermissionError (文件被占用)。注意要检查**全部**运行中
    的任务, 而不是只看第一个: 不同模型是可以并发下载的。

    Raises:
        ValueError: 模型 id 未知, 或该模型正在下载。
    """
    spec = model_spec(model_id)
    with _jobs_lock:
        downloading = [
            job["jobID"]
            for job in _jobs.values()
            if job["status"] == "running" and job["modelID"] == model_id
        ]
    if downloading:
        raise ValueError("该模型正在下载, 请先取消下载再移除")
    remove(model_id)
    return {
        "modelID": model_id,
        "modelName": spec["name"],
        "operation": "remove",
        "status": "ready",
        "stage": "ready",
        "progress": 1.0,
    }