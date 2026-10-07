"""
模型清单 — 固定版本、逐文件校验的本地模型目录。

设计参照 SubPop 的 ``config/models.json`` 与 ``probes/models.py``:
  * 每个模型声明 repository + pinned revision + 每文件精确字节数与 SHA-256
  * 可用性判定只做「文件存在 + 大小一致」(快), 完整 SHA-256 校验在下载发布时做
  * 可用性查询永远不会触发下载, 下载必须由用户显式发起

模型目录布局::

    ~/.shiyu/models/
      zipformer-ctc-zh/model.int8.onnx, tokens.txt
      sensevoice-small/model.int8.onnx, tokens.txt
      fire-red-asr2-ctc/model.int8.onnx, tokens.txt
      silero-vad/silero_vad.onnx

清单文件为 ``backend/config/models.json``。新增模型只改清单, 不动代码。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 清单相对 backend/ 定位: 打包后 config/ 与 app/ 同级
_CODE_ROOT = Path(__file__).resolve().parent.parent.parent

CATALOG_PATH = Path(
    os.environ.get("SHIYU_MODEL_CATALOG", str(_CODE_ROOT / "config" / "models.json"))
)


def _load_catalog() -> Dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CATALOG = _load_catalog()
DEFAULT_MODEL_ID = CATALOG["defaultModelID"]

# 用户设置的默认模型 (运行时可改); 空表示用清单默认
# 存在 ~/.shiyu/config.json 的 default_model 字段, 不回写清单文件
# —— 清单是随安装包分发的, 用户数据不该混进去。
_USER_DEFAULT_FILE = Path(os.environ.get(
    "SHIYU_CONFIG_FILE", str(Path.home() / ".shiyu" / "config.json")
))


def default_model_id() -> str:
    """当前默认模型: 用户设置优先, 否则清单默认。"""
    try:
        import json as _json

        with open(_USER_DEFAULT_FILE, "r", encoding="utf-8") as f:
            saved = _json.load(f).get("default_model")
        if saved and model_spec(str(saved)):
            return str(saved)
    except (OSError, ValueError, ImportError):
        pass
    return DEFAULT_MODEL_ID


def set_default_model(model_id: str) -> None:
    """设置默认模型 (持久化到用户配置); 未知 id 拒绝。"""
    spec = model_spec(model_id)  # 未知 id 在此抛 ValueError (含对齐器)
    if spec is ALIGNER_SPEC:
        raise ValueError("对齐器不是识别引擎, 不能设为默认模型")
    _USER_DEFAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if _USER_DEFAULT_FILE.exists():
        try:
            with open(_USER_DEFAULT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data["default_model"] = model_id
    tmp = _USER_DEFAULT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(_USER_DEFAULT_FILE)


# 对齐器不是 ASR 引擎: 它拿「音频 + 已识别文本」产出逐词时间,
# 给 sherpa-onnx 不返回 token 时间戳的模型 (Qwen3-ASR) 补上可断句的时间。
# 与 models 数组平级 (参照 SubPop 的清单结构), 复用同一套下载/校验/移除机制。
ALIGNER_SPEC: Optional[Dict] = CATALOG.get("aligner")

# 模型根目录 (~/.shiyu/models), 可用 SHIYU_MODEL_ROOT 覆盖以便测试隔离
DEFAULT_MODEL_ROOT = Path.home() / ".shiyu" / "models"


def model_root() -> Path:
    """返回模型根目录 (展开 ``~`` 并转为绝对路径)。

    刻意**不**调用 ``.resolve()``: 它会静默展开符号链接与 Windows junction,
    使下载层无从判断模型库是否被链到别处 (那样模型会被写到链接指向的位置)。
    规范化交给 ``Path.absolute()``, 链接检查交给 ``model_download``。
    """
    raw = os.environ.get("SHIYU_MODEL_ROOT", str(DEFAULT_MODEL_ROOT))
    return Path(os.path.expanduser(raw)).absolute()


def model_ids() -> List[str]:
    """返回清单中所有模型 id (声明顺序)。"""
    return [m["id"] for m in CATALOG["models"]]


def model_spec(model_id: str) -> Dict:
    """按 id 取模型声明 (含对齐器), 未知 id 直接报错。"""
    if not isinstance(model_id, str):
        raise ValueError("模型 id 必须是字符串")
    for model in CATALOG["models"]:
        if model["id"] == model_id:
            return model
    # 对齐器与模型共用同一套下载/校验/移除机制, 所以这里也认它的 id
    if ALIGNER_SPEC and model_id == ALIGNER_SPEC["id"]:
        return ALIGNER_SPEC
    raise ValueError(
        f"未知的模型 id: {model_id}; 可选: {', '.join(model_ids())}"
    )


def model_size(model_id: str) -> int:
    """模型全部文件的字节数合计 (用于下载前显示与磁盘检查)。"""
    spec = model_spec(model_id)
    return sum(spec["files"].values())


def total_size() -> int:
    """全部模型的字节数合计。"""
    return sum(m["files"][k] for m in CATALOG["models"] for k in m["files"])



# ── 路径安全 ─────────────────────────────────────────────────────────


def model_base(root: Optional[Path] = None) -> Path:
    """模型根目录 (不 resolve, 保留链接身份以便检查)。"""
    return Path(root).absolute() if root is not None else model_root()

def model_directory(model_id: str, root: Optional[Path] = None) -> Path:
    """模型目录 (已校验未越出根目录、非符号链接)。

    ``directory`` 必须是**单个目录名**: 不得含路径分隔符, 不得是 ``.``/``..``。
    仅靠 ``resolve()`` 后比较是不够的 —— ``"x/.."`` 会解析回根目录本身, 既是
    相对路径又等于根目录, 会被当成合法模型目录传给 ``shutil.rmtree`` 而删掉
    整个模型库。

    Returns:
        解析后的模型目录 (绝对路径), 一定位于根目录之下且不等于根目录。

    Raises:
        ValueError: ``directory`` 不合法、越界或是符号链接。
    """
    spec = model_spec(model_id)
    base = model_base(root)
    name = spec["directory"]

    if not isinstance(name, str) or not name:
        raise ValueError(f"模型目录名不合法: {name!r}")
    # 只允许单个目录名: 拒绝分隔符与相对路径成分
    if name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"模型目录名必须是单个目录名, 已拒绝: {name!r}")
    if Path(name).is_absolute() or Path(name).drive:
        raise ValueError(f"模型目录名不得为绝对路径, 已拒绝: {name!r}")

    path = base / name
    if path.is_symlink():
        raise ValueError(f"模型目录是符号链接, 已拒绝: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(base) or resolved == base:
        raise ValueError(f"模型目录越出模型根目录, 已拒绝: {path}")
    return resolved


def _check_file(path: Path, size: int, base: Path) -> bool:
    """单文件校验: 非符号链接、是普通文件、大小一致。"""
    if path.is_symlink():
        return False
    if not path.resolve().is_relative_to(base.resolve()):
        return False
    if not path.is_file():
        return False
    return path.stat().st_size == size


def missing_files(model_id: str, root: Optional[Path] = None) -> Dict[str, int]:
    """返回缺失/不完整的文件名 → 期望字节数; 空字典表示模型完整。"""
    spec = model_spec(model_id)
    base = model_base(root)
    directory = base / spec["directory"]
    missing: Dict[str, int] = {}
    for name, size in spec["files"].items():
        if not _check_file(directory / name, size, base):
            missing[name] = size
    return missing


def is_installed(model_id: str, root: Optional[Path] = None) -> bool:
    """模型是否完整可用 (只比对大小, 不做 SHA-256 全文校验)。"""
    try:
        return not missing_files(model_id, root)
    except (ValueError, OSError):
        return False


def resolve_files(model_id: str, root: Optional[Path] = None) -> Dict[str, str]:
    """解析模型为 ``{清单内相对路径: 绝对路径}``。

    清单里的文件名并不统一: 单文件模型是 ``model.int8.onnx`` + ``tokens.txt``,
    Whisper 是 encoder/decoder 两个 onnx, Qwen3-ASR 还要 conv_frontend 与
    tokenizer/ 子目录。因此这里按清单声明的名字逐个解析, 不在代码里写死文件名。

    不完整时抛 ValueError, 由调用方决定是提示下载还是报错。
    """
    spec = model_spec(model_id)
    directory = model_directory(model_id, root)
    missing = missing_files(model_id, root)
    if missing:
        raise ValueError(
            f"模型「{spec['name']}」未安装或不完整, 缺少: {', '.join(sorted(missing))}"
        )
    return {name: str(directory / name) for name in spec["files"]}


def resolve_model(model_id: str, root: Optional[Path] = None) -> Tuple[str, str]:
    """单文件模型的快捷解析, 返回 (模型文件路径, tokens 路径)。

    仅适用于 Zipformer / SenseVoice / FireRedASR2 这类「一个 onnx + 一份词表」
    的模型; Whisper 与 Qwen3-ASR 请用 :func:`resolve_files`。
    """
    files = resolve_files(model_id, root)
    spec = model_spec(model_id)
    if "model.int8.onnx" not in files or "tokens.txt" not in files:
        raise ValueError(
            f"模型「{spec['name']}」不是单文件结构, 请用 resolve_files 解析"
        )
    return files["model.int8.onnx"], files["tokens.txt"]


# ── 可用性 (绝不触发下载) ────────────────────────────────────────────


def availability(root: Optional[Path] = None) -> List[Dict]:
    """列出每个模型的安装状态, 供 /models 接口与前端模型面板使用。"""
    result: List[Dict] = []
    base = model_base(root)
    for spec in CATALOG["models"]:
        model_id = spec["id"]
        directory = base / spec["directory"]
        try:
            missing = missing_files(model_id, root)
            installed = not missing
            reason = "" if installed else "模型文件缺失或不完整"
        except (ValueError, OSError) as e:
            missing, installed, reason = {}, False, str(e)
        result.append(
            {
                "id": model_id,
                "name": spec["name"],
                "engine": spec["engine"],
                # 引擎注册名: 前端把「默认模型」同步到主界面引擎下拉时用
                # (与 /engines 的 name 一致; engine 字段是工厂类型, 对不上)
                "engine_name": spec.get("engine_name", ""),
                "description": spec.get("description", ""),
                "languages": spec.get("languages", []),
                "installed": installed,
                "hasFiles": directory.exists(),
                "missing": sorted(missing),
                "reason": reason,
                "sizeBytes": sum(spec["files"].values()),
                "directory": str(directory),
            }
        )
    if ALIGNER_SPEC:
        # 对齐器不是 ASR 引擎 (不进识别下拉), 但要出现在模型面板供下载/移除
        spec = ALIGNER_SPEC
        directory = base / spec["directory"]
        try:
            missing = aligner_missing_files(root)
            installed = not missing
            reason = "" if installed else "对齐器文件缺失或不完整"
        except (ValueError, OSError) as e:
            missing, installed, reason = {}, False, str(e)
        result.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "engine": "forced_aligner",
                # 对齐器不对应可选的识别引擎, engine_name 留空;
                # 前端据此不把它当 ASR 模型处理
                "engine_name": "",
                "description": spec.get("description", ""),
                "languages": [],
                "installed": installed,
                "hasFiles": directory.exists(),
                "missing": sorted(missing),
                "reason": reason,
                "sizeBytes": sum(spec["files"].values()),
                "directory": str(directory),
            }
        )
    return result


def aligner_spec() -> Dict:
    """返回对齐器声明; 未配置时报错而不是静默返回空。"""
    if not ALIGNER_SPEC:
        raise ValueError("清单未配置对齐器 (aligner)")
    return ALIGNER_SPEC


def aligner_directory(root: Optional[Path] = None) -> Path:
    """对齐器目录, 与模型目录同规则 (拒绝越界与符号链接)。"""
    spec = aligner_spec()
    base = model_base(root)
    path = base / spec["directory"]
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"对齐器目录越出模型根目录: {path}")
    return resolved


def aligner_missing_files(root: Optional[Path] = None) -> Dict[str, int]:
    """对齐器缺失文件 → 期望字节数; 空字典表示完整。"""
    spec = aligner_spec()
    base = model_base(root)
    directory = base / spec["directory"]
    missing: Dict[str, int] = {}
    for name, size in spec["files"].items():
        if not _check_file(directory / name, size, base):
            missing[name] = size
    return missing


def aligner_files(root: Optional[Path] = None) -> Dict[str, str]:
    """对齐器清单内相对路径 → 绝对路径; 不完整时抛 ValueError。"""
    spec = aligner_spec()
    directory = aligner_directory(root)
    missing = aligner_missing_files(root)
    if missing:
        raise ValueError(
            f"对齐器「{spec['name']}」未安装或不完整, 缺少: {', '.join(sorted(missing))}"
        )
    return {name: str(directory / name) for name in spec["files"]}

