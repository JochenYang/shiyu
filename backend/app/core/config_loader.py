"""
Configuration loader for subtitle assistant.

模型路径不再写死在 config.yaml 里 —— 由 ``backend/config/models.json`` 清单
管理, 运行时经 ``app.core.model_catalog`` 解析。这里只负责:
  * 展开 ``model.dir`` 的 ``~`` 并转成绝对路径
  * 在缺少可选段时补上默认值, 让调用方不必到处 ``.get`` 判空
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "model": {"dir": "~/.shiyu/models"},
    "audio": {"sample_rate": 16000, "max_duration": 300},
    "inference": {"language": "auto", "use_itn": True, "device": "cpu"},
    "output": {"default_format": "srt"},
    "engines": {},
}


def _merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """把缺失的段补上默认值 (只补顶层段, 不覆盖用户写的值)。"""
    for section, values in DEFAULTS.items():
        if section not in config or config[section] is None:
            config[section] = dict(values) if isinstance(values, dict) else values
        elif isinstance(values, dict):
            for key, value in values.items():
                config[section].setdefault(key, value)
    return config


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config and resolve ``model.dir`` to an absolute path.

    Args:
        config_path: Path to config.yaml (relative paths resolve against backend/).

    Returns:
        config dict with ``model.dir`` expanded to an absolute path.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        backend_dir = Path(__file__).parent.parent.parent
        config_file = backend_dir / config_path

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config = _merge_defaults(config)

    model_dir_raw = config["model"].get("dir", "~/.shiyu/models")
    config["model"]["dir"] = str(Path(os.path.expanduser(model_dir_raw)).resolve())

    return config
