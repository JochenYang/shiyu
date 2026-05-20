"""
Configuration loader for subtitle assistant.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config, resolve model paths relative to model.dir.

    Args:
        config_path: Path to config.yaml

    Returns:
        config dict with absolute paths stored under legacy keys
        (onnx_path, tokens_path, config_path) for backward compat
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        backend_dir = Path(__file__).parent.parent.parent
        config_file = backend_dir / config_path

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve model.dir: expand ~ and convert to absolute
    model_dir_raw = config["model"].get("dir", "~/.shiyu/models")
    model_dir = Path(os.path.expanduser(model_dir_raw)).resolve()
    config["model"]["dir"] = str(model_dir)

    # Resolve model paths relative to model.dir, stored under legacy keys
    path_keys = {
        "onnx": "onnx_path",
        "tokens": "tokens_path",
        "model_config": "config_path",
    }
    for yaml_key, config_key in path_keys.items():
        rel_path = config["model"].get(yaml_key)
        if rel_path:
            absolute = (model_dir / rel_path).resolve()
            config["model"][config_key] = str(absolute)

    # Resolve audio cmvn_file: extract basename and locate in model.dir
    cmvn_path = config.get("audio", {}).get("cmvn_file")
    if cmvn_path and not Path(cmvn_path).is_absolute():
        resolved = (model_dir / "sensevoice-small" / Path(cmvn_path).name).resolve()
        config["audio"]["cmvn_file"] = str(resolved)

    return config
