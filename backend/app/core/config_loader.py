"""
Configuration loader for subtitle assistant.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config, resolve relative paths to absolute.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        config dict with absolute paths
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        # Try relative to backend dir
        backend_dir = Path(__file__).parent.parent.parent
        config_file = backend_dir / config_path
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Resolve model paths to absolute
    base_dir = config_file.parent
    for key in ["onnx_path", "tokens_path", "config_path"]:
        path = config["model"].get(key)
        if path and not Path(path).is_absolute():
            resolved = (base_dir / path).resolve()
            config["model"][key] = str(resolved)
    
    # Resolve audio cmvn_file path to absolute
    cmvn_path = config.get("audio", {}).get("cmvn_file")
    if cmvn_path and not Path(cmvn_path).is_absolute():
        resolved = (base_dir / cmvn_path).resolve()
        config["audio"]["cmvn_file"] = str(resolved)
    
    return config