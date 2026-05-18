"""
Standalone server launcher for Subtitle Assistant backend.
Usage: python run_server.py [--config config.yaml]
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import uvicorn
from app.core.config_loader import load_config

def setup_logging():
    # 尝试在当前目录（安装目录）创建日志
    log_dir = Path(os.getcwd()) / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "shiyu_error.log"
        # 测试写入权限
        with open(log_file, "a") as f:
            pass
    except PermissionError:
        # 如果安装在 C盘 Program Files 无权限，则回退到用户目录
        log_dir = Path.home() / ".shiyu" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "shiyu_error.log"

    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    os.environ["SHIYU_LOG_DIR"] = str(log_dir)
    return log_file


def main():
    log_file = setup_logging()
    
    parser = argparse.ArgumentParser(description="Subtitle Assistant Backend")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--host", help="Override host")
    parser.add_argument("--port", type=int, help="Override port")
    args = parser.parse_args()
    
    config = load_config(args.config)
    server_cfg = config["server"]
    
    host = args.host or server_cfg["host"]
    port = args.port or server_cfg["port"]
    
    print(f"=" * 50)
    print(f"时语 Shiyu Backend")
    print(f"Model: {config['model']['onnx_path']}")
    print(f"Listening: http://{host}:{port}")
    print(f"Log File: {log_file}")
    print(f"=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="error"
    )


if __name__ == "__main__":
    main()