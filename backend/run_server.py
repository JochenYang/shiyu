"""
Standalone server launcher for Subtitle Assistant backend.
Usage: python run_server.py [--config config.yaml]
"""
import os
import sys
import logging
from pathlib import Path
import argparse
import builtins
import uvicorn
from app.core.config_loader import load_config

# Log level mapping
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Save the original print before any replacement
_original_print = builtins.print


def _teed_print(*args, **kwargs):
    """Print replacement that also writes to the startup log file."""
    _original_print(*args, **kwargs)
    msg = " ".join(str(a) for a in args)
    logging.getLogger("startup").info(msg)


def setup_logging(log_level_name: str = "ERROR"):
    """Set up logging to ~/.shiyu/logs/shiyu_error.log.

    Args:
        log_level_name: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
            Defaults to ERROR for quiet production operation.
    """
    shiyu_dir = Path.home() / ".shiyu"
    log_dir = shiyu_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "shiyu_error.log"

    level = _LOG_LEVELS.get(log_level_name.upper(), logging.ERROR)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    os.environ["SHIYU_LOG_DIR"] = str(log_dir)

    # Route print() calls to log file via module-level teed_print
    builtins.print = _teed_print

    return log_file


def main():
    parser = argparse.ArgumentParser(description="Subtitle Assistant Backend")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--host", help="Override host")
    parser.add_argument("--port", type=int, help="Override port")
    parser.add_argument(
        "--log-level", default=None,
        help="Log level: DEBUG/INFO/WARNING/ERROR/CRITICAL (default: ERROR)"
    )
    args = parser.parse_args()

    # Priority: CLI arg > env var > default
    log_level_name = args.log_level or os.environ.get("SHIYU_LOG_LEVEL", "ERROR")
    log_file = setup_logging(log_level_name)

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
