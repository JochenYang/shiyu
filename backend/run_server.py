"""
Standalone server launcher for Subtitle Assistant backend.
Usage: python run_server.py [--config config.yaml]
"""
import argparse
import uvicorn
from app.core.config_loader import load_config


def main():
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
    print(f"=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()