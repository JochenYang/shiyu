"""
PyInstaller build script for Shiyu backend.
Produces a portable single executable: shiyu-backend (.exe on Windows).

Usage:
    pip install pyinstaller
    python build.py

Output: dist/shiyu-backend (or dist/shiyu-backend.exe on Windows)
"""
import platform
import PyInstaller.__main__


def main():
    system = platform.system()
    is_windows = system == "Windows"

    args = [
        "run_server.py",
        "--name=shiyu-backend",
        "--onefile",                     # Single executable output
        "--add-data=config.yaml:.",       # Embed config.yaml in the exe
        "--add-data=config/models.json:config",  # 模型清单 (固定 revision + SHA-256)
        "--collect-data=jieba",            # jieba 词典 (断句依赖, 跨平台由 PyInstaller 定位)
        "--clean",
        "--noconfirm",
    ]

    if is_windows:
        args.append("--noconsole")  # No cmd window on Windows

    # --- Hidden imports ---
    # PyInstaller may miss these because they are loaded dynamically
    hidden = [
        # App modules (uvicorn.run("app.main:app") is string-based, not auto-detected)
        "app.main",
        "app.core.model_catalog",
        "app.core.model_download",
        "app.core.sherpa_engine",
        "app.core.caption_builder",
        "app.core.semantic_breaks",
        "app.core.text_units",
        "app.core.audio_processor",
        "app.core.subtitle_formatter",
        "app.core.config_loader",
        # v2.4 起新增 (在函数内部导入, 或由字符串路由间接引用)
        "app.core.engine_base",
        "app.core.pipeline",
        "app.core.cloud_engine",
        "app.core.doubao_cloud",
        "app.core.forced_aligner",
        "app.core.editor_export",
        "app.core.itn",
        "app.core.punctuation",
        "app.core.asr_corrector",
        "app.core.bbpe_map",
        "app.core.text_splitter",
        "app.core.ffmpeg_downloader",
        # 断句依赖 jieba (纯 Python, 字典是数据文件)
        "jieba",
        # Audio processing (fbank/LFR/CMVN 由 sherpa-onnx 内部完成, 不再需要 kaldi-native-fbank)
        "soundfile",                # C extension, needs libsndfile
        "requests",                 # 云端引擎 HTTP 调用
        "onnxruntime",              # 强制对齐器推理
        "tokenizers",               # 强制对齐器分词 (Rust 扩展)
        "librosa",                  # mel 特征 / 重采样 / 时长
        # Web framework
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",
        "multipart",                # For FastAPI file uploads
        # Serialization
        "yaml",
        "pydantic",
        # Subtitle
        "pysubs2",
        # zhconv: Traditional → Simplified Chinese (dynamic import in engines)
        "zhconv",
    ]

    for mod in hidden:
        args.append(f"--hidden-import={mod}")

    # --- Excludes ---
    # Reduce bundle size by dropping packages not needed at runtime
    excludes = [
        "matplotlib",
        "tkinter",
        "PIL",              # Pillow (image processing, not needed here)
        # NOTE: setuptools is intentionally NOT excluded because PyInstaller's
        # built-in runtime hook (pyi_rth_pkgres) depends on pkg_resources, which
        # in turn requires setuptools' vendored modules (jaraco, etc.).
        # Removing it causes startup crash: "No module named 'jaraco'".
        "pip",
        # NOTE: wheel 也不能排除。PyInstaller 6.x 的 setuptools hook 会把
        # setuptools 内嵌的 wheel 别名成顶层 wheel, 排除它会让构建直接报
        # "Target module wheel already imported as ExcludedModule('wheel')"。
    ]
    for mod in excludes:
        args.append(f"--exclude-module={mod}")

    print(f"[build.py] Starting PyInstaller build for {system}...")
    print(f"[build.py] Args: {' '.join(args)}")
    PyInstaller.__main__.run(args)
    print(f"[build.py] Build complete! Check dist/ directory.")


if __name__ == "__main__":
    main()
