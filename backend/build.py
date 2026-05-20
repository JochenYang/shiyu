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
        "app.core.audio_processor",
        "app.core.subtitle_formatter",
        "app.core.model_downloader",
        "app.core.config_loader",
        "app.core.sensevoice_engine",
        # Audio processing
        "kaldi_native_fbank",       # C++ extension, loaded via import
        "soundfile",                # C extension, needs libsndfile
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
        "wheel",
    ]
    for mod in excludes:
        args.append(f"--exclude-module={mod}")

    print(f"[build.py] Starting PyInstaller build for {system}...")
    print(f"[build.py] Args: {' '.join(args)}")
    PyInstaller.__main__.run(args)
    print(f"[build.py] Build complete! Check dist/ directory.")


if __name__ == "__main__":
    main()
