<p align="center">
  <img src="frontend/assets/logo.png" alt="Shiyu Logo" width="120"/>
</p>

<h1 align="center">Shiyu Subtitle</h1>

<p align="center">
  <strong>An ultra-fast, lightweight, and 100% offline local AI subtitle assistant powered by Tauri, Vue 3, and SenseVoice-Small.</strong>
</p>

<p align="center">
  <a href="README.zh-CN.md">简体中文</a> | <span>English</span>
</p>

---

### Application Preview

<p align="center">
  <img src="frontend/assets/shiyu.png" alt="Shiyu Subtitle Preview" width="100%" style="border-radius: 8px; box-shadow: 0 8px 30px rgba(0,0,0,0.5);"/>
</p>

---

Shiyu Subtitle is an ultra-fast, lightweight, and 100% offline local AI subtitle assistant designed for creators, developers, and power users. By combining the high-performance speech recognition of SenseVoice-Small with the modern, efficient Tauri + Vue 3 desktop framework, Shiyu delivers a premium, distraction-free subtitle generation and editing experience directly on your local machine.

### Key Features

* **100% Local & Private**: No cloud API keys, no network required. Your audio, video, and transcripts never leave your machine.
* **SenseVoice-Small Powered**: Ultra-fast voice-to-text inference utilizing optimized ONNX Runtime. Compresses 1 hour of audio transcription into less than 1 minute.
* **Smart Speech Segmentation**: Advanced bilingual segmenting algorithms for natural, human-readable line breaks, bypassing model-native limitations.
* **Temporal Offset Compensation**: Integrated -150ms latency compensation, perfectly aligning subtitles to the audio waveform and correcting SenseVoice CTC peak delays.
* **Modern & Sleek UI**: A premium dark-mode glassmorphism interface featuring smooth micro-animations, synchronized wave previews, and seamless timing navigation.
* **Silent Daemon Management**: Seamless, headless backend process lifecycle control. The backend runs as a PyInstaller onefile executable—no Python runtime required, no annoying terminal windows.
* **CI/CD Ready**: Ready-to-go GitHub Actions configuration for automated multi-platform compilation (Windows, macOS, and Linux). One push of a tag triggers the full build pipeline with zero manual steps.

---

### System Architecture

```mermaid
graph TD
    A[Tauri GUI - Vue 3 / Vite] <-->|Local Localhost API| B[Backend: PyInstaller Onefile]
    B -->|ONNX Runtime Inference| C[SenseVoice-Small Model ~/.shiyu/models/]
    A -->|Tauri Command| D[Rust Process Controller]
    D -->|Spawns / Terminates| B
```

* **Frontend**: Vue 3, Vite, Naive UI, Lucide Icons, Web Audio API
* **Tauri Core**: Rust (system windowing, silent process daemonization, process lifecycle)
* **Backend**: FastAPI, Uvicorn, PyInstaller onefile executable (ONNX Runtime + SenseVoice inference)
* **Model Storage**: Auto-downloaded to `~/.shiyu/models/sensevoice-small/` on first launch

---

### Quick Start (Local Development)

#### Prerequisites
* **Node.js** (v18+) & **npm**
* **Rust** (stable toolchain)
* **Python** (3.10.x recommended)

#### 1. Setup Backend
1. Clone the repository and navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > The SenseVoice-Small model (~230 MB) is **auto-downloaded** on first app launch. No manual download needed.
   > Model location: `~/.shiyu/models/sensevoice-small/`

#### 2. Setup Frontend & Run Dev
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Run the application in development mode:
   ```bash
   npm run tauri dev
   ```

---

### Packaging and Releasing

A GitHub Actions workflow (`.github/workflows/release.yml`) automates building installers for Windows (.exe), macOS (.dmg), and Linux (.deb).

To trigger a release:
1. Push your code to GitHub.
2. Create and push a version tag:
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```
3. The runner will compile Rust (Tauri), build the Python backend into a PyInstaller onefile executable, bundle everything into native installers, and publish them as draft releases.
4. The model is **auto-downloaded** at runtime on first launch — no model files are included in the installer.

> 💡 To build the backend executable locally for production testing:
> ```bash
> cd backend
> pip install pyinstaller
> python build.py
> # Output: dist/shiyu-backend.exe
> ```

---

### License

This project is licensed under the [MIT License](LICENSE).
