<p align="center">
  <img src="frontend/assets/logo.png" alt="时语 Logo" width="120"/>
</p>

<h1 align="center">时语 Shiyu Subtitle</h1>

<p align="center">
  <strong>基于 Tauri + Vue 3 + SenseVoice-Small 的极速、轻量级 100% 离线本地 AI 字幕生成与精修助手。</strong>
</p>

<p align="center">
  <span>简体中文</span> | <a href="README.md">English</a>
</p>

---

### 界面预览

<p align="center">
  <img src="frontend/assets/shiyu.png" alt="时语字幕助手界面预览" width="100%" style="border-radius: 8px; box-shadow: 0 8px 30px rgba(0,0,0,0.5);"/>
</p>

---

时语字幕助手 (Shiyu Subtitle) 是一款为创作者、开发者与极客打造的极速、轻量级、100% 离线本地 AI 字幕助手。它将阿里巴巴开源的 SenseVoice-Small 高性能语音识别能力，与现代高效的 Tauri + Vue 3 桌面端架构完美融合，旨在为您提供最纯粹、私密且极其流畅的本地字幕生成与精修体验。

### 核心特性

* **100% 本地私密安全**：无需云端 API 密钥，无需联网。您的音视频数据与文本绝不上云，彻底杜绝隐私泄露。
* **SenseVoice-Small 驱动**：基于深度优化的 ONNX Runtime 进行本地推理。1 小时的音频可在 1 分钟内极速识别完毕。
* **智能断句与排版**：内置中英文双语高精度智能断句算法，彻底解决语音模型原生的长句粘连问题，排版更符合人类阅读习惯。
* **CTC 峰值时值补偿**：自动应用精准的 -150ms 时间戳偏移补偿，完美对齐声音波形，消除语音模型特有的 CTC 峰值延迟。
* **现代极简美学 UI**：精心打磨的深色磨砂玻璃微动效界面，配合波形同步预览、点击时间码快捷跳转与顺滑的快捷键精修体验。
* **静默后台守护管理**：后端打包为 PyInstaller 单文件可执行程序，无需安装 Python 运行时，没有烦人的终端窗口闪现。
* **CI/CD 一键发布**：预置 GitHub Actions 自动构建脚本，推送 Tag 即自动编译生成 Windows (.exe)、macOS (.dmg) 和 Linux (.deb) 安装包。

---

### 技术架构

时语采用高性能的前后端分离桌面架构：
```mermaid
graph TD
    A[Tauri GUI - Vue 3 / Vite] <-->|本地 Localhost API| B[后端: PyInstaller 单文件]
    B -->|ONNX Runtime 推理| C[SenseVoice-Small 模型 ~/.shiyu/models/]
    A -->|Tauri 命令| D[Rust 进程控制器]
    D -->|拉起 / 销毁| B
```

* **前端 (GUI)**：Vue 3, Vite, Naive UI, Lucide Icons (交互与音视频时间轴控制)
* **Tauri 核心**：Rust (系统窗口管理、静默进程守护、进程生命周期控制)
* **后端 (推理引擎)**：FastAPI, Uvicorn, PyInstaller 单文件可执行程序 (ONNX Runtime + SenseVoice 推理)
* **模型存储**：首次启动自动下载至 `~/.shiyu/models/sensevoice-small/`

---

### 快速开始 (本地开发)

#### 环境准备
* **Node.js** (v18+) 及 **npm** 包管理工具
* **Rust** 编译器 (Stable toolchain)
* **Python** 环境 (推荐 3.10.x)

#### 1. 初始化 Python 后端
1. 进入后端目录：
   ```bash
   cd backend
   ```
2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   # Windows 系统:
   .\venv\Scripts\activate
   # macOS / Linux 系统:
   source venv/bin/activate
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   > SenseVoice-Small 模型（约 230MB）会在首次启动时**自动下载**，无需手动操作。
   > 模型存放位置：`~/.shiyu/models/sensevoice-small/`

#### 2. 初始化前端并启动开发服务
1. 进入前端目录：
   ```bash
   cd ../frontend
   ```
2. 安装依赖：
   ```bash
   npm install
   ```
3. 启动开发版软件：
   ```bash
   npm run tauri dev
   ```

---

### 打包与发布

GitHub Actions 工作流 (`.github/workflows/release.yml`) 可自动构建 Windows (.exe)、macOS (.dmg) 和 Linux (.deb) 安装包。

触发发布流程：
1. 将代码推送到 GitHub。
2. 创建并推送版本 Tag：
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```
3. Runner 将编译 Rust (Tauri) 代码、将 Python 后端打包为 PyInstaller 单文件可执行程序、整合为原生安装包，并发布为草稿 Release。
4. 模型在用户首次启动时**自动下载**，安装包不包含模型文件。

> 💡 如需本地构建后端可执行程序进行生产测试：
> ```bash
> cd backend
> pip install pyinstaller
> python build.py
> # 输出: dist/shiyu-backend.exe
> ```

---

### 开源协议

本项目基于 [MIT 开源协议](LICENSE) 授权。
