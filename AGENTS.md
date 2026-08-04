# 时语 Shiyu Subtitle — AGENTS.md

> 面向 AI 编码代理的项目知识库。本文档描述项目结构、技术栈、构建流程与开发约定。

---

## 项目概述

**时语 Shiyu Subtitle** 是一款基于 Tauri + Vue 3 + SenseVoice-Small 的极速、轻量级、100% 离线本地 AI 字幕生成与精修桌面应用。项目定位为创作者、开发者和极客提供纯粹的本地字幕体验。

### 核心特性

- **100% 本地离线**：无需云端 API，音视频数据不上传
- **多 ASR 引擎**：Zipformer-CTC-zh (主走)、Paraformer-large (CIF 边界更准, 可选)、SenseVoice-Small (多语言备选)、Cloud (OpenAI 兼容 API)
- **智能断句**：三级分段策略（句子→逗号→硬切），自研智能验算词典
- **CTC 峰值补偿**：-150ms 时间戳偏移补偿，对齐声音波形
- **现代 UI**：深色磨砂玻璃微动效界面，波形同步预览
- **静默后台守护**：后端打包为 PyInstaller 单文件，无需 Python 运行时
- **CI/CD 一键发布**：GitHub Actions 自动构建 Windows/macOS 安装包

### 技术架构

```
Tauri GUI (Vue 3 / Vite / Naive UI)
    ↕ 本地 Localhost HTTP API (FastAPI)
Rust 进程控制器 (Tauri Core)
    ↕ 拉起/销毁
PyInstaller 单文件后端 (FastAPI + ONNX Runtime)
    ↕ ONNX 推理
SenseVoice-Small / Zipformer-CTC-zh 模型 (~/.shiyu/models/)
```

---

## 目录结构

```
shiyu/
├── backend/                    # Python 后端 (FastAPI + ONNX)
│   ├── app/
│   │   ├── main.py             # FastAPI 应用入口, REST API 定义
│   │   └── core/               # 后端核心模块
│   │       ├── engine_base.py          # 引擎统一抽象基类
│   │       ├── engine_registry.py      # 引擎注册中心
│   │       ├── sensevoice_engine.py    # SenseVoice ONNX 推理引擎
│   │       ├── zipformer_engine.py     # Zipformer-CTC-zh (sherpa-onnx)
│   │       ├── paraformer_engine.py    # Paraformer-large (sherpa-onnx)
│   │       ├── cloud_engine.py         # 云端 OpenAI/DashScope 引擎
│   │       ├── pipeline.py             # 字幕生成流水线编排器 (4 阶段)
│   │       ├── audio_processor.py      # 音频预处理 (fbank/LFR/CMVN)
│   │       ├── model_downloader.py     # 模型自动下载
│   │       ├── ffmpeg_downloader.py    # FFmpeg 自动下载
│   │       ├── vad_detector.py         # Silero-VAD 语音活动检测
│   │       ├── asr_corrector.py        # 确定性 ASR 修正 (智能验算词典)
│   │       ├── punctuation.py          # 标点恢复 (Rule + NN stub)
│   │       ├── subtitle_formatter.py   # SRT/LRC/ASS 格式化
│   │       ├── subtitle_validator.py   # 字幕校验器
│   │       ├── text_splitter.py        # 文本切分工具
│   │       ├── itn.py                  # 逆文本归一化 (中文数字→阿拉伯)
│   │       ├── post_process.py         # 统一后处理管线
│   │       └── config_loader.py        # YAML 配置加载
│   ├── run_server.py           # 后端启动入口 (uvicorn)
│   ├── build.py                # PyInstaller 打包脚本
│   ├── config.yaml             # 后端配置文件
│   ├── requirements.txt        # Python 依赖
│   └── tests/                  # 测试目录
│       └── core/
│           ├── test_pipeline_v2.py
│           ├── test_zipformer_engine.py
│           ├── test_audio_extraction.py
│           ├── test_cloud_engine_config_routing.py
│           ├── test_cloud_engine_errors.py
│           ├── test_zipformer_e2e_real.py
│           └── test_zipformer_engine.py
├── frontend/                   # Vue 3 前端
│   ├── src/
│   │   ├── main.js             # Vue 入口
│   │   ├── App.vue             # 根组件 (Naive UI 主题, 全局样式)
│   │   ├── api.js              # 后端 API 客户端
│   │   └── components/
│   │       ├── AppContent.vue  # 主布局 (上传/识别/编辑/设置)
│   │       ├── SubtitlePreview.vue  # 字幕预览组件
│   │       └── WavePlayer.vue  # 波形播放器 (wavesurfer.js)
│   ├── src-tauri/              # Tauri Rust 核心
│   │   ├── src/main.rs         # 后端进程管理 (启动/销毁)
│   │   ├── Cargo.toml          # Rust 依赖
│   │   └── tauri.conf.json     # Tauri 配置
│   ├── package.json
│   └── vite.config.js
├── models/                     # 模型文件 (开发环境)
│   └── sensevoice-small/
│       ├── config.yaml
│       └── tokens.json
├── .github/workflows/
│   └── release.yml             # CI/CD 发布工作流
└── AGENTS.md                   # 本文档
```

---

## 技术栈详情

### 前端 (Vue 3 + Tauri)

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 框架 | Vue 3 (Composition API) | 响应式 UI |
| 构建 | Vite 5 | 开发/生产构建 |
| UI 库 | Naive UI 2.44 | 组件库 (深色主题) |
| 图标 | Lucide Vue Next | 图标集 |
| 波形 | wavesurfer.js 7 | 音频波形可视化 |
| 工具 | @vueuse/core 14 | 组合式工具函数 |
| 桌面壳 | Tauri 1.5 (Rust) | 系统窗口、进程管理 |
| API | @tauri-apps/api 1.5 | Tauri 前端 API |

### 后端 (Python FastAPI)

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| Web 框架 | FastAPI 0.104+ | REST API 服务 |
| ASGI 服务器 | Uvicorn 0.24+ | HTTP 服务 |
| 推理引擎 | onnxruntime 1.16+ | ONNX 模型推理 |
| 音频处理 | librosa 0.10+ | 音频加载 |
| 音频特征 | kaldi-native-fbank 1.1 | fbank 特征提取 |
| 语音识别 | sherpa-onnx 1.10+ | Zipformer/Paraformer 引擎 |
| 字幕格式 | pysubs2 1.6+ | SRT/LRC/ASS 生成 |
| 配置 | PyYAML 6.0+ | YAML 配置解析 |
| 打包 | PyInstaller | 单文件可执行程序 |

### Rust (Tauri)

| 组件 | 版本 | 用途 |
|------|------|------|
| tauri | 1.5.0 | 窗口管理、进程控制 |
| tauri-build | 1.5.0 | 构建工具 |
| serde | 1.0 | 序列化 |
| serde_json | 1.0 | JSON 处理 |

---

## ASR 引擎体系

项目采用基于 `BaseTranscriber` 抽象基类的多引擎体系，所有引擎实现统一的 `transcribe(audio_path, language)` 接口。

### 引擎列表

| 引擎标识符 | 类名 | 类型 | 语言 | 说明 |
|-----------|------|------|------|------|
| `zipformer_local` | `ZipformerEngine` | 本地 | 中文 | 主走, sherpa-onnx, 350MB int8 |
| `sensevoice_local` | `SenseVoiceEngine` | 本地 | zh/en/ja/ko/yue | 多语言备选, ONNX Runtime, 230MB |
| `paraformer_local` | `ParaformerEngine` | 本地 | 中文 | 备选, CIF 边界 ±20ms, 232MB |
| `cloud` | `CloudTranscriber` | 云端 | 多语言 | OpenAI/DashScope 兼容 API |

### 流水线 (v2.0)

字幕生成分为 4 个阶段：

1. **ASR** — 主引擎转录 (Zipformer/SenseVoice/Paraformer)
2. **CORRECT** — 确定性修正 + ITN + 标点恢复
3. **VALIDATE** — 字幕校验 (轻量日志)
4. **FORMAT** — SRT/LRC/ASS 格式化输出

流水线通过 `PipelineContext` 传递中间结果，每个阶段独立可测，异常降级。

---

## 配置体系

### 后端配置 (`backend/config.yaml`)

```yaml
model:
  dir: "~/.shiyu/models"
  onnx: "sensevoice-small/model.onnx"
  tokens: "sensevoice-small/tokens.json"

audio:
  sample_rate: 16000
  n_mels: 80
  lfr_m: 7        # Low Frame Rate 参数
  lfr_n: 6
  max_duration: 300  # 秒, 超长则切分

inference:
  language: "auto"
  use_itn: true
  device: "cpu"

server:
  host: "127.0.0.1"
  port: 11235
  cors_origins: ["http://localhost:1420", "tauri://localhost"]

engines:
  paraformer:
    enabled: true
    model_dir: "~/.shiyu/models/paraformer-large"
    num_threads: 4
  zipformer:
    enabled: true
    model_dir: "~/.shiyu/models/zipformer-ctc-zh"
  sensevoice:
    enabled: true
    model_dir: "~/.shiyu/models/sensevoice-small"
```

### 云端配置 (`~/.shiyu/config.json`)

云引擎和 LLM 修正的 API 密钥等凭据存储在用户目录下的 `~/.shiyu/config.json`，通过 `cloud_engine.py` 中的 `load_engine_config()` / `save_engine_config()` 读写。

---

## 构建与运行

### 开发环境

```bash
# 后端
cd backend
python -m venv venv
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
python run_server.py       # 启动后端 (默认 http://127.0.0.1:11235)

# 前端 (另一个终端)
cd frontend
npm install
npm run tauri dev          # 启动 Tauri 开发模式
```

### 生产构建

```bash
# 后端 PyInstaller 打包
cd backend
pip install pyinstaller
python build.py
# 输出: dist/shiyu-backend.exe (Windows)

# Tauri 前端 + 后端整合构建
cd frontend
npm run tauri build
```

### 测试

```bash
cd backend
.\venv\Scripts\activate
pytest                          # 运行所有单元测试
pytest -m integration           # 仅运行集成测试 (需要真实模型文件)
pytest -v tests/core/           # 详细输出
```

### CI/CD

GitHub Actions 工作流位于 `.github/workflows/release.yml`：
- 推送 tag `v*` 时触发
- 构建 Windows (.exe, NSIS) 和 macOS (.dmg) 安装包
- 自动将 PyInstaller 后端打包进 Tauri 资源
- 生成本次发布以来的 commit changelog
- 发布为 Draft Release

---

## 测试策略

### 测试框架

- **pytest**: 后端测试框架
- 测试文件位于 `backend/tests/core/`
- 集成测试用 `@pytest.mark.integration` 标记，默认不执行

### 测试覆盖

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_pipeline_v2.py` | 4 阶段流水线 (ASR/CORRECT/VALIDATE/FORMAT) |
| `test_zipformer_engine.py` | 引擎导入安全、is_available 路径、配置解析、结果转换、E2E |
| `test_audio_extraction.py` | ffmpeg 错误处理、输出验证 |
| `test_cloud_engine_*.py` | 云端引擎配置路由、错误处理 |

### 测试约定

- 单元测试使用 mock 引擎，不依赖真实模型文件
- 引擎测试构造 `_make_mock_engine()` 返回模拟 `TranscribeResult`
- `is_available()` 的失败路径全覆盖（缺模块、缺目录、缺文件、文件太小）
- 集成测试用 `pytest -m integration` 单独运行，需要真实模型下载完毕

---

## 代码风格约定

### 注释语言

- 代码注释、文档字符串使用**中文**（项目主要贡献者和用户为中文社区）
- 关键设计决策和版本演进标注在注释中（如 `v2.0 简化: 砍 FUSE 阶段`）
- 英文技术术语保持原文（如 ASR, CTC, ITN, VAD）

### 命名规范

- Python: `snake_case` 函数/变量，`PascalCase` 类名
- Vue: `PascalCase` 组件名，`camelCase` 属性和方法
- Rust: `snake_case` 函数/变量，`PascalCase` 类型
- 引擎标识符使用 `snake_case` 加 `_local`/`_cloud` 后缀

### 架构原则

- **最小正确**：不改无关文件、不顺手重构、不增加当前不需要的抽象
- **不变性**：流水线每阶段不修改输入 segments，返回新对象或浅拷贝
- **降级**：任何阶段异常不中断流水线，降级到上一个稳定状态
- **引擎可插拔**：所有引擎实现 `BaseTranscriber` 接口，`main.py` 中通过 `engine_registry` 统一注册

---

## 数据与模型

### 模型存储位置

所有模型文件自动下载至 `~/.shiyu/models/`：

| 模型 | 路径 | 大小 | 来源 |
|------|------|------|------|
| SenseVoice-Small | `~/.shiyu/models/sensevoice-small/` | ~230MB | ModelScope / HF Mirror |
| Zipformer-CTC-zh | `~/.shiyu/models/zipformer-ctc-zh/` | ~350MB | sherpa-onnx 模型 |
| Paraformer-large | `~/.shiyu/models/paraformer-large/` | ~232MB | sherpa-onnx 模型 |
| Silero-VAD | `~/.shiyu/models/silero-vad/` | ~20MB | PyPI / HF Mirror |
| 标点模型 | `~/.shiyu/models/punc_ct-transformer/` | ~50MB | ModelScope (stub) |

### 模型下载策略

- 首次启动自动下载（后台线程，非阻塞）
- 多源降级：ModelScope → HF Mirror → GitHub Release
- 下载进度通过 API `/model/status` 返回给前端
- 从 Tauri 资源包中迁移（生产环境安装包内嵌）

---

## 安全注意事项

### 凭据管理

- Cloud API key 和 LLM API key 存储在 `~/.shiyu/config.json`（明文）
- 文件权限设为 `0o600`（Unix）以限制访问
- API 路由 `/settings/engine` 和 `/settings/llm-correction` 接受明文密钥
- 前端密码输入框默认掩码显示

### 数据安全

- 所有数据处理在本地完成，不上传至任何外部服务
- 云引擎模式需要用户显式配置 API endpoint 和 key
- 临时文件在上传转录后自动清理（`shutil.rmtree`）

### 进程安全

- Rust 层通过 `taskkill /T` 确保整个进程树被清理
- 窗口关闭和程序退出时双重保证后端进程销毁
- 子进程创建时使用 `CREATE_NO_WINDOW` 标志避免控制台窗口闪现

---

## 版本演进记录

| 版本 | 变更 |
|------|------|
| v1.0 | 初始版本, SenseVoice 单引擎 |
| v2.0 | 重构为多引擎体系, Zipformer 主走, 砍 FUSE/CLAUSE/LLM 阶段, 4 阶段流水线 |
| v2.1 | Paraformer-CIF 接入 (CIF 边界精度更高), 引擎配置化；主走引擎经回归评估仍为 Zipformer-CTC |

### 废弃功能

- Bcut (必剪) 和 Jianying (剪映) 免费 ASR 引擎 (2026 年 5 月移除，API 不可达)
- LLM 修正层 (owner 决策，太慢)
- 基于 VAD 的单独断句阶段 (sherpa-onnx 自带 CTC 分段 + Silero-VAD 冗余)