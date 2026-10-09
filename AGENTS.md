# 时语 Shiyu Subtitle — AGENTS.md

> 面向 AI 编码代理的项目知识库。本文档描述项目结构、技术栈、构建流程与开发约定。

---

## 项目概述

**时语 Shiyu Subtitle** 是一款基于 Tauri + Vue 3 + sherpa-onnx 的极速、轻量级、100% 离线本地 AI 字幕生成与精修桌面应用。项目定位为创作者、开发者和极客提供纯粹的本地字幕体验。

### 核心特性

- **100% 本地离线**：无需云端 API，音视频数据不上传
- **清单式模型管理**：模型由 `backend/config/models.json` 声明（固定 revision + 逐文件 SHA-256），支持下载/续传/校验/移除
- **多 ASR 引擎**：Zipformer-CTC-zh（主走中文，逐 token 时间戳最细）、SenseVoice-Small（中英日韩粤，轻量）、FireRedASR2-CTC（20+ 方言）、Cloud（OpenAI 兼容 API）
- **测量驱动断句**：在引擎逐 token 时间戳还原的**已测量词边界**上做最小代价 DP 切分，不用字符数比例插值
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
PyInstaller 单文件后端 (FastAPI + sherpa-onnx)
    ↕ ONNX 推理
Zipformer / SenseVoice / FireRedASR2 模型 (~/.shiyu/models/)
```

---

## 模型管理（清单式）

模型**不在代码里硬编码**，全部由 `backend/config/models.json` 声明：每个模型带 `repository` + 固定 `revision`（40 位 commit，不用 `main`）、每个文件的精确字节数与 SHA-256。

- `app/core/model_catalog.py` — 清单查询、路径安全（拒绝符号链接与越界）、可用性判断（只比大小，快）。**可用性查询永远不触发下载。**
- `app/core/model_download.py` — `.part` 暂存 → HTTP Range 续传 → SHA-256 校验 → 原子替换；下载前查磁盘余量；多镜像共用同一份校验值；`install` 与 `remove` 对称。
- `app/core/sherpa_engine.py` — 所有本地模型共用一条推理路径，区别只在 sherpa-onnx 工厂方法、工厂参数组装与 token→字符时间的映射方式；多文件模型（Qwen3-ASR 的 conv_frontend+encoder+decoder+tokenizer）由 `_factory_args` 按文件名特征映射，不在代码里写死路径。

新增模型只改清单，不动代码。API：`GET /models`、`POST /models/{id}/install`、`POST /models/{id}/remove`、`POST /models/jobs/{id}/cancel`。

---

## 字幕断句

断句在**已测量的词边界**上做最小代价 DP，而不是按字符数比例切分：

- `app/core/text_units.py` — `protected_ranges`（`0.1`/`1/30`/`5GHz`/`C++` 等数值与标识符不被标点清理破坏）、`display_length`（ASCII 记半宽）、`clean_generated_text`
- `app/core/semantic_breaks.py` — `phrase_partition`（DP 代价：段尾虚词加罚、段首虚词加罚、真实停顿减罚、句读减罚、引导词减罚）、`coordinated_ranges`（并列结构整体保留）
- `app/core/caption_builder.py` — 把逐字符时间戳经 jieba 词边界聚成测量词，再交给 DP 切分；**时间戳全部来自测量值，长度不匹配时直接报错而不是猜测**

时间信号来源：Zipformer 的 byte-level BPE token 经 `bbpe_map` 还原到逐字符；SenseVoice / FireRedASR2-CTC 的 token 本身即字符。

---


## 目录结构

```
shiyu/
├── backend/                    # Python 后端 (FastAPI + ONNX)
│   ├── app/
│   │   ├── main.py             # FastAPI 应用入口, REST API 定义
│   │   └── core/               # 后端核心模块
│   │       ├── engine_base.py          # 引擎统一抽象基类
│   │       ├── cloud_engine.py         # 云端 OpenAI/DashScope 引擎
│   │       ├── pipeline.py             # 字幕生成流水线编排器
│   │       ├── audio_processor.py      # 音频抽取 (ffmpeg)
│   │       ├── model_catalog.py        # 模型清单 (pinned revision + SHA-256)
│   │       ├── model_download.py       # 校验下载 / 续传 / 原子发布 / 安装移除
│   │       ├── sherpa_engine.py        # sherpa-onnx 统一推理路径 (所有本地模型)
│   │       ├── caption_builder.py      # 逐字符时间 → 测量词 → 字幕行
│   │       ├── semantic_breaks.py      # 最小代价 DP 断句
│   │       ├── text_units.py           # 功能性写法保护 + 显示宽度
│   │       ├── ffmpeg_downloader.py    # FFmpeg 自动下载
│   │       ├── asr_corrector.py        # 确定性 ASR 修正 (验算词典)
│   │       ├── punctuation.py          # 标点恢复 (规则版)
│   │       ├── subtitle_formatter.py   # SRT/LRC/ASS 格式化
│   │       ├── text_splitter.py        # 停顿阈值等共享常量
│   │       ├── itn.py                  # 逆文本归一化 (中文数字→阿拉伯)
│   │       ├── bbpe_map.py             # byte-level BPE → 逐字符时间
│   │       └── config_loader.py        # YAML 配置加载
│   ├── config/
│   │   └── models.json         # 模型清单 (下载地址/校验值/大小)
│   ├── run_server.py           # 后端启动入口 (uvicorn)
│   ├── build.py                # PyInstaller 打包脚本
│   ├── config.yaml             # 后端配置文件
│   ├── requirements.txt        # Python 依赖
│   └── tests/                  # 测试目录
│       └── core/
│           ├── test_model_management.py   # 清单/下载/校验/安装移除/取消
│           ├── test_caption_rules.py      # 文本单元/测量词/DP 断句
│           ├── test_engine_routing.py     # 引擎路由与端点
│           ├── test_pipeline_v2.py
│           ├── test_audio_extraction.py
│           └── test_cloud_engine_*.py
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
| 推理引擎 | sherpa-onnx 1.13+ | 本地 ASR 模型推理 (统一路径) |
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

引擎由 `backend/config/models.json` 的 `engine_name` 决定，注册时未安装的模型不会出现在可选列表中。

| 引擎标识符 | 模型 | 语言 | 大小 (int8) |
|-----------|------|------|------|
| `zipformer_local` | Zipformer-CTC-zh | 中文 | 350MB |
| `sensevoice_local` | SenseVoice Small | zh/en/ja/ko/yue | 228MB |
| `firered_local` | FireRedASR2-CTC | zh/en + 20+ 方言 | 740MB |
| `qwen3_asr_local` | Qwen3-ASR-0.6B | 中英日韩粤 | 941MB |
| `cloud` / `cloud:<id>` | 云端配置 (可多个并存) | 多语言 | — |

对齐器 `qwen3-forced-aligner-0.6b` (1GB, 清单 `aligner` 节点) 不是 ASR 引擎:
它拿「音频 + 已识别文本」产出逐词时间, 在 ASR 后由 pipeline 的 `_align_segments`
自动为无逐字符时间的段补测量时间; 不出现在识别下拉, 但在模型面板可下载/移除。
### 流水线 (v2.0)

字幕生成分为 4 个阶段：

1. **ASR** — 引擎转录，输出带逐字符时间戳的段
2. **CORRECT** — 跨段碎片合并 + glossary 替换 + 确定性修正（含标点）
3. **VALIDATE** — 时间戳自检（只记日志，不修改）
4. **FORMAT** — SRT/LRC/ASS 格式化输出

流水线通过 `PipelineContext` 传递中间结果，每个阶段独立可测，异常降级。

---

## 配置体系

### 后端配置 (`backend/config.yaml`)

```yaml
model:
  dir: "~/.shiyu/models"

audio:
  sample_rate: 16000
  max_duration: 300  # 秒, 超长则切分

inference:
  language: "auto"
  use_itn: true
  device: "cpu"

server:
  host: "127.0.0.1"
  port: 11235
  cors_origins: ["http://localhost:1420", "tauri://localhost"]

# key 必须与 config/models.json 的 engine_name 一致
engines:
  zipformer_local:
    enabled: true
    num_threads: 4
    ctc_delay_sec: 0.15
  sensevoice_local:
    enabled: true
    num_threads: 4
  firered_local:
    enabled: true
    num_threads: 4
  qwen3_asr_local:
    enabled: true
    num_threads: 4
    ctc_delay_sec: 0.0
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
| `test_model_management.py` | 清单完整性、路径安全、续传/校验/原子发布、安装移除、取消 |
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
- **引擎可插拔**：所有引擎实现 `BaseTranscriber` 接口，`main.py` 按模型清单注册

---

## 数据与模型

### 模型存储位置

所有模型文件自动下载至 `~/.shiyu/models/`：

| 模型 | 路径 | 大小 | 来源 |
|------|------|------|------|
| Zipformer-CTC-zh | `~/.shiyu/models/zipformer-ctc-zh/` | 350MB | HuggingFace (pinned) |
| SenseVoice-Small | `~/.shiyu/models/sensevoice-small/` | 228MB | HuggingFace (pinned) |
| FireRedASR2-CTC | `~/.shiyu/models/fire-red-asr2-ctc/` | 740MB | HuggingFace (pinned) |
| Qwen3-ASR-0.6B | `~/.shiyu/models/qwen3-asr-0.6b/` | 941MB | HuggingFace (pinned) |

### 模型下载策略

- **不再自动下载**: 启动只报告安装状态, 下载必须由用户在「模型」面板显式发起
- 多源降级：HF-Mirror → HuggingFace 官方 (同一份 SHA-256)
- 下载进度通过 API `/models` 返回给前端 (含阶段、字节数、速度)
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
| v2.1 | Paraformer-CIF 接入, 引擎配置化；主走引擎经回归评估仍为 Zipformer-CTC |
| v2.2 | 模型层改为清单式管理 (pinned revision + SHA-256 + 断点续传 + 安装/移除)；本地模型统一走 sherpa-onnx；断句改为在已测量词边界上做最小代价 DP 切分 |
| v2.3 | 新增 Qwen3-ASR-0.6B (多文件模型, `sherpa_engine._factory_args` 按文件名映射工厂参数)；云端配置改为合并写入, 不再静默丢掉 `request_mode`/`auth_header` 等字段。Whisper 评估后未纳入：sherpa-onnx 的 Whisper 导出缺 cross-attention, 拿不到逐 token 时间戳 |
| v2.4 | 强制对齐器接入 (`aligner` 清单节点 + `forced_aligner.py`, onnxruntime+tokenizers+librosa mel, 不引 transformers)；无逐字符时间的段由 pipeline `_align_segments` 自动补测量时间；修复 DP 行级 char_times 丢失 (v2.2 起) 与下载器嵌套路径；云端配置多化 (`engines` 列表 + `active_engine`, 引擎下拉同时显示)；设置页新增「关于」页 (仓库地址/问题反馈) |

> 上表是内部特性版本 (代码注释里在用)；**发布版本**见 `package.json` / `tauri.conf.json`，
> 两者不是同一套编号。首个发布版 1.2.0 对应上表 v2.4 及之后的全部工作。

| 发布版 | 变更 |
|------|------|
| 1.0.0 / 1.1.0 | 早期发布 (对应上表 v1.x – v2.x) |
| 1.2.0 | 豆包录音文件识别 2.0 云端引擎 (submit/query 两段式, 逐词毫秒时间戳)；剪辑软件导出 (FCPXML / 剪映草稿 / EDL, 每句独立片段可单独加效果)；长音频分块不再在块边界丢内容 (实测文字保留率 100%)；ITN 修复成语与固定搭配被拆成数字 (说一下 ≠ 说1下)；内置云端 provider 清单, 只需填 KEY；默认模型与引擎选择修复 (用用户设置、只挑已安装) |

### 废弃功能

- Bcut (必剪) 和 Jianying (剪映) 免费 ASR 引擎 (2026 年 5 月移除，API 不可达)
- LLM 修正层 (owner 决策，太慢)
- Paraformer 引擎与其 232MB 模型 (无验证收益，且从未接入下载器)
- 标点 NN 后端与其约 50MB 下载链 (`is_available()` 恒为 False，从未生效)
- Silero-VAD 与运行期 `pip install` 路径 (逐字符测量时间戳已覆盖其作用)
- 自研 SenseVoice ONNX 推理 + fbank/LFR/CMVN 预处理 (改由 sherpa-onnx 原生实现)
- 云端 CTC 时间戳精修 (依赖已在 v2.2 删除的 `ctc_aligner`，实际永远返回 None)