<template>
  <div class="app-layout">
    <div class="app-body">
      <!-- 左侧边栏 -->
      <aside class="sidebar">
        <!-- 上传/文件信息区域 -->
        <div class="file-zone">
          <n-upload
            v-if="!selectedFile"
            :max="1"
            accept="audio/*,video/*"
            :show-file-list="false"
            :custom-request="handleUpload"
            class="upload-container"
          >
            <n-upload-dragger
              class="upload-dragger"
              :class="{ 'drag-hover': isDraggingOver }"
            >
              <Upload class="upload-icon" :size="28" />
              <div class="upload-text">拖拽音视频文件到此处</div>
              <div class="upload-subtext">或点击导入文件</div>
            </n-upload-dragger>
          </n-upload>

          <!-- 已选择文件展示卡片 -->
          <div class="file-pill" v-else>
            <div class="file-info">
              <Film v-if="isVideo" class="file-icon" :size="20" />
              <Music v-else class="file-icon" :size="20" />
              <div class="file-details">
                <VTooltip :content="selectedFile.name">
                  <span class="file-name">{{
                    selectedFile.name
                  }}</span>
                </VTooltip>
                <span class="file-meta">{{
                  isVideo ? "视频文件" : "音频文件"
                }}</span>
              </div>
            </div>

            <n-upload
              :key="uploadKey"
              :max="1"
              accept="audio/*,video/*"
              :show-file-list="false"
              :custom-request="handleUpload"
              style="display: inline-flex"
            >
              <button class="secondary-btn small-btn replace-btn">
                更换文件
              </button>
            </n-upload>
          </div>
        </div>

        <!-- 引擎选择器 -->
        <div class="engine-selector" v-if="engines.length > 0">
          <label class="engine-label">识别引擎</label>
          <select v-model="selectedEngine" class="engine-select" :disabled="isProcessing">
            <option v-for="eng in engines" :key="eng.name" :value="eng.name">
              {{ eng.display_name }}{{ eng.available ? '' : ' ⚠️ 不可用' }}
            </option>
          </select>
          <span v-if="selectedEngine && !getEngineInfo(selectedEngine)?.available" class="engine-warning">
            该引擎当前网络不可达，尝试连接可能失败
          </span>
          <button v-if="canRegenerate" class="secondary-btn small-btn regenerate-btn"
            :disabled="isProcessing" @click="onRegenerate">
            ↻ 用 {{ getEngineInfo(selectedEngine)?.display_name || '新引擎' }} 重新生成
          </button>
        </div>

        <!-- 字幕时间偏移 (主人手动微调, 不依赖后端重启) -->
        <div class="offset-control">
          <label class="engine-label">
            字幕时间偏移
            <span class="offset-value">{{ subtitleOffset.toFixed(1) }}s</span>
            <span class="offset-hint">{{ subtitleOffset > 0 ? '(字幕提前)' : subtitleOffset < 0 ? '(字幕推后)' : '' }}</span>
          </label>
          <div class="offset-row">
            <button class="offset-btn" @click="adjustOffset(-0.5)" :disabled="isProcessing">-0.5s</button>
            <SmartSlider
              v-model="subtitleOffset"
              :min="-5"
              :max="5"
              :step="0.1"
              :disabled="isProcessing"
              :format="(v) => v.toFixed(1) + 's'"
              aria-label="字幕时间偏移"
            />
            <button class="offset-btn" @click="adjustOffset(0.5)" :disabled="isProcessing">+0.5s</button>
            <button class="offset-btn offset-reset" @click="subtitleOffset = 0" :disabled="isProcessing">重置</button>
          </div>
          <div class="offset-hint-row">
            字幕晚 2 秒 → 滑到 -2.0 (或点 -0.5s 四次)
          </div>
        </div>

        <!-- 列表头部 -->
        <div class="list-header">
          <span class="list-title">字幕列表</span>
          <span class="list-count">{{ parsedSegments.length }} 条</span>
        </div>

        <!-- 列表 -->
        <div class="subtitle-list" ref="listContainer">
          <div
            v-for="(seg, idx) in parsedSegments"
            :key="idx"
            class="list-item"
            :class="{ active: currentSegIndex === idx }"
            @click="seekToSegment(idx)"
          >
            <span class="item-time">{{ fmtTimeFull(seg.start) }}</span>
            <span class="item-text">{{ seg.text }}</span>
            <VTooltip content="删除此字幕">
              <button class="item-delete" @click.stop="removeSegment(idx)">
                <Trash2 :size="14" />
              </button>
            </VTooltip>
          </div>
          <div class="list-empty" v-if="parsedSegments.length === 0">
            暂无字幕
          </div>
        </div>
      </aside>

      <!-- 右侧主区域 -->
      <main class="main-content">
        <!-- 预览区 -->
        <div class="media-preview-area">
          <div class="media-container">
            <video
              ref="mediaPlayer"
              :src="audioUrl"
              class="media-element"
              @timeupdate="onTimeUpdate"
              @loadedmetadata="onLoadedMetadata"
              @play="onPlay"
              @pause="onPause"
              @ended="onPause"
              v-show="isVideo && audioUrl"
            ></video>

            <div
              class="media-placeholder"
              v-if="!audioUrl || (!isVideo && !isProcessing)"
            >
              <Film :size="56" class="placeholder-icon" />
              <div class="placeholder-text">
                {{
                  audioUrl
                    ? "导入音视频文件以开始预览"
                    : "导入音视频文件以开始预览"
                }}
              </div>
            </div>

            <!-- 视频/音频字幕叠加层 -->
            <div
              class="subtitle-overlay"
              v-if="audioUrl && currentSegment"
              :class="{ 'is-audio': !isVideo }"
            >
              <span class="subtitle-text">{{
                currentSegmentText || currentSegment.text
              }}</span>
            </div>

            <!-- 开始识别的按钮遮罩层 (如果是已导入且尚未识别) -->
            <div
              class="processing-overlay"
              v-if="audioUrl && parsedSegments.length === 0 && !isProcessing"
            >
              <button class="primary-btn run-btn" @click="startTranscribe">
                开始生成字幕
              </button>
            </div>

            <!-- 正在处理 -->
            <div class="processing-overlay" v-if="isProcessing">
              <span class="spinner"></span>
              <div class="processing-text">正在识别中...</div>
            </div>
          </div>
        </div>

        <!-- 编辑区 -->
        <div class="editor-area">
          <div class="editor-header">
            <span class="editor-title">字幕编辑</span>
            <span class="editor-count"
              >{{
                parsedSegments.length > 0
                  ? currentSegIndex >= 0
                    ? currentSegIndex + 1
                    : 1
                  : 0
              }}
              / {{ parsedSegments.length }}</span
            >
          </div>
          <div class="editor-time" v-if="currentSegment">
            开始
            <span>{{ fmtTimeFull(currentSegment.start) }}</span> &nbsp;→&nbsp;
            结束 <span>{{ fmtTimeFull(currentSegment.end) }}</span> &nbsp;时长
            <span
              >{{
                (currentSegment.end - currentSegment.start).toFixed(1)
              }}s</span
            >
          </div>
          <div class="editor-time" v-else>&nbsp;</div>
          <textarea
            class="editor-input"
            placeholder="点击编辑字幕文本..."
            v-model="currentSegmentText"
            :disabled="!currentSegment"
          ></textarea>
          <div class="editor-actions">
            <div class="action-group-left">
              <button
                class="secondary-btn"
                @click="prevSegment"
                :disabled="currentSegIndex <= 0"
              >
                上一条
              </button>
              <button
                class="secondary-btn"
                @click="nextSegment"
                :disabled="
                  currentSegIndex >= parsedSegments.length - 1 ||
                  currentSegIndex < 0
                "
              >
                下一条
              </button>
            </div>
            <div class="action-group-right">
              <button
                class="primary-btn"
                @click="saveCurrentSegment"
                :disabled="!currentSegment"
              >
                保存
              </button>
              <select
                v-model="format"
                class="format-select"
                :disabled="parsedSegments.length === 0"
              >
                <option value="srt">SRT</option>
                <option value="lrc">LRC</option>
                <option value="ass">ASS</option>
                <option value="txt">TXT</option>
              </select>
              <button
                class="secondary-btn"
                :disabled="parsedSegments.length === 0"
                @click="downloadResult"
              >
                导出字幕
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>

    <!-- 底部控制栏 -->
    <footer class="player-controls">
      <!-- 全局底部进度条 -->
      <div
        class="progress-container-global"
        :class="{ dragging: isDraggingProgress }"
        @mousedown="onProgressMouseDown"
        @mousemove="onProgressHover"
        @mouseleave="onProgressLeave"
        ref="progressContainer"
      >
        <div
          class="progress-bar-global"
          :style="{ width: progressPercent + '%' }"
        >
          <div class="progress-thumb"></div>
        </div>
        <!-- 悬停时间提示框 -->
        <div
          class="progress-tooltip"
          v-show="showProgressTooltip"
          :style="{ left: tooltipX + 'px' }"
        >
          {{ tooltipTimeStr }}
        </div>
      </div>

      <div class="control-time">
        {{ fmtTime(playbackTime) }} / {{ fmtTime(mediaDuration) }}
      </div>
      <div class="control-center">
        <button class="icon-btn" @click="skip(-5)">
          <Rewind :size="20" />
        </button>
        <button class="play-btn" @click="togglePlay">
          <Play v-if="!isPlaying" :size="22" class="play-icon" />
          <Pause v-else :size="22" />
        </button>
        <button class="icon-btn" @click="skip(5)">
          <FastForward :size="20" />
        </button>
      </div>
      <div class="control-volume">
        <button class="icon-btn vol-icon" @click="toggleMute">
          <VolumeX v-if="volume === 0 || isMuted" :size="18" />
          <Volume2 v-else :size="18" />
        </button>
        <n-slider
          v-model:value="volume"
          :max="1"
          :step="0.01"
          class="volume-slider"
          @update:value="onVolumeChange"
          :format-tooltip="formatVolumeTooltip"
        />
        <div class="divider"></div>
        <VTooltip content="设置">
          <button class="icon-btn" @click="showSettings = true">
            <Settings :size="18" />
          </button>
        </VTooltip>
      </div>
    </footer>

    <!-- 设置弹窗 -->
    <div
      class="custom-modal-overlay"
      v-if="showSettings"
      @click.self="showSettings = false"
    >
      <div class="custom-modal">
        <div class="custom-modal-header">
          <h3>软件设置</h3>
          <button class="icon-btn" @click="showSettings = false">
            <X :size="18" />
          </button>
        </div>
        <div class="custom-modal-body nav-layout">
          <div class="settings-sidebar">
            <!-- 滑动高亮块: transform 平滑跟随激活项 (借鉴 beUI morphing-tabs) -->
            <div class="settings-nav-glider" :style="navGliderStyle" aria-hidden="true"></div>
            <div
              v-for="tab in settingsTabs"
              :key="tab.key"
              :data-tab="tab.key"
              class="settings-nav-item"
              :class="{ active: currentSettingsTab === tab.key }"
              @click="currentSettingsTab = tab.key"
            >
              <component :is="tab.icon" :size="18" class="nav-icon" />
              <span class="nav-label">{{ tab.label }}</span>
            </div>
          </div>
          <div class="settings-panel">
            <!-- 通用 -->
            <template v-if="currentSettingsTab === 'general'">
              <div class="setting-item">
                <div class="setting-label">开机自动启动</div>
                <n-switch v-model:value="appSettings.autostart" />
              </div>
              <div class="setting-item">
                <div class="setting-label">启动时最小化到托盘</div>
                <n-switch v-model:value="appSettings.startMinimized" />
              </div>
              <div class="setting-item col">
                <div class="setting-label">默认字幕导出目录</div>
                <div class="path-picker">
                  <n-input
                    v-model:value="appSettings.defaultSavePath"
                    placeholder="默认将与源文件同目录..."
                    readonly
                  />
                  <button class="secondary-btn small-btn" @click="selectDefaultDir">
                    选择
                  </button>
                  <button
                    class="secondary-btn small-btn"
                    @click="appSettings.defaultSavePath = ''"
                  >
                    清除
                  </button>
                </div>
              </div>
              <div class="setting-item col">
                <div class="setting-label">
                  自定义极客字典 (一行一个，如: 错词=对词)
                </div>
                <n-input
                  v-model:value="appSettings.customGlossaryText"
                  type="textarea"
                  placeholder="goold it=Godot&#10;思家家=C++&#10;微优伊=Vue"
                  :autosize="{ minRows: 3, maxRows: 6 }"
                />
              </div>
            </template>

            <!-- 模型 -->
            <template v-if="currentSettingsTab === 'model'">
              <div class="setting-item col">
                <div class="setting-label">语音模型</div>
                <div class="model-status-text model-hint">
                  下载后离线运行，识别时不上传音频。已安装的模型可在「识别引擎」中选用。
                </div>
              </div>

              <div v-for="m in models" :key="m.id" class="model-row">
                <div class="model-row-main">
                  <div class="model-row-title">
                    <span class="model-status-dot" :class="m.installed ? 'dot-green' : 'dot-gray'"></span>
                    <span class="model-name">{{ m.name }}</span>
                    <span v-if="m.id === defaultModelID" class="model-tag">默认</span>
                  </div>
                  <div class="model-row-desc">{{ m.description }}</div>
                  <div class="model-row-meta">
                    <span>{{ formatSize(m.sizeBytes) }}</span>
                    <span v-if="m.languages && m.languages.length"> · {{ m.languages.join(" / ") }}</span>
                    <span v-if="!m.installed && m.reason"> · {{ m.reason }}</span>
                  </div>

                  <!-- 该模型的下载进度 -->
                  <div v-if="jobFor(m.id)" class="model-progress">
                    <div class="model-progress-bar">
                      <div class="model-progress-fill"
                        :style="{ width: Math.round((jobFor(m.id).progress || 0) * 100) + '%' }"></div>
                    </div>
                    <div class="model-progress-text">
                      {{ jobStageText(jobFor(m.id)) }}
                    </div>
                  </div>
                </div>

                <div class="model-row-actions">
                  <button v-if="m.installed && m.id !== defaultModelID && m.engine !== 'forced_aligner'"
                    class="secondary-btn small-btn"
                    :disabled="isProcessing"
                    @click="onSetDefaultModel(m)">设为默认</button>
                  <button v-if="!m.installed && !jobFor(m.id)"
                    class="secondary-btn small-btn"
                    :disabled="isProcessing"
                    @click="onInstallModel(m.id)">下载</button>
                  <button v-else-if="jobFor(m.id) && jobFor(m.id).status === 'running'"
                    class="secondary-btn small-btn"
                    @click="onCancelModelJob(jobFor(m.id).jobID)">取消</button>
                  <button v-else-if="m.installed"
                    class="secondary-btn small-btn danger-btn"
                    :disabled="isProcessing"
                    @click="onRemoveModel(m)">移除</button>
                </div>
              </div>

              <div class="setting-item">
                <button class="secondary-btn small-btn" @click="openModelDir">
                  打开模型文件夹
                </button>
              </div>
            </template>

            <!-- 云端 -->
            <!-- 云端: 多配置并存 (v2.4) -->
            <template v-if="currentSettingsTab === 'cloud'">
              <div class="setting-item col">
                <div class="setting-label">云端配置（可添加多个，同时显示在引擎下拉）</div>
                <select class="log-level-select" v-model="editingEngineId">
                  <option v-for="e in engineConfigs" :key="e.id" :value="e.id">
                    {{ (e.display_name || e.model || '(未命名)') }}{{ activeEngineId === e.id ? ' ✓当前' : '' }}
                  </option>
                </select>
                <div class="setting-row">
                  <button class="secondary-btn small-btn" @click="onAddEngineConfig">+ 新增配置</button>
                  <button v-if="engineConfigs.length > 1" class="secondary-btn small-btn danger-btn"
                    @click="onDeleteEngineConfig">删除当前</button>
                  <button v-if="activeEngineId !== editingEngineId" class="secondary-btn small-btn"
                    @click="onActivateEngine">设为当前使用</button>
                </div>
              </div>
              <div class="setting-item col">
                <div class="setting-label">API 地址</div>
                <n-input v-model:value="editingEngine.api_url"
                  placeholder="https://api.openai.com/v1" />
              </div>
              <div class="setting-item col">
                <div class="setting-label">API Key</div>
                <n-input v-model:value="editingEngine.api_key"
                  :type="showEngineApiKey ? 'text' : 'password'"
                  :placeholder="editingEngine.has_key ? '已保存（留空则不修改）' : 'sk-...'">
                  <template #suffix>
                    <button class="icon-btn" @click="showEngineApiKey = !showEngineApiKey" tabindex="-1">
                      <Eye v-if="!showEngineApiKey" :size="16" />
                      <EyeOff v-else :size="16" />
                    </button>
                  </template>
                </n-input>
              </div>
              <div class="setting-item col">
                <div class="setting-label">模型名</div>
                <n-input v-model:value="editingEngine.model" placeholder="whisper-1" />
              </div>
              <div class="setting-item col">
                <div class="setting-label">请求模式</div>
                <select class="log-level-select" v-model="editingEngine.request_mode">
                  <option value="whisper">whisper — /audio/transcriptions</option>
                  <option value="chat_completions">chat_completions — /chat/completions</option>
                </select>
                <div class="model-row-desc">
                  OpenAI、Groq 等选 whisper；DashScope Qwen3-ASR 等只提供 chat 接口的选 chat_completions。
                </div>
              </div>
              <div class="setting-item col">
                <div class="setting-label">显示名称（可选，用于区分多个云模型）</div>
                <n-input v-model:value="editingEngine.display_name" placeholder="DashScope" />
              </div>
              <div class="setting-item">
                <div class="setting-label">启用该配置</div>
                <n-switch v-model:value="editingEngine.enabled" />
              </div>
              <div class="setting-item">
                <button class="secondary-btn" @click="onSaveEngineSettings">保存全部配置</button>
                <span v-if="engineSettingsStatus" class="setting-status">{{ engineSettingsStatus }}</span>
              </div>
            </template>

            <!-- 日志 -->
            <template v-if="currentSettingsTab === 'log'">
              <div class="setting-item">
                <div class="setting-label">排错日志</div>
                <button class="secondary-btn small-btn" @click="openLogDir">
                  打开日志文件夹
                </button>
              </div>
              <div class="setting-item">
                <div class="setting-label">日志级别</div>
                <select class="log-level-select" v-model="appSettings.logLevel" @change="onLogLevelChange">
                  <option value="debug">DEBUG</option>
                  <option value="info">INFO</option>
                  <option value="warning">WARNING</option>
                  <option value="error">ERROR</option>
                </select>
              </div>
            </template>

            <!-- 关于: 项目地址与版本 -->
            <template v-if="currentSettingsTab === 'about'">
              <div class="setting-item col">
                <div class="setting-label">项目仓库</div>
                <a class="about-link" :href="PROJECT_INFO.repo" target="_blank" rel="noopener">
                  <Github :size="15" class="about-link-icon" />
                  {{ PROJECT_INFO.repo }}
                </a>
                <div class="model-row-desc">GitHub 仓库地址 —— 问题反馈、文档与源码</div>
              </div>
              <div class="setting-item col">
                <div class="setting-label">问题反馈</div>
                <a class="about-link" :href="PROJECT_INFO.issues" target="_blank" rel="noopener">
                  {{ PROJECT_INFO.issues }}
                </a>
                <div class="model-row-desc">提交 Issue 时请附上排错日志（日志页可打开日志文件夹）</div>
              </div>
              <div class="setting-item">
                <div class="setting-label">版本</div>
                <span class="setting-status">{{ PROJECT_INFO.version }}</span>
              </div>
              <div class="setting-item">
                <div class="setting-label">开源许可</div>
                <span class="setting-status">MIT</span>
              </div>
            </template>
            <div class="copyright-info">
              © 2026 时语 Shiyu Subtitle {{ PROJECT_INFO.version }} · 开源 (MIT)
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- 模型下载进度悬浮条 (不再遮挡整个界面: 未下载模型时由识别报错引导) -->
  <div class="download-toast" v-if="activeJob">
    <div class="download-toast-title">{{ activeJob.modelName }} 下载中</div>
    <div class="model-progress-bar">
      <div class="model-progress-fill"
        :style="{ width: Math.round((activeJob.progress || 0) * 100) + '%' }"></div>
    </div>
    <div class="download-toast-text">
      {{ jobStageText(activeJob) }} · {{ formatSize(activeJob.completedBytes || 0) }} / {{ formatSize(activeJob.totalBytes || 0) }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from "vue";
import {
  NUpload,
  NUploadDragger,
  NSlider,
  NSwitch,
  NInput,
  useMessage,
} from "naive-ui";
import {
  Settings,
  Upload,
  Film,
  Music,
  Rewind,
  Play,
  FastForward,
  Volume2,
  VolumeX,
  X,
  Trash2,
  Mic,
  Cloud,
  BrainCircuit,
  FileText,
  Info,
  Github,
  Pause,
  Eye,
  EyeOff,
} from "lucide-vue-next";
import { listen } from "@tauri-apps/api/event";
import { convertFileSrc } from "@tauri-apps/api/tauri";
import { save, open } from "@tauri-apps/api/dialog";
import { writeTextFile } from "@tauri-apps/api/fs";
import VTooltip from "./VTooltip.vue";
import SmartSlider from "./SmartSlider.vue";
import {
  healthCheck,
  transcribeFile,
  transcribeJson,
  transcribeLocal,
  downloadText,
  openLogFolder,
  getModels,
  installModel,
  removeModel,
  cancelModelJob,
  setDefaultModel,
  openModelFolder,
  setLogLevel,
  getEngineConfigs,
  saveEngineConfigsApi,
  getEngines,
  getLlmCorrectionSettings,
  saveLlmCorrectionSettings,
  correctTranscription,
} from "../api.js";

const message = useMessage();

// State
const selectedFile = ref(null);
const localFilePath = ref(null);
const audioUrl = ref(null);
const isVideo = ref(false);
const isProcessing = ref(false);
const isCorrecting = ref(false);
const backendOnline = ref(false);

// Engine selector state
const engines = ref([]);
const selectedEngine = ref("zipformer_local");
const subtitleOffset = ref(0.0);
function adjustOffset(delta) {
  subtitleOffset.value = Math.max(-5, Math.min(5, subtitleOffset.value + delta));
}

// 模型清单状态 (清单式管理: 每个模型独立下载/移除)
const models = ref([]);
const defaultModelID = ref("");
const activeJob = ref(null);
const modelPath = ref("");
let modelPollTimer = null;

/** 取某模型当前运行中的下载任务 (无则 null)。 */
function jobFor(modelId) {
  const job = activeJob.value;
  return job && job.modelID === modelId && job.status === "running" ? job : null;
}

/** 把任务阶段翻成中文进度描述。 */
function jobStageText(job) {
  if (!job) return "";
  if (job.status === "failed") return job.error || "下载失败";
  if (job.status === "cancelled") return job.error || "已取消";
  const percent = Math.round((job.progress || 0) * 100);
  const stage = {
    connecting: "连接中",
    "switching-source": `切换镜像 (${job.source || ""})`,
    downloading: "下载中",
    resuming: "继续下载",
    checking: "校验中",
    cancelling: "正在取消",
    ready: "完成",
  }[job.stage] || "处理中";
  return `${stage} ${percent}%`;
}

/** 字节数 → 人类可读大小。 */
function formatSize(bytes) {
  const value = Number(bytes) || 0;
  if (value >= 1024 ** 3) return (value / 1024 ** 3).toFixed(2) + " GB";
  if (value >= 1024 ** 2) return (value / 1024 ** 2).toFixed(0) + " MB";
  if (value >= 1024) return (value / 1024).toFixed(0) + " KB";
  return value + " B";
}


const format = ref("srt");
const language = ref("zh");

const resultSegments = ref([]);
// 生成字幕时实际用的引擎; 切换后不同则提示可重新生成 (问题 2)
const lastGeneratedEngine = ref("");
const canRegenerate = computed(() =>
  !!lastGeneratedEngine.value &&
  !!selectedEngine.value &&
  lastGeneratedEngine.value !== selectedEngine.value &&
  resultSegments.value.length > 0
);
const uploadKey = ref(0);
const currentSegIndex = ref(-1);
const currentSegmentText = ref("");
const showEngineApiKey = ref(false);
const showLlmApiKey = ref(false);

// Settings state
const showSettings = ref(false);
const currentSettingsTab = ref("general");
const settingsTabs = [
  { key: "general", icon: Settings, label: "通用" },
  { key: "model", icon: Mic, label: "模型" },
  { key: "cloud", icon: Cloud, label: "云端" },
  { key: "log", icon: FileText, label: "日志" },
  { key: "about", icon: Info, label: "关于" },
];

// ── 设置侧边栏滑动高亮块 (morphing) ──
const navGlider = ref({ top: 0, height: 0, ready: false });
const navGliderStyle = computed(() => ({
  transform: `translateY(${navGlider.value.top}px)`,
  height: `${navGlider.value.height}px`,
  opacity: navGlider.value.ready ? 1 : 0,
}));

function updateNavGlider() {
  // 不用 v-for ref 数组 (组件更新时 ref 回调先置 null 会错位),
  // 直接从 DOM 按 data-tab 查, 并要求布局已稳定 (有非零高度)
  const items = document.querySelectorAll(".settings-nav-item[data-tab]");
  const el = [...items].find(
    (n) => n.dataset.tab === currentSettingsTab.value
  );
  if (!el || el.offsetHeight < 8) return;
  navGlider.value = {
    top: el.offsetTop,
    height: el.offsetHeight,
    ready: true,
  };
}

watch(currentSettingsTab, () => nextTick(updateNavGlider));
watch(showSettings, (open) => {
  if (!open) return;
  // 弹窗有缩放动画, 动画进行中 offsetTop/Height 不稳定;
  // 首次测量 + 动画结束后再校正一次, 两值一致才用
  nextTick(updateNavGlider);
  setTimeout(updateNavGlider, 420);
});
let navResizeObserver = null;
onMounted(() => {
  // 字体加载可能改变行高; resize 与侧边栏尺寸变化都要重算
  nextTick(updateNavGlider);
  window.addEventListener("resize", updateNavGlider);
  const sidebar = document.querySelector(".settings-sidebar");
  if (sidebar && "ResizeObserver" in window) {
    navResizeObserver = new ResizeObserver(() => updateNavGlider());
    navResizeObserver.observe(sidebar);
  }
});
onBeforeUnmount(() => {
  window.removeEventListener("resize", updateNavGlider);
  navResizeObserver?.disconnect();
});

// 项目信息 (关于页展示; 发布包里也应可追溯)
const PROJECT_INFO = {
  repo: "https://github.com/JochenYang/shiyu",
  issues: "https://github.com/JochenYang/shiyu/issues",
  version: "v2.4",
};
const appSettings = ref({
  autostart: false,
  startMinimized: false,
  defaultSavePath: "",
  logLevel: "info",
  // 云端多配置状态在 engineConfigs/activeEngineId/editingEngineId (v2.4),
  // 旧单配置字段已移除 — 保留这里会让人误以为还有第二个数据源。
  llmEnabled: false,
  llmProvider: "openai",
  llmApiUrl: "",
  llmApiKey: "",
  llmModel: "",
});
// Player State
const mediaPlayer = ref(null);
const isPlaying = ref(false);
const mediaDuration = ref(0);
const playbackTime = ref(0);
const volume = ref(1);
const isMuted = ref(false);
const prevVolume = ref(1);
const progressContainer = ref(null);
const listContainer = ref(null);

// Progress Tooltip & Drag State
const showProgressTooltip = ref(false);
const tooltipX = ref(0);
const tooltipTimeStr = ref("00:00:00");
const isDraggingProgress = ref(false);
const dragPercent = ref(0);

const isDraggingOver = ref(false);
let pollTimer = null;
let unlistenDrop = null;
let unlistenHover = null;
let unlistenCancel = null;

const MEDIA_EXTENSIONS =
  /\.(mp4|mkv|avi|mov|wmv|flv|webm|mp3|wav|m4a|flac|ogg|aac)$/i;
const VIDEO_EXTENSIONS = /\.(mp4|mkv|avi|mov|wmv|flv|webm)$/i;

const parsedSegments = computed(() => {
  return resultSegments.value;
});

const progressPercent = computed(() => {
  if (mediaDuration.value <= 0) return 0;
  if (isDraggingProgress.value) return dragPercent.value * 100;
  return (playbackTime.value / mediaDuration.value) * 100;
});

const currentSegment = computed(() => {
  if (
    currentSegIndex.value >= 0 &&
    currentSegIndex.value < parsedSegments.value.length
  ) {
    return parsedSegments.value[currentSegIndex.value];
  }
  return null;
});

// Sync text input with current segment
watch(currentSegIndex, (idx) => {
  if (idx >= 0 && idx < parsedSegments.value.length) {
    currentSegmentText.value = parsedSegments.value[idx].text;
  } else {
    currentSegmentText.value = "";
  }
  // Auto-scroll the active list item into view during playback
  if (idx >= 0 && listContainer.value) {
    nextTick(() => {
      const items = listContainer.value.querySelectorAll(".list-item");
      if (items[idx]) {
        items[idx].scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  }
});

const engineSettingsStatus = ref("");

// ── 云端多配置 (v2.4) ──────────────────────────────────────────
const engineConfigs = ref([]);          // 后端全部配置 (脱敏)
const activeEngineId = ref("");          // 当前激活配置 id
const editingEngineId = ref("");

// 正在编辑的配置对象 (直接绑定表单; has_key 由后端回传, 不回传)
const editingEngine = computed(() => {
  const found = engineConfigs.value.find(e => e.id === editingEngineId.value);
  return found || { api_url: "", api_key: "", model: "", enabled: false,
                    request_mode: "whisper", display_name: "" };
});

function _blankEngine() {
  return { id: "", api_url: "", api_key: "", model: "", enabled: true,
           request_mode: "whisper", display_name: "", has_key: false };
}

async function fetchEngineConfigs() {
  try {
    const data = await getEngineConfigs();
    engineConfigs.value = data.engines || [];
    activeEngineId.value = data.active || "";
    if (!engineConfigs.value.length) {
      engineConfigs.value = [_blankEngine()];
      editingEngineId.value = engineConfigs.value[0].id;
    } else if (!engineConfigs.value.some(e => e.id === editingEngineId.value)) {
      editingEngineId.value = activeEngineId.value || engineConfigs.value[0].id;
    }
  } catch {
    // 后端未就绪 (开发模式) — 留空白供手填
    if (!engineConfigs.value.length) {
      engineConfigs.value = [_blankEngine()];
      editingEngineId.value = engineConfigs.value[0].id;
    }
  }
}

function onAddEngineConfig() {
  const fresh = _blankEngine();
  fresh.id = `new-${Date.now()}`;   // 临时 id, 后端保存时重新分配
  engineConfigs.value.push(fresh);
  editingEngineId.value = fresh.id;
}

function onDeleteEngineConfig() {
  if (engineConfigs.value.length <= 1) return;
  const idx = engineConfigs.value.findIndex(e => e.id === editingEngineId.value);
  if (idx < 0) return;
  const removed = engineConfigs.value.splice(idx, 1)[0];
  if (activeEngineId.value === removed.id) {
    activeEngineId.value = engineConfigs.value[0]?.id || "";
  }
  editingEngineId.value = activeEngineId.value || engineConfigs.value[0]?.id || "";
}

async function onActivateEngine() {
  // 立即保存并把当前编辑项设为激活
  await onSaveEngineSettings();
}

async function onSaveEngineSettings() {
  try {
    engineSettingsStatus.value = "保存中...";
    await saveEngineConfigsApi({
      engines: engineConfigs.value.map(e => ({
        id: e.id && !String(e.id).startsWith("new-") ? e.id : undefined,
        api_url: e.api_url || "",
        api_key: e.api_key || "",          // 空值后端保留旧密钥
        model: e.model || "",
        enabled: !!e.enabled,
        request_mode: e.request_mode || "whisper",
        display_name: e.display_name || "",
      })),
      active: activeEngineId.value || undefined,
    });
    engineSettingsStatus.value = "✓ 已保存";
    setTimeout(() => { engineSettingsStatus.value = ""; }, 3000);
    await fetchEngineConfigs();          // 回读真实 id 与激活态
    await fetchEngines();                 // 云端引擎下拉同步
  } catch (err) {
    engineSettingsStatus.value = "保存失败: " + (err.message || err);
  }
}

const llmSettingsStatus = ref("");

async function syncLlmCorrectionSettings() {
  try {
    const data = await getLlmCorrectionSettings();
    appSettings.value.llmEnabled = data.enabled || false;
    appSettings.value.llmProvider = data.provider || "openai";
    appSettings.value.llmApiUrl = data.api_url || "";
    appSettings.value.llmApiKey = data.api_key || "";
    appSettings.value.llmModel = data.model || "";
  } catch (e) {
    console.warn("Failed to load LLM correction settings:", e);
  }
}

async function onToggleLlmEnabled(val) {
  // Auto-save on toggle
  appSettings.value.llmEnabled = val;
  await onSaveLlmCorrectionSettings();
}
async function onSaveLlmCorrectionSettings() {
  llmSettingsStatus.value = "";
  try {
    const settings = {
      enabled: appSettings.value.llmEnabled,
      provider: appSettings.value.llmProvider,
      api_url: appSettings.value.llmApiUrl,
      api_key: appSettings.value.llmApiKey === "********" ? "" : appSettings.value.llmApiKey,
      model: appSettings.value.llmModel,
    };
    await saveLlmCorrectionSettings(settings);
    llmSettingsStatus.value = "保存成功 ✓";
    setTimeout(() => { llmSettingsStatus.value = ""; }, 3000);
  } catch (e) {
    llmSettingsStatus.value = "保存失败: " + e.message;
  }
}

// Load settings
onMounted(async () => {
  const saved = localStorage.getItem("shiyu_settings");
  if (saved) {
    appSettings.value = { ...appSettings.value, ...JSON.parse(saved) };
  }

  // Sync saved log level to backend on startup
  if (appSettings.value.logLevel && appSettings.value.logLevel !== "info") {
    try {
      await setLogLevel(appSettings.value.logLevel);
    } catch {
      // Backend not ready yet; will retry on user change
    }
  }

  // Sync cloud engine configs (multi, v2.4) from backend
  try {
    await fetchEngineConfigs();
  } catch {
    // Backend may not have engine endpoint yet during dev
  }

  // Sync LLM correction settings from backend
  syncLlmCorrectionSettings();

  // Fetch available engines + model list
  await fetchEngines();
  await refreshModels();

  pollTimer = setInterval(async () => {
    try {
      await healthCheck();
      if (!backendOnline.value) {
        backendOnline.value = true;
        // Fetch engines once backend comes online
        fetchEngines();
      }
      // 有下载任务在跑时刷新清单, 以便看到进度
      if (activeJob.value) {
        refreshModels();
      }
    } catch {
      backendOnline.value = false;
    }
  }, 2000);

  try {
    unlistenDrop = await listen("tauri://file-drop", (event) => {
      isDraggingOver.value = false;
      const paths = event.payload;
      if (!paths || paths.length === 0) return;
      const mediaPath = paths.find((p) => MEDIA_EXTENSIONS.test(p));
      if (!mediaPath) {
        message.warning("不支持的文件格式，请拖入音频/视频文件");
        return;
      }
      handleFileSelected(mediaPath, true);
    });

    unlistenHover = await listen("tauri://file-drop-hover", () => {
      isDraggingOver.value = true;
    });

    unlistenCancel = await listen("tauri://file-drop-cancelled", () => {
      isDraggingOver.value = false;
    });
  } catch {
    // Running in browser dev mode
  }
});

watch(
  appSettings,
  (newVal) => {
    localStorage.setItem("shiyu_settings", JSON.stringify(newVal));
  },
  { deep: true },
);

onBeforeUnmount(() => {
  clearInterval(pollTimer);
  if (modelPollTimer) clearInterval(modelPollTimer);
  if (audioUrl.value && !localFilePath.value)
    URL.revokeObjectURL(audioUrl.value);
  unlistenDrop?.();
  unlistenHover?.();
  unlistenCancel?.();
});

function handleFileSelected(filePathOrFile, isTauriPath = false) {
  uploadKey.value++;
  if (audioUrl.value && !localFilePath.value)
    URL.revokeObjectURL(audioUrl.value);

  resultSegments.value = [];
  currentSegIndex.value = -1;
  playbackTime.value = 0;
  mediaDuration.value = 0;
  isPlaying.value = false;

  if (isTauriPath) {
    const fileName = filePathOrFile.split(/[\\/]/).pop() || "file";
    localFilePath.value = filePathOrFile;
    selectedFile.value = { name: fileName, size: 0 };
    audioUrl.value = convertFileSrc(filePathOrFile);
    isVideo.value = VIDEO_EXTENSIONS.test(filePathOrFile);
  } else {
    const file = filePathOrFile;
    selectedFile.value = file;
    localFilePath.value = null;
    audioUrl.value = URL.createObjectURL(file);
    isVideo.value = VIDEO_EXTENSIONS.test(file.name);
  }

  message.info(`已加载文件: ${selectedFile.value.name}`);
}

function handleUpload({ file, onFinish, onError }) {
  try {
    handleFileSelected(file.file, false);
    onFinish();
  } catch (e) {
    onError();
    message.error("导入文件失败");
  }
}

async function onCorrectWithLLM() {
  isCorrecting.value = true;
  try {
    const data = await correctTranscription(parsedSegments.value, selectedEngine.value);
    if (data.segments) {
      resultSegments.value = data.segments;
      if (data.llm_corrected) {
        message.success(`已修正 ${data.segments.length} 条字幕`);
      } else {
        message.info("修正无变化");
      }
    }
  } catch (e) {
    message.error("LLM 修正失败: " + e.message);
  } finally {
    isCorrecting.value = false;
  }
}
// 切换引擎后重新生成 (问题 2): 复用当前文件与设置, 走同一条转录链路
async function onRegenerate() {
  if (isProcessing.value) return;
  if (!localFilePath.value && !selectedFile.value) {
    message.warning("没有可重新生成的文件");
    return;
  }
  message.info("正在用新引擎重新生成字幕…");
  await startTranscribe();
}

async function startTranscribe() {
  if (!selectedFile.value) return;
  isProcessing.value = true;
  resultSegments.value = [];
  currentSegIndex.value = -1;
  // 记住本次生成用的引擎, 供「切换引擎后重新生成」提示判断
  lastGeneratedEngine.value = selectedEngine.value;
  try {
    let glossaryDict = {};
    if (appSettings.value.customGlossaryText) {
      const lines = appSettings.value.customGlossaryText.split("\n");
      for (const line of lines) {
        if (line.includes("=")) {
          const [bad, good] = line.split("=");
          if (bad.trim() && good.trim()) {
            glossaryDict[bad.trim()] = good.trim();
          }
        }
      }
    }
    const glossaryStr = JSON.stringify(glossaryDict);
    // 云端选中态: 引擎下拉里激活云配置叫 "cloud" (后端保证), 兼容旧回退
    const currentEngine = selectedEngine.value || (activeEngineId.value ? "cloud" : "zipformer_local");
    let llmApplied = false;

    if (localFilePath.value) {
      const result = await transcribeLocal(localFilePath.value, {
        language: language.value,
        output_format: format.value,
        glossary: glossaryStr,
        engine: currentEngine,
        subtitleOffset: subtitleOffset.value,
      });
      if (result.segments) {
        resultSegments.value = result.segments;
      }
      llmApplied = result.llm_corrected;
    } else {
      const json = await transcribeJson(
        selectedFile.value,
        language.value,
        glossaryStr,
        currentEngine,
        subtitleOffset.value,
      );
      if (json?.segments) {
        resultSegments.value = json.segments;
      }
      llmApplied = json?.llm_corrected;
    }
    if (resultSegments.value.length > 0) {
      currentSegIndex.value = 0;
    }
    if (resultSegments.value.length > 0) {
      currentSegIndex.value = 0;
      message.success(llmApplied ? "✓ 已修正" : "字幕生成完成");
    } else {
      // 0 段不能只说「完成」: 主人会以为成功了却找不到字幕。
      // 常见原因按可能性排序给提示 (静音/无语音/引擎输出为空)。
      message.warning(
        "未识别到语音内容：请确认视频包含清晰的说话声，\n" +
        "或换一个识别引擎（如 SenseVoice Small）重试",
        { duration: 6000 }
      );
    }
    message.error(err.message || "处理失败，请检查后端服务");
  } finally {
    isProcessing.value = false;
  }
}

// Player controls
function togglePlay() {
  if (!mediaPlayer.value || !audioUrl.value) return;
  if (isPlaying.value) {
    mediaPlayer.value.pause();
  } else {
    mediaPlayer.value.play();
  }
}

function skip(seconds) {
  if (!mediaPlayer.value) return;
  mediaPlayer.value.currentTime += seconds;
}

function onPlay() {
  isPlaying.value = true;
}
function onPause() {
  isPlaying.value = false;
}
function onLoadedMetadata() {
  if (mediaPlayer.value) {
    mediaDuration.value = mediaPlayer.value.duration;
  }
}
function onTimeUpdate() {
  if (mediaPlayer.value) {
    playbackTime.value = mediaPlayer.value.currentTime;
    autoUpdateSegmentIndex(playbackTime.value);
  }
}

function onVolumeChange(val) {
  if (mediaPlayer.value) {
    mediaPlayer.value.volume = val;
    if (val > 0) isMuted.value = false;
  }
}

function toggleMute() {
  if (!mediaPlayer.value) return;
  if (isMuted.value || volume.value === 0) {
    // Unmute
    isMuted.value = false;
    volume.value = prevVolume.value > 0 ? prevVolume.value : 0.5;
    mediaPlayer.value.volume = volume.value;
  } else {
    // Mute
    isMuted.value = true;
    prevVolume.value = volume.value;
    volume.value = 0;
    mediaPlayer.value.volume = 0;
  }
}

function formatVolumeTooltip(val) {
  return `${Math.round(val * 100)}%`;
}

function updateProgressFromMouse(e) {
  if (!mediaDuration.value || !progressContainer.value || !mediaPlayer.value)
    return;
  const rect = progressContainer.value.getBoundingClientRect();
  let clickX = e.clientX - rect.left;
  clickX = Math.max(0, Math.min(clickX, rect.width));
  const percent = clickX / rect.width;
  dragPercent.value = percent;
}

function onProgressMouseDown(e) {
  if (!mediaDuration.value || !progressContainer.value || !mediaPlayer.value)
    return;
  isDraggingProgress.value = true;
  updateProgressFromMouse(e);

  window.addEventListener("mousemove", onProgressMouseMove);
  window.addEventListener("mouseup", onProgressMouseUp);
}

function onProgressMouseMove(e) {
  if (isDraggingProgress.value) {
    updateProgressFromMouse(e);
    onProgressHover(e); // 同时更新悬停时间
  }
}

function onProgressMouseUp(e) {
  if (isDraggingProgress.value) {
    isDraggingProgress.value = false;
    window.removeEventListener("mousemove", onProgressMouseMove);
    window.removeEventListener("mouseup", onProgressMouseUp);
    if (mediaDuration.value && mediaPlayer.value) {
      mediaPlayer.value.currentTime = dragPercent.value * mediaDuration.value;
    }
  }
}

function onProgressHover(e) {
  if (!mediaDuration.value || !progressContainer.value) return;
  const rect = progressContainer.value.getBoundingClientRect();
  const hoverX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
  const percent = hoverX / rect.width;
  tooltipX.value = hoverX;
  tooltipTimeStr.value = fmtTimeFull(percent * mediaDuration.value).split(
    ".",
  )[0]; // 只显示到秒
  showProgressTooltip.value = true;
}

function onProgressLeave() {
  showProgressTooltip.value = false;
}

let lastManualSeekTime = 0;

function autoUpdateSegmentIndex(time) {
  if (Date.now() - lastManualSeekTime < 800) return;
  if (parsedSegments.value.length === 0) return;

  // Step 1: Handle fallback for untimestamped subtitles (all start/end are 0)
  const isAllZero = parsedSegments.value.every(seg => seg.start === 0 && seg.end === 0);
  if (isAllZero && mediaDuration.value > 0) {
    const totalChars = parsedSegments.value.reduce((sum, seg) => sum + (seg.text || "").length, 0) || 1;
    let accumulatedTime = 0;
    for (let i = 0; i < parsedSegments.value.length; i++) {
      const seg = parsedSegments.value[i];
      const duration = ((seg.text || "").length / totalChars) * mediaDuration.value;
      const segStart = accumulatedTime;
      const segEnd = accumulatedTime + duration;
      if (time >= segStart && time <= segEnd) {
        currentSegIndex.value = i;
        return;
      }
      accumulatedTime = segEnd;
    }
    return;
  }

  // Step 2: Standard time-range match for timestamped subtitles
  const idx = parsedSegments.value.findIndex(
    (seg) => time >= seg.start && time <= seg.end,
  );
  if (idx !== -1) {
    currentSegIndex.value = idx;
  }
}

function seekToSegment(idx) {
  lastManualSeekTime = Date.now();
  currentSegIndex.value = idx;
  if (mediaPlayer.value && parsedSegments.value[idx]) {
    // Step 1: Seek proportionally if all timestamps are 0
    const isAllZero = parsedSegments.value.every(seg => seg.start === 0 && seg.end === 0);
    if (isAllZero && mediaDuration.value > 0) {
      const totalChars = parsedSegments.value.reduce((sum, seg) => sum + (seg.text || "").length, 0) || 1;
      let accumulatedTime = 0;
      for (let i = 0; i < idx; i++) {
        accumulatedTime += ((parsedSegments.value[i].text || "").length / totalChars) * mediaDuration.value;
      }
      mediaPlayer.value.currentTime = accumulatedTime;
    } else {
      mediaPlayer.value.currentTime = parsedSegments.value[idx].start;
    }
  }

  // 滚动列表定位
  setTimeout(() => {
    if (!listContainer.value) return;
    const items = listContainer.value.querySelectorAll(".list-item");
    if (items[idx]) {
      items[idx].scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, 50);
}

// Editor actions
function prevSegment() {
  if (currentSegIndex.value > 0) {
    seekToSegment(currentSegIndex.value - 1);
  }
}
function removeSegment(idx) {
  if (idx < 0 || idx >= parsedSegments.value.length) return;
  parsedSegments.value.splice(idx, 1);
  if (currentSegIndex.value === idx) {
    currentSegIndex.value = -1;
  } else if (currentSegIndex.value > idx) {
    currentSegIndex.value--;
  }
}
function nextSegment() {
  if (currentSegIndex.value < parsedSegments.value.length - 1) {
    seekToSegment(currentSegIndex.value + 1);
  }
}
function saveCurrentSegment() {
  if (
    currentSegIndex.value >= 0 &&
    currentSegIndex.value < parsedSegments.value.length
  ) {
    parsedSegments.value[currentSegIndex.value].text = currentSegmentText.value;
    message.success("已保存修改");
  }
}

async function downloadResult() {
  if (parsedSegments.value.length === 0) return;
  // Format segments to SRT
  let srtContent = "";
  parsedSegments.value.forEach((seg, i) => {
    srtContent += `${i + 1}\n`;
    srtContent += `${fmtTimeSrt(seg.start)} --> ${fmtTimeSrt(seg.end)}\n`;
    srtContent += `${seg.text}\n\n`;
  });

  const defaultName =
    selectedFile.value?.name.replace(/\.[^.]+$/, "") || "subtitle";

  try {
    const filePath = await save({
      defaultPath: appSettings.value.defaultSavePath
        ? `${appSettings.value.defaultSavePath}\\${defaultName}.srt`
        : `${defaultName}.srt`,
      filters: [{ name: "Subtitle", extensions: ["srt"] }],
    });

    if (filePath) {
      await writeTextFile(filePath, srtContent);
      message.success("导出成功！");
    }
  } catch (err) {
    // Fallback to browser download if not running in Tauri
    if (
      err.toString().includes("__TAURI_IPC__") ||
      err.toString().includes("window.__TAURI__")
    ) {
      downloadText(srtContent, `${defaultName}.srt`);
    } else {
      message.error("导出失败: " + err.toString());
    }
  }
}

async function selectDefaultDir() {
  try {
    const selected = await open({
      directory: true,
      multiple: false,
    });
    if (selected) {
      appSettings.value.defaultSavePath = selected;
    }
  } catch (err) {
    message.error("选择目录失败");
  }
}

async function openLogDir() {
  try {
    const res = await openLogFolder();
    if (res.status !== "ok") {
      message.error(res.msg || "无法打开日志文件夹，请检查后端服务");
    }
  } catch (err) {
    message.error("调用后端打开日志文件夹失败: " + err);
  }
}

async function onLogLevelChange() {
  try {
    await setLogLevel(appSettings.value.logLevel);
  } catch (err) {
    message.error("设置日志级别失败: " + err);
  }
}

function getEngineInfo(name) {
  return engines.value.find(e => e.name === name) || null;
}

async function fetchEngines() {
  try {
    const data = await getEngines();
    if (data && data.engines && data.engines.length > 0) {
      engines.value = data.engines;
      // Migrate selected engine: if current selection not in list, pick first
      if (!data.engines.some(e => e.name === selectedEngine.value)) {
        selectedEngine.value = data.engines[0].name;
      }
    }
  } catch {
    // Backend may not have /engines endpoint (old version)
  }
}

/** 拉取模型清单与当前下载任务 (只读, 不触发任何下载)。 */
async function refreshModels() {
  try {
    const data = await getModels();
    models.value = data.models || [];
    defaultModelID.value = data.defaultModelID || "";
    activeJob.value = data.job || null;
    // 引擎列表随安装状态变化, 同步刷新引擎选择器
    await fetchEngines();
  } catch {
    // Backend not ready yet, will retry
  }
}

/** 下载模型。 */
async function onInstallModel(modelId) {
  try {
    activeJob.value = await installModel(modelId);
    await refreshModels();
  } catch (err) {
    message.error("无法开始下载: " + (err.message || err));
  }
}

/** 取消运行中的下载任务 (已下载部分保留, 重试可续传)。 */
async function onCancelModelJob(jobId) {
  try {
    await cancelModelJob(jobId);
    await refreshModels();
  } catch (err) {
    message.error("取消失败: " + (err.message || err));
  }
}

/** 移除模型。删除是不可逆操作, 先确认。 */
async function onRemoveModel(model) {
  const confirmed = window.confirm(
    `确定移除「${model.name}」？\n\n` +
    `将删除 ${formatSize(model.sizeBytes)} 的模型文件，下次使用需重新下载。`
  );
  if (!confirmed) return;
  try {
    await removeModel(model.id);
    message.success(`已移除 ${model.name}`);
    await refreshModels();
  } catch (err) {
    message.error("移除失败: " + (err.message || err));
  }
}
async function onSetDefaultModel(model) {
  try {
    await setDefaultModel(model.id);
    defaultModelID.value = model.id;
    message.success(`已将 ${model.name} 设为默认模型`);
  } catch (err) {
    message.error("设置失败: " + (err.message || err));
  }
}


async function openModelDir() {
  try {
    const res = await openModelFolder();
    if (res.status !== "ok") {
    }
  } catch (err) {
    message.error("调用后端失败: " + err);
  }
}

// Utils
function fmtTimeFull(sec) {
  if (sec == null || sec < 0) return "00:00:00.000";
  const h = Math.floor(sec / 3600)
    .toString()
    .padStart(2, "0");
  const m = Math.floor((sec % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, "0");
  const ms = Math.floor((sec % 1) * 1000)
    .toString()
    .padStart(3, "0");
  return `${h}:${m}:${s}.${ms}`;
}

function fmtTime(sec) {
  if (sec == null || sec < 0) return "00:00:00";
  const h = Math.floor(sec / 3600)
    .toString()
    .padStart(2, "0");
  const m = Math.floor((sec % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function fmtTimeSrt(sec) {
  if (sec == null || sec < 0) return "00:00:00,000";
  const h = Math.floor(sec / 3600)
    .toString()
    .padStart(2, "0");
  const m = Math.floor((sec % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, "0");
  const ms = Math.floor((sec % 1) * 1000)
    .toString()
    .padStart(3, "0");
  return `${h}:${m}:${s},${ms}`;
}
</script>

<style scoped>
/* ─── Variables ─── */
:root {
  --bg-app: #121212;
  --bg-panel: #1a1a1a;
  --bg-card: #222222;
  --border-color: #2a2a2a;
  --text-main: #f0f0f0;
  --text-muted: #888888;
  --text-active: #1fbc5b; /* 主题绿 */
  --btn-green: #1fbc5b;
  --btn-green-hover: #1db155;
}

/* ─── Layout ─── */
.app-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background-color: var(--bg-app, #121212);
  color: var(--text-main, #f0f0f0);
  font-family:
    system-ui,
    -apple-system,
    sans-serif;
  overflow: hidden;
}

.icon-btn {
  background: transparent;
  border: none;
  color: var(--text-muted, #888);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.2s;
  padding: 4px;
}
.icon-btn:hover {
  color: var(--text-main, #f0f0f0);
}

.app-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ─── Sidebar ─── */
.sidebar {
  width: 320px;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-app, #121212);
  border-right: 1px solid var(--border-color, #2a2a2a);
  padding: 20px;
  gap: 20px;
}

.file-zone {
  flex-shrink: 0;
}

/* ─── Engine Selector ─── */
.engine-selector {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex-shrink: 0;
}
.engine-label {
  font-size: 0.75rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.engine-select {
  background: #111;
  color: #e0e0e0;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  padding: 8px 12px;
  font-family: inherit;
  font-size: 0.85rem;
  outline: none;
  cursor: pointer;
  transition: border-color 0.2s;
  width: 100%;
}
.engine-select:focus {
  border-color: #1fbc5b;
}
.engine-select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.engine-warning {
  font-size: 0.7rem;
  color: #e8a838;
  margin-top: 2px;
}

.regenerate-btn {
  margin-top: 8px;
  align-self: flex-start;
}
/* 字幕时间偏移滑块 */
.offset-control {
  margin-top: 12px;
}
.offset-control .engine-label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.offset-value {
  font-weight: 600;
  color: #1fbc5b;
  font-variant-numeric: tabular-nums;
}
.offset-hint {
  font-size: 0.7rem;
  color: #888;
  font-weight: 400;
}
.offset-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.offset-btn {
  background: #333;
  color: #fff;
  border: 1px solid #444;
  border-radius: 4px;
  padding: 4px 10px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: background 0.2s;
}
.offset-btn:hover:not(:disabled) {
  background: #444;
}
.offset-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.offset-reset {
  margin-left: auto;
  color: #888;
}
.offset-slider {
  flex: 1;
  height: 4px;
  background: #333;
  border-radius: 2px;
  outline: none;
  -webkit-appearance: none;
  appearance: none;
  cursor: pointer;
}
.offset-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  background: #1fbc5b;
  border-radius: 50%;
  cursor: pointer;
}
.offset-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  background: #1fbc5b;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}
.offset-slider:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.offset-hint-row {
  font-size: 0.7rem;
  color: #666;
  margin-top: 4px;
  font-style: italic;
}

.upload-container :deep(.n-upload-dragger) {
  background-color: var(--bg-card, #222) !important;
  border: 1px dashed #444 !important;
  border-radius: 8px !important;
  padding: 30px 20px !important;
  transition: all 0.3s;
}
.upload-container :deep(.n-upload-dragger:hover),
.drag-hover {
  border-color: #666 !important;
  background-color: #282828 !important;
}

.upload-icon {
  color: #666;
  margin-bottom: 12px;
}
.upload-text {
  font-size: 0.9rem;
  color: #ccc;
  margin-bottom: 4px;
}
.upload-subtext {
  font-size: 0.75rem;
  color: #666;
}

/* File Pill */
.file-pill {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px;
  background-color: var(--bg-card, #222);
  border-radius: 8px;
  border: 1px solid var(--border-color, #2a2a2a);
}
.file-info {
  display: flex;
  align-items: center;
  gap: 12px;
  overflow: hidden;
}
.file-icon {
  color: var(--text-active, #1fbc5b);
  flex-shrink: 0;
}
.file-details {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.file-name {
  font-size: 0.9rem;
  color: #eee;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.file-meta {
  font-size: 0.75rem;
  color: #888;
  margin-top: 2px;
}
.replace-btn {
  width: 100%;
}

.llm-correct-row {
  padding: 0 0 8px;
}
.llm-correct-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
  font-weight: 500;
}
.list-count {
  font-size: 0.8rem;
  color: #666;
}

.subtitle-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-right: 4px;
}
.subtitle-list::-webkit-scrollbar {
  width: 4px;
}
.subtitle-list::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 2px;
}

.list-item {
  display: flex;
  gap: 12px;
  padding: 12px;
  background-color: var(--bg-card, #222);
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
  border: 1px solid transparent;
}
.list-item:hover {
  background-color: #282828;
}
.list-item.active {
  background-color: rgba(31, 188, 91, 0.1);
  border-color: rgba(31, 188, 91, 0.3);
}
.item-time {
  font-family: monospace;
  font-size: 0.75rem;
  color: #888;
  flex-shrink: 0;
  padding-top: 2px;
}
.item-text {
  font-size: 0.85rem;
  line-height: 1.4;
  color: #ddd;
}
.list-item.active .item-text {
  color: var(--text-active, #1fbc5b);
}
.item-delete {
  display: none;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  padding: 2px;
  border-radius: 4px;
  flex-shrink: 0;
  margin-left: auto;
  align-self: flex-start;
  transition: color 0.15s, background 0.15s;
}
.list-item:hover .item-delete {
  display: flex;
}
.item-delete:hover {
  color: #ff4757;
  background: rgba(255, 71, 87, 0.1);
}

.list-empty {
  text-align: center;
  color: #666;
  font-size: 0.85rem;
  margin-top: 20px;
}

/* ─── Main Content ─── */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
  gap: 20px;
  background-color: var(--bg-app, #121212);
  overflow-y: auto;
}

.media-preview-area {
  background-color: var(--bg-panel, #1a1a1a);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  min-height: 0;
  flex: 1;
}

.media-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  background-color: #000;
  min-height: 0;
}

.media-element {
  width: 100%;
  height: 100%;
  object-fit: contain;
  min-height: 0;
}

.media-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  color: #444;
}
.placeholder-icon {
  margin-bottom: 16px;
}
.placeholder-text {
  font-size: 0.9rem;
}

.processing-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
}
.processing-text {
  margin-top: 16px;
  font-size: 0.9rem;
  color: #eee;
}

/* ─── Overlays ─── */
.subtitle-overlay {
  position: absolute;
  bottom: 24px;
  left: 0;
  right: 0;
  text-align: center;
  pointer-events: none;
  z-index: 10;
  padding: 0 40px;
}
.subtitle-overlay.is-audio {
  bottom: 50%;
  transform: translateY(50%);
}
.subtitle-text {
  display: inline-block;
  background-color: rgba(0, 0, 0, 0.75);
  color: #fff;
  padding: 6px 16px;
  border-radius: 6px;
  font-size: 1.15rem;
  text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
  word-wrap: break-word;
  line-height: 1.5;
}

/* ─── Editor Area ─── */
.editor-area {
  background-color: var(--bg-panel, #1a1a1a);
  border-radius: 10px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.editor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.editor-title {
  font-size: 0.95rem;
  font-weight: 500;
}
.editor-count {
  font-size: 0.8rem;
  color: #666;
}

.editor-time {
  font-size: 0.8rem;
  color: #888;
  margin-bottom: 12px;
  font-family: monospace;
}
.editor-time span {
  color: #aaa;
}

.editor-input {
  background-color: var(--bg-app, #121212);
  border: 1px solid var(--border-color, #2a2a2a);
  border-radius: 6px;
  padding: 12px;
  color: #eee;
  font-size: 0.95rem;
  line-height: 1.5;
  resize: none;
  height: 100px;
  margin-bottom: 16px;
  font-family: inherit;
}
.editor-input:focus {
  outline: none;
  border-color: #444;
}

.editor-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.action-group-left,
.action-group-right {
  display: flex;
  gap: 12px;
}

.format-select {
  background: #111;
  color: #e0e0e0;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  padding: 0 12px;
  font-family: inherit;
  font-size: 0.85rem;
  outline: none;
  cursor: pointer;
  height: 33px; /* 匹配按钮高度 */
}
.format-select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.log-level-select {
  background: #111;
  color: #e0e0e0;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  padding: 4px 12px;
  font-family: inherit;
  font-size: 0.85rem;
  outline: none;
  cursor: pointer;
  height: 34px;
}
.log-level-select:focus {
  border-color: #1fbc5b;
}

.divider {
  height: 1px;
  background: #333;
  margin: 16px 0;
}

.setting-label-group {
  font-size: 13px;
  font-weight: 600;
  color: #888;
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.setting-status {
  font-size: 12px;
  color: #1fbc5b;
  margin-left: 8px;
}

.about-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #7ab8ff;
  text-decoration: none;
  font-size: 13px;
  word-break: break-all;
}
.about-link:hover {
  text-decoration: underline;
}

.setting-row {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}
.primary-btn,
.secondary-btn {
  padding: 8px 16px;
  border-radius: 6px;
  font-size: 0.85rem;
  cursor: pointer;
  border: none;
  transition: all 0.2s;
}
.primary-btn {
  background-color: var(--btn-green, #1fbc5b);
  color: #000;
  font-weight: 500;
}
.primary-btn:hover:not(:disabled) {
  background-color: var(--btn-green-hover, #1db155);
}
.primary-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.secondary-btn {
  background-color: #2a2a2a;
  color: #ccc;
}
.secondary-btn:hover:not(:disabled) {
  background-color: #333;
}
.secondary-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ─── Footer Controls ─── */
.player-controls {
  height: 64px;
  background-color: var(--bg-app, #121212);
  border-top: 1px solid var(--border-color, #2a2a2a);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 48px; /* Increased padding from 30px to 48px for a more spacious, elegant look */
  flex-shrink: 0;
  position: relative;
}

/* Global Progress Bar */
.progress-container-global {
  position: absolute;
  top: -2px;
  left: 0;
  right: 0;
  height: 4px;
  background-color: rgba(255, 255, 255, 0.1);
  cursor: pointer;
  z-index: 20;
  transition: all 0.2s;
}
.progress-container-global:hover {
  height: 8px;
  top: -6px;
}
.progress-bar-global {
  height: 100%;
  background-color: var(--text-active, #1fbc5b);
  transition: width 0.1s linear;
  box-shadow: 0 0 8px rgba(31, 188, 91, 0.5);
  position: relative;
}

.progress-thumb {
  position: absolute;
  right: -6px;
  top: 50%;
  transform: translateY(-50%) scale(0);
  width: 12px;
  height: 12px;
  background-color: #fff;
  border-radius: 50%;
  box-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
  transition: transform 0.2s;
  pointer-events: none;
}
.progress-container-global:hover .progress-thumb,
.progress-container-global.dragging .progress-thumb {
  transform: translateY(-50%) scale(1);
}

.progress-tooltip {
  position: absolute;
  bottom: 12px;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.85);
  color: #fff;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.8rem;
  pointer-events: none;
  white-space: nowrap;
}

.control-time {
  font-family: monospace;
  font-size: 0.85rem;
  color: #888;
  flex: 1; /* Allow left side to flex and balance the right side */
  display: flex;
  justify-content: flex-start;
  align-items: center;
}

.control-center {
  display: flex;
  align-items: center;
  gap: 24px;
}

.play-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: var(--btn-green, #1fbc5b);
  border: none;
  color: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.1s;
}
.play-btn:active {
  transform: scale(0.95);
}
.play-icon {
  margin-left: 2px; /* Visual center adjustment for play icon */
}

.control-volume {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
  justify-content: flex-end;
}
.vol-icon {
  width: 24px;
  color: #888;
}
.volume-slider {
  width: 80px;
}
.divider {
  width: 1px;
  height: 16px;
  background-color: #333;
  margin: 0 4px;
}
.volume-slider :deep(.n-slider-rail) {
  background-color: #333;
}
.volume-slider :deep(.n-slider-fill) {
  background-color: #888;
}
.volume-slider :deep(.n-slider-thumb) {
  display: none; /* Hide thumb to match typical sleek video players unless hovered */
}
.volume-slider:hover :deep(.n-slider-thumb) {
  display: block;
  width: 12px;
  height: 12px;
  box-shadow: none;
  border: 2px solid #fff;
  background: #fff;
}

/* ─── Settings Modal ─── */
.custom-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(2px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.custom-modal {
  width: 680px;
  background: #1c1c1c;
  border: 1px solid #333;
  border-radius: 10px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  display: flex;
  flex-direction: column;
}
.custom-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid #2a2a2a;
}
.custom-modal-header h3 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 500;
  color: #eee;
}
.custom-modal-body {
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.custom-modal-body.nav-layout {
  flex-direction: row;
  gap: 0;
  min-height: 320px;
}

/* ─── Settings Sidebar ─── */
.settings-sidebar {
  width: 130px;
  flex-shrink: 0;
  background: #161616;
  border-right: 1px solid #2a2a2a;
  border-radius: 10px 0 0 10px;
  padding: 12px 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.settings-sidebar {
  position: relative;
}
/* 滑动高亮块: 弹性跟随激活项 (借鉴 beUI morphing-tabs 的共享布局动效) */
.settings-nav-glider {
  position: absolute;
  top: 0; /* 必须显式归零: 不设时元素停在静态位置 (padding-top 之后),
             再叠加 translateY(offsetTop) 会双重偏移一个 padding-top */
  left: 0;
  width: 100%;
  background: rgba(31, 188, 91, 0.08);
  border-right: 2px solid #1fbc5b;
  border-radius: 0 8px 8px 0;
  pointer-events: none;
  transition:
    transform 0.32s cubic-bezier(0.34, 1.4, 0.64, 1),
    height 0.32s cubic-bezier(0.34, 1.4, 0.64, 1),
    opacity 0.18s ease;
  z-index: 0;
}
.settings-nav-item {
  position: relative;
  z-index: 1; /* 文字浮在 glider 上 */
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  cursor: pointer;
  font-size: 0.9rem;
  color: #999;
  transition: all 0.15s;
  user-select: none;
}
.settings-nav-item:hover {
  color: #ddd;
  background: rgba(255, 255, 255, 0.04);
}
.settings-nav-item.active {
  color: #1fbc5b;
  /* 背景与右缘条由 settings-nav-glider 弹性滑动呈现 */
}

.nav-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
}
.nav-label {
  line-height: 1;
}

/* ─── Settings Panel ─── */
.settings-panel {
  flex: 1;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  overflow-y: auto;
  max-height: 400px;
}

.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.setting-item.col {
  flex-direction: column;
  align-items: flex-start;
  gap: 12px;
}
.setting-label {
  font-size: 0.95rem;
  color: #ddd;
}
.path-picker {
  display: flex;
  width: 100%;
  gap: 10px;
  align-items: center;
}
.small-btn {
  padding: 0 14px;
  height: 34px;
  font-size: 0.85rem;
  white-space: nowrap;
  flex-shrink: 0;
}
.copyright-info {
  margin-top: 24px;
  text-align: center;
  font-size: 0.75rem;
  color: var(--text-muted, #666);
  border-top: 1px solid var(--border-color, #2a2a2a);
  padding-top: 16px;
  letter-spacing: 0.5px;
}

/* ─── Model Status ─── */
.model-status-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.model-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.model-status-dot.active  { background: #1fbc5b; box-shadow: 0 0 6px rgba(31,188,91,0.5); }
.model-status-dot.inactive { background: #666; }
.model-status-dot.dot-green { background: #1fbc5b; box-shadow: 0 0 6px rgba(31,188,91,0.5); }
.model-status-dot.dot-yellow { background: #f0ad4e; box-shadow: 0 0 6px rgba(240,173,78,0.5); }
.model-status-dot.dot-red    { background: #e74c3c; box-shadow: 0 0 6px rgba(231,76,60,0.5); }
.model-status-dot.dot-gray   { background: #666; }
.model-status-text {
  font-size: 0.85rem;
  color: #aaa;
}

.api-key-row {
  display: flex;
  gap: 6px;
  align-items: center;
}
.api-key-row .n-input {
  flex: 1;
}
.api-key-row .icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 6px;
  color: #888;
  cursor: pointer;
  padding: 6px;
  transition: color 0.15s, border-color 0.15s;
  flex-shrink: 0;
}
.api-key-row .icon-btn:hover {
  color: #1fbc5b;
  border-color: #1fbc5b;
}

.llm-status {
  font-size: 0.85rem;
  color: #aaa;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ─── Model Manager ─── */
.model-hint {
  font-size: 0.8rem;
  color: #777;
  line-height: 1.5;
}

.model-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 12px;
  margin-bottom: 8px;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  background: #1a1a1a;
}

.model-row-main {
  flex: 1;
  min-width: 0;
}

.model-row-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-name {
  font-size: 0.9rem;
  color: #e0e0e0;
}

.model-tag {
  font-size: 0.7rem;
  color: #1fbc5b;
  border: 1px solid rgba(31, 188, 91, 0.4);
  border-radius: 4px;
  padding: 0 4px;
}

.model-row-desc {
  font-size: 0.8rem;
  color: #999;
  margin-top: 4px;
}

.model-row-meta {
  font-size: 0.75rem;
  color: #666;
  margin-top: 4px;
}

.model-row-actions {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.danger-btn:hover {
  color: #e74c3c;
  border-color: #e74c3c;
}

.model-progress {
  margin-top: 8px;
}

.model-progress-bar {
  height: 4px;
  border-radius: 2px;
  background: #2a2a2a;
  overflow: hidden;
}

.model-progress-fill {
  height: 100%;
  background: #1fbc5b;
  transition: width 0.3s;
}

.model-progress-text {
  font-size: 0.75rem;
  color: #999;
  margin-top: 4px;
}

/* 下载进度悬浮条: 不遮挡界面 */
.download-toast {
  position: fixed;
  right: 20px;
  bottom: 20px;
  width: 280px;
  padding: 12px 14px;
  border-radius: 10px;
  border: 1px solid #2a2a2a;
  background: rgba(26, 26, 26, 0.96);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
  z-index: 9999;
}

.download-toast-title {
  font-size: 0.85rem;
  color: #e0e0e0;
  margin-bottom: 8px;
}

.download-toast-text {
  font-size: 0.75rem;
  color: #999;
  margin-top: 6px;
}
</style>
