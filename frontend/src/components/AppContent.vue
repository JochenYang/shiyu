<template>
  <div class="app-shell">
    <!-- Background Decor -->
    <div class="glow-orb orb-1"></div>
    <div class="glow-orb orb-2"></div>

    <!-- Header -->
    <header class="app-header">
      <div class="brand">
        <h1 class="brand-title">时语 <span>Shiyu</span></h1>
        <p class="brand-sub">专业本地 AI 字幕生成器</p>
      </div>
      <div class="status-badge" :class="{ online: backendOnline }">
        <div class="pulse-dot"></div>
        {{ backendOnline ? "服务在线" : "服务离线" }}
      </div>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <!-- Step 1: Upload -->
      <section>
        <n-upload
          :max="1"
          accept="audio/*,video/*"
          :show-file-list="false"
          :custom-request="handleUpload"
        >
          <n-upload-dragger class="premium-dragger" :class="{ 'drag-hover': isDraggingOver }">
            <div class="dragger-icon-wrapper">
              <UploadCloud :size="32" />
            </div>
            <div class="upload-content">
              <h3>拖拽或点击上传音频 / 视频文件</h3>
              <p>支持 MP3, WAV, MP4, MKV 等格式</p>
            </div>
          </n-upload-dragger>
        </n-upload>

        <div v-if="selectedFile" class="file-glass-card">
          <div class="file-meta">
            <FileText :size="18" class="meta-icon" />
            <div class="meta-text">
              <span class="filename">{{ selectedFile.name }}</span>
              <span class="filesize">{{ formatFileSize(selectedFile.size) }}</span>
            </div>
          </div>
          <n-button quaternary circle size="small" @click="clearFile">
            <template #icon><X :size="14" /></template>
          </n-button>
        </div>
      </section>

      <!-- Step 2: Settings & Action -->
      <transition name="slide-fade">
        <div v-if="selectedFile" class="action-grid">
          <div class="settings-strip">
            <div class="config-item">
              <span class="config-label">输出格式</span>
              <n-select v-model:value="format" :options="formatOptions" size="medium" class="premium-select" />
            </div>
            <div class="config-item">
              <span class="config-label">语言</span>
              <n-select v-model:value="language" :options="languageOptions" size="medium" class="premium-select" />
            </div>
            <div class="config-item center">
              <span class="config-label">逆文本规范化</span>
              <n-switch v-model:value="useItn" size="medium" />
            </div>
          </div>
          
          <n-button
            type="primary"
            block
            strong
            size="large"
            :loading="isProcessing"
            :disabled="!backendOnline"
            @click="startTranscribe"
            class="generate-btn"
          >
            <template #icon v-if="!isProcessing">
              <Zap :size="18" />
            </template>
            {{ isProcessing ? "识别中..." : "开始生成字幕" }}
          </n-button>
        </div>
      </transition>

      <!-- Step 3: Wave Player -->
      <transition name="scale-in">
        <n-card v-if="audioUrl" class="section-card player-section" size="small">
          <template #header>
            <div class="card-header">
              <Music :size="16" />
              <span>音频预览</span>
            </div>
          </template>
          <WavePlayer
            :src="audioUrl"
            :segments="parsedSegments"
            @timeupdate="onTimeUpdate"
          />
        </n-card>
      </transition>

      <!-- Step 4: Subtitle Preview -->
      <transition name="scale-in">
        <n-card v-if="resultContent" class="section-card preview-section" size="small">
          <SubtitlePreview
            :segments="parsedSegments"
            :format="format"
            :current-time="playbackTime"
            @download="downloadResult"
            @copy="copyResult"
            @seek="seekPlayer"
          />
        </n-card>
      </transition>

      <!-- Step 5: Raw Output (Collapsible) -->
      <n-card v-if="resultContent" class="section-card raw-section" size="small">
        <n-collapse arrow-placement="right">
          <n-collapse-item name="raw">
            <template #header>
              <div class="collapse-trigger">
                <Code :size="14" />
                <span>原始 {{ format.toUpperCase() }} 内容</span>
              </div>
            </template>
            <div class="code-viewer">
              <n-code :code="resultContent" language="txt" word-wrap />
            </div>
          </n-collapse-item>
        </n-collapse>
      </n-card>
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <div class="footer-line"></div>
      <p>©2026 时语 Shiyu · Jochen · 完全本地运行</p>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from "vue";
import {
  NCard, NButton, NTag,
  NUpload, NUploadDragger, NSelect, NSwitch,
  NIcon, NCollapse, NCollapseItem, NCode,
  useMessage,
} from "naive-ui";
import {
  UploadCloud, FileText, X, Zap, Music, Code,
} from "lucide-vue-next";
import { listen } from "@tauri-apps/api/event";
import { readBinaryFile } from "@tauri-apps/api/fs";
import WavePlayer from "./WavePlayer.vue";
import SubtitlePreview from "./SubtitlePreview.vue";
import { healthCheck, transcribeFile, transcribeJson, downloadText } from "../api.js";

const message = useMessage();

const formatOptions = [
  { label: "SRT（推荐）", value: "srt" },
  { label: "LRC（歌词）", value: "lrc" },
  { label: "ASS（高级）", value: "ass" },
];
const languageOptions = [
  { label: "自动检测", value: "auto" },
  { label: "中文", value: "zh" },
  { label: "English", value: "en" },
  { label: "日本語", value: "ja" },
  { label: "한국어", value: "ko" },
  { label: "粤语", value: "yue" },
];

const selectedFile = ref(null);
const audioUrl = ref(null);
const format = ref("srt");
const language = ref("auto");
const useItn = ref(true);
const isProcessing = ref(false);
const resultContent = ref("");
const resultSegments = ref([]);
const backendOnline = ref(false);
const playbackTime = ref(0);

const isDraggingOver = ref(false);
let pollTimer = null;
let unlistenDrop = null;
let unlistenHover = null;
let unlistenCancel = null;

const MEDIA_EXTENSIONS = /\.(mp4|mkv|avi|mov|wmv|flv|webm|mp3|wav|m4a|flac|ogg|aac)$/i;

onMounted(async () => {
  pollTimer = setInterval(async () => {
    try {
      await healthCheck();
      backendOnline.value = true;
    } catch {
      backendOnline.value = false;
    }
  }, 2000);

  // Tauri native file-drop listeners
  try {
    unlistenDrop = await listen("tauri://file-drop", async (event) => {
      isDraggingOver.value = false;
      const paths = event.payload;
      if (!paths || paths.length === 0) return;
      const mediaPath = paths.find((p) => MEDIA_EXTENSIONS.test(p));
      if (!mediaPath) {
        message.warning("不支持的文件格式，请拖入音频/视频文件");
        return;
      }
      try {
        const bytes = await readBinaryFile(mediaPath);
        const blob = new Blob([bytes]);
        const fileName = mediaPath.split(/[\\/]/).pop() || "file";
        const file = new File([blob], fileName, { type: blob.type || "video/mp4" });
        selectedFile.value = file;
        if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
        audioUrl.value = URL.createObjectURL(blob);
        resultContent.value = "";
        resultSegments.value = [];
        message.info(`已加载: ${fileName}`);
      } catch (err) {
        message.error("读取文件失败: " + err);
      }
    });

    unlistenHover = await listen("tauri://file-drop-hover", () => {
      isDraggingOver.value = true;
    });

    unlistenCancel = await listen("tauri://file-drop-cancelled", () => {
      isDraggingOver.value = false;
    });
  } catch {
    // Running in browser (dev mode without Tauri), skip native drop
  }
});

onBeforeUnmount(() => {
  clearInterval(pollTimer);
  if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
  unlistenDrop?.();
  unlistenHover?.();
  unlistenCancel?.();
});

function handleUpload({ file }) {
  const f = file.file;
  selectedFile.value = f;
  if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
  audioUrl.value = URL.createObjectURL(f);
  resultContent.value = "";
  resultSegments.value = [];
}

function clearFile() {
  selectedFile.value = null;
  if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
  audioUrl.value = null;
  resultContent.value = "";
  resultSegments.value = [];
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

async function startTranscribe() {
  if (!selectedFile.value) return;
  isProcessing.value = true;
  resultContent.value = "";
  resultSegments.value = [];

  try {
    const [content, json] = await Promise.all([
      transcribeFile(selectedFile.value, {
        language: language.value,
        output_format: format.value,
      }),
      transcribeJson(selectedFile.value, language.value).catch(() => null),
    ]);

    resultContent.value = content;
    if (json?.segments) {
      resultSegments.value = json.segments;
    }
    message.success("字幕生成完成");
  } catch (err) {
    message.error(err.message || "处理失败，请检查后端服务");
  } finally {
    isProcessing.value = false;
  }
}

const parsedSegments = computed(() => {
  return resultSegments.value.map((s) => ({
    start: s.start,
    end: s.end,
    text: s.text,
  }));
});

function onTimeUpdate(t) {
  playbackTime.value = t;
}

function seekPlayer(time) {
  playbackTime.value = time;
}

function downloadResult() {
  if (!resultContent.value || !selectedFile.value) return;
  const baseName = selectedFile.value.name.replace(/\.[^.]+$/, "") || "subtitle";
  downloadText(resultContent.value, `${baseName}.${format.value}`);
}

async function copyResult() {
  try {
    await navigator.clipboard.writeText(resultContent.value);
    message.success("已复制到剪贴板");
  } catch {
    message.error("复制失败");
  }
}
</script>

<style scoped>
.app-shell {
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 24px;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  position: relative;
  z-index: 1;
}

/* Background Glows */
.glow-orb {
  position: fixed;
  border-radius: 50%;
  filter: blur(80px);
  z-index: -1;
  opacity: 0.15;
}
.orb-1 {
  width: 400px;
  height: 400px;
  background: #6366f1;
  top: -100px;
  right: -100px;
}
.orb-2 {
  width: 300px;
  height: 300px;
  background: #4f46e5;
  bottom: 50px;
  left: -50px;
}

/* Header */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 48px;
}

.brand-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -1px;
  margin: 0;
  color: #fff;
}
.brand-title span {
  font-weight: 300;
  opacity: 0.4;
  font-size: 1.4rem;
  margin-left: 8px;
}

.brand-sub {
  margin: 4px 0 0;
  color: #94a3b8;
  font-size: 0.9rem;
  font-weight: 400;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  background: rgba(30, 41, 59, 0.5);
  border: 1px solid rgba(51, 65, 85, 0.5);
  border-radius: 100px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #94a3b8;
  transition: all 0.3s;
}
.status-badge.online {
  color: #818cf8;
  border-color: rgba(129, 140, 248, 0.3);
}

.pulse-dot {
  width: 6px;
  height: 6px;
  background: currentColor;
  border-radius: 50%;
  box-shadow: 0 0 0 0 rgba(129, 140, 248, 0.4);
}
.online .pulse-dot {
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(129, 140, 248, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(129, 140, 248, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(129, 140, 248, 0); }
}

/* Main */
.app-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Dragger */
.premium-dragger {
  background: rgba(15, 23, 42, 0.6) !important;
  border: 2px dashed rgba(51, 65, 85, 0.5) !important;
  border-radius: 20px !important;
  padding: 48px 0 !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.premium-dragger:hover {
  background: rgba(30, 41, 59, 0.4) !important;
  border-color: #6366f1 !important;
  transform: translateY(-2px);
}
.premium-dragger.drag-hover {
  background: rgba(99, 102, 241, 0.08) !important;
  border-color: #818cf8 !important;
  border-style: solid !important;
  transform: scale(1.01);
  box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
}

.dragger-icon-wrapper {
  width: 64px;
  height: 64px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 16px;
  color: #818cf8;
}

.upload-content h3 {
  margin: 0;
  font-size: 1.1rem;
  color: #f1f5f9;
}
.upload-content p {
  margin: 8px 0 0;
  font-size: 0.85rem;
  color: #64748b;
}

/* File Card */
.file-glass-card {
  margin-top: 16px;
  padding: 12px 16px;
  background: rgba(15, 23, 42, 0.4);
  border: 1px solid rgba(51, 65, 85, 0.5);
  backdrop-filter: blur(12px);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.file-meta {
  display: flex;
  align-items: center;
  gap: 12px;
}
.meta-icon {
  color: #818cf8;
}
.meta-text {
  display: flex;
  flex-direction: column;
}
.filename {
  font-size: 0.9rem;
  font-weight: 500;
  color: #f1f5f9;
}
.filesize {
  font-size: 0.75rem;
  color: #64748b;
}

/* Action Area */
.action-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.settings-strip {
  display: grid;
  grid-template-columns: 1fr 1fr auto;
  gap: 12px;
  padding: 16px;
  background: rgba(15, 23, 42, 0.4);
  border-radius: 16px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.config-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #475569;
}

.premium-select :deep(.n-base-selection) {
  background: rgba(2, 6, 23, 0.3);
}

.generate-btn {
  height: 52px;
  font-size: 1rem;
  box-shadow: 0 4px 20px -4px rgba(99, 102, 241, 0.5);
}

/* Cards & Sections */
.section-card {
  border: none;
  background: rgba(15, 23, 42, 0.4);
  backdrop-filter: blur(20px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9rem;
  font-weight: 600;
  color: #94a3b8;
}

.collapse-trigger {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  font-weight: 600;
  color: #64748b;
}

.code-viewer {
  margin-top: 8px;
  padding: 12px;
  background: #020617;
  border-radius: 8px;
}

/* Transitions */
.slide-fade-enter-active { transition: all 0.4s ease-out; }
.slide-fade-enter-from { opacity: 0; transform: translateY(10px); }

.scale-in-enter-active { transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1); }
.scale-in-enter-from { opacity: 0; transform: scale(0.98); }

/* Footer */
.app-footer {
  margin-top: 64px;
  padding-bottom: 24px;
  text-align: center;
}
.footer-line {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(51, 65, 85, 0.5), transparent);
  margin-bottom: 24px;
}
.app-footer p {
  font-size: 0.8rem;
  color: #334155;
  font-weight: 500;
}
</style>
