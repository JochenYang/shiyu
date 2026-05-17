<template>
  <div class="app-shell">
    <!-- Ambient Glow Effects -->
    <div class="ambient-glow glow-warm"></div>
    <div class="ambient-glow glow-cool"></div>

    <!-- Header -->
    <header class="app-header">
      <div class="brand">
        <div class="brand-mark">
          <div class="brand-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" x2="12" y1="19" y2="22"/>
            </svg>
          </div>
          <div class="brand-text">
            <h1>时语 <span class="brand-en">Shiyu</span></h1>
            <p class="brand-tagline">本地 AI 字幕引擎</p>
          </div>
        </div>
      </div>
      <div class="status-chip" :class="{ online: backendOnline }">
        <div class="status-dot"></div>
        <span>{{ backendOnline ? "引擎就绪" : "引擎离线" }}</span>
      </div>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <!-- Upload Zone -->
      <section class="upload-zone">
        <n-upload
          :max="1"
          accept="audio/*,video/*"
          :show-file-list="false"
          :custom-request="handleUpload"
        >
          <n-upload-dragger class="drop-surface" :class="{ 'drag-hover': isDraggingOver }">
            <div class="drop-inner">
              <div class="drop-icon-ring">
                <UploadCloud :size="24" />
              </div>
              <div class="drop-copy">
                <h3>拖入或点击选择文件</h3>
                <p>支持 MP3、WAV、MP4、MKV 等常见音视频格式</p>
              </div>
            </div>
          </n-upload-dragger>
        </n-upload>

        <!-- File Indicator -->
        <transition name="slide-up">
          <div v-if="selectedFile" class="file-pill">
            <div class="file-info">
              <FileText :size="16" class="file-icon" />
              <span class="file-name">{{ selectedFile.name }}</span>
              <span class="file-size">{{ formatFileSize(selectedFile.size) }}</span>
            </div>
            <button class="file-clear" @click="clearFile" aria-label="移除文件">
              <X :size="14" />
            </button>
          </div>
        </transition>
      </section>

      <!-- Config + Action -->
      <transition name="slide-up">
        <section v-if="selectedFile" class="config-section">
          <div class="config-row">
            <div class="config-field">
              <label class="field-label">输出格式</label>
              <n-select v-model:value="format" :options="formatOptions" size="medium" />
            </div>
            <div class="config-field">
              <label class="field-label">识别语言</label>
              <n-select v-model:value="language" :options="languageOptions" size="medium" />
            </div>
            <div class="config-field config-toggle">
              <label class="field-label">逆文本规范化</label>
              <n-switch v-model:value="useItn" size="medium" />
            </div>
          </div>

          <button
            class="action-btn"
            :class="{ loading: isProcessing }"
            :disabled="!backendOnline || isProcessing"
            @click="startTranscribe"
          >
            <span class="action-btn-bg"></span>
            <span class="action-btn-content">
              <Zap v-if="!isProcessing" :size="18" />
              <span class="spinner" v-else></span>
              {{ isProcessing ? "识别中…" : "开始生成字幕" }}
            </span>
          </button>
        </section>
      </transition>

      <!-- Audio Player -->
      <transition name="scale-in">
        <section v-if="audioUrl" class="player-section">
          <div class="section-label">
            <Music :size="14" />
            <span>音频预览</span>
          </div>
          <WavePlayer
            :src="audioUrl"
            :segments="parsedSegments"
            @timeupdate="onTimeUpdate"
          />
        </section>
      </transition>

      <!-- Subtitle Preview -->
      <transition name="scale-in">
        <section v-if="resultContent" class="preview-section">
          <SubtitlePreview
            :segments="parsedSegments"
            :format="format"
            :current-time="playbackTime"
            @download="downloadResult"
            @copy="copyResult"
            @seek="seekPlayer"
          />
        </section>
      </transition>

      <!-- Raw Output -->
      <transition name="scale-in">
        <section v-if="resultContent" class="raw-section">
          <n-collapse arrow-placement="right">
            <n-collapse-item name="raw">
              <template #header>
                <div class="collapse-label">
                  <Code :size="14" />
                  <span>原始 {{ format.toUpperCase() }} 内容</span>
                </div>
              </template>
              <div class="raw-viewer">
                <n-code :code="resultContent" language="txt" word-wrap />
              </div>
            </n-collapse-item>
          </n-collapse>
        </section>
      </transition>
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      <div class="footer-divider"></div>
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
import { convertFileSrc } from "@tauri-apps/api/tauri";
import WavePlayer from "./WavePlayer.vue";
import SubtitlePreview from "./SubtitlePreview.vue";
import { healthCheck, transcribeFile, transcribeJson, transcribeLocal, downloadText } from "../api.js";

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
const localFilePath = ref(null);  // Tauri drag-drop: store path for direct backend access
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
    unlistenDrop = await listen("tauri://file-drop", (event) => {
      isDraggingOver.value = false;
      const paths = event.payload;
      if (!paths || paths.length === 0) return;
      const mediaPath = paths.find((p) => MEDIA_EXTENSIONS.test(p));
      if (!mediaPath) {
        message.warning("不支持的文件格式，请拖入音频/视频文件");
        return;
      }
      // Instant: no readBinaryFile, use convertFileSrc for audio preview
      const fileName = mediaPath.split(/[\\/]/).pop() || "file";
      localFilePath.value = mediaPath;
      selectedFile.value = { name: fileName, size: 0 };  // Lightweight placeholder
      if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
      audioUrl.value = convertFileSrc(mediaPath);
      resultContent.value = "";
      resultSegments.value = [];
      message.info(`已加载: ${fileName}`);
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
  localFilePath.value = null;
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
    if (localFilePath.value) {
      // Tauri drag-drop: use path-based API (single request, no upload)
      const result = await transcribeLocal(localFilePath.value, {
        language: language.value,
        output_format: format.value,
      });
      resultContent.value = result.content;
      if (result.segments) {
        resultSegments.value = result.segments;
      }
    } else {
      // Browser click-upload: use FormData upload (two requests)
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
/* ─── Layout Shell ─── */
.app-shell {
  max-width: 760px;
  margin: 0 auto;
  padding: 36px 28px 24px;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  position: relative;
  z-index: 1;
}

/* ─── Ambient Glow ─── */
.ambient-glow {
  position: fixed;
  border-radius: 50%;
  filter: blur(100px);
  z-index: -1;
  pointer-events: none;
}
.glow-warm {
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, rgba(245, 158, 11, 0.08), transparent 70%);
  top: -150px;
  right: -120px;
}
.glow-cool {
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(20, 184, 166, 0.06), transparent 70%);
  bottom: 10%;
  left: -100px;
}

/* ─── Header ─── */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40px;
}

.brand-mark {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-icon {
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(20, 184, 166, 0.1));
  border: 1px solid rgba(245, 158, 11, 0.2);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--accent-amber);
}

.brand-text h1 {
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.5px;
  margin: 0;
  color: var(--text-primary);
  line-height: 1.2;
}
.brand-en {
  font-weight: 300;
  opacity: 0.35;
  font-size: 1rem;
  margin-left: 6px;
}

.brand-tagline {
  margin: 2px 0 0;
  color: var(--text-muted);
  font-size: 0.78rem;
  font-weight: 400;
  letter-spacing: 0.02em;
}

/* Status Chip */
.status-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  background: rgba(15, 21, 32, 0.6);
  border: 1px solid var(--border-subtle);
  border-radius: 100px;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-muted);
  transition: all 0.4s ease;
}
.status-chip.online {
  color: var(--accent-teal);
  border-color: rgba(20, 184, 166, 0.25);
  background: rgba(20, 184, 166, 0.05);
}

.status-dot {
  width: 6px;
  height: 6px;
  background: currentColor;
  border-radius: 50%;
  transition: box-shadow 0.4s;
}
.status-chip.online .status-dot {
  box-shadow: 0 0 0 0 rgba(20, 184, 166, 0.5);
  animation: pulse-teal 2s infinite;
}

@keyframes pulse-teal {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(20, 184, 166, 0.5); }
  70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(20, 184, 166, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(20, 184, 166, 0); }
}

/* ─── Main ─── */
.app-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ─── Upload Zone ─── */
.drop-surface {
  background: rgba(15, 21, 32, 0.5) !important;
  border: 1.5px dashed rgba(71, 85, 105, 0.35) !important;
  border-radius: 16px !important;
  padding: 40px 24px !important;
  transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
  position: relative;
  overflow: hidden;
}
.drop-surface::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at center, rgba(245, 158, 11, 0.03), transparent 70%);
  opacity: 0;
  transition: opacity 0.35s;
}
.drop-surface:hover {
  border-color: rgba(245, 158, 11, 0.35) !important;
  background: rgba(20, 26, 38, 0.5) !important;
}
.drop-surface:hover::before {
  opacity: 1;
}
.drop-surface.drag-hover {
  background: rgba(245, 158, 11, 0.04) !important;
  border-color: var(--accent-amber) !important;
  border-style: solid !important;
  transform: scale(1.005);
  box-shadow: 0 0 40px rgba(245, 158, 11, 0.08);
}

.drop-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  position: relative;
  z-index: 1;
}

.drop-icon-ring {
  width: 56px;
  height: 56px;
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(20, 184, 166, 0.06));
  border: 1px solid rgba(245, 158, 11, 0.15);
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--accent-amber);
  transition: transform 0.3s, box-shadow 0.3s;
}
.drop-surface:hover .drop-icon-ring {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(245, 158, 11, 0.08);
}

.drop-copy h3 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-primary);
  font-weight: 500;
  text-align: center;
}
.drop-copy p {
  margin: 4px 0 0;
  font-size: 0.8rem;
  color: var(--text-muted);
  text-align: center;
}

/* ─── File Pill ─── */
.file-pill {
  margin-top: 12px;
  padding: 10px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.file-icon {
  color: var(--accent-amber);
  flex-shrink: 0;
}
.file-name {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.file-size {
  font-size: 0.72rem;
  color: var(--text-muted);
  flex-shrink: 0;
}
.file-clear {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  border: none;
  background: rgba(51, 65, 85, 0.2);
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}
.file-clear:hover {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
}

/* ─── Config Section ─── */
.config-section {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.config-row {
  display: grid;
  grid-template-columns: 1fr 1fr auto;
  gap: 12px;
  padding: 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
}

.config-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.config-toggle {
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.field-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}

/* ─── Action Button (Custom, no Naive UI) ─── */
.action-btn {
  position: relative;
  width: 100%;
  height: 50px;
  border: none;
  border-radius: 13px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.95rem;
  font-weight: 600;
  color: #0a0e17;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.action-btn:not(:disabled):hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 30px rgba(245, 158, 11, 0.25);
}
.action-btn:not(:disabled):active {
  transform: translateY(0);
}
.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.action-btn-bg {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, #f59e0b, #fbbf24, #f59e0b);
  background-size: 200% 200%;
  animation: gradient-shift 4s ease infinite;
}

@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.action-btn-content {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

/* Loading spinner */
.spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(10, 14, 23, 0.3);
  border-top-color: #0a0e17;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ─── Sections ─── */
.player-section,
.preview-section,
.raw-section {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 16px;
}

.section-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 12px;
}

.collapse-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-muted);
}

.raw-viewer {
  margin-top: 8px;
  padding: 12px;
  background: var(--bg-base);
  border-radius: 8px;
}

/* ─── Transitions ─── */
.slide-up-enter-active { transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
.slide-up-enter-from { opacity: 0; transform: translateY(12px); }

.scale-in-enter-active { transition: all 0.45s cubic-bezier(0.16, 1, 0.3, 1); }
.scale-in-enter-from { opacity: 0; transform: scale(0.97) translateY(6px); }

/* ─── Footer ─── */
.app-footer {
  margin-top: 48px;
  padding-bottom: 20px;
  text-align: center;
}
.footer-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
  margin-bottom: 20px;
}
.app-footer p {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-weight: 400;
  letter-spacing: 0.01em;
}
</style>
