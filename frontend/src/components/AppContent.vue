<template>
  <div class="app-shell">
    <!-- Header -->
    <header class="app-header">
      <div class="brand">
        <h1 class="brand-title">时语</h1>
        <span class="brand-sub">Shiyu · 本地 AI 字幕生成器</span>
      </div>
      <n-tag :type="backendOnline ? 'success' : 'error'" size="small" round :bordered="false">
        {{ backendOnline ? "服务在线" : "服务离线" }}
      </n-tag>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <!-- Step 1: Upload -->
      <n-card class="section-card" size="small">
        <n-upload
          :max="1"
          accept="audio/*,video/*"
          :show-file-list="false"
          :custom-request="handleUpload"
        >
          <n-upload-dragger class="upload-dragger">
            <n-icon size="36" :component="CloudUploadOutline" color="#667eea" />
            <div class="upload-text">
              <span class="upload-primary">拖拽或点击上传音频 / 视频文件</span>
              <span class="upload-hint">支持 MP3, WAV, FLAC, MP4, MKV 等</span>
            </div>
          </n-upload-dragger>
        </n-upload>

        <div v-if="selectedFile" class="file-info">
          <n-icon :component="DocumentTextOutline" color="#667eea" />
          <span class="file-name">{{ selectedFile.name }}</span>
          <span class="file-size">{{ formatFileSize(selectedFile.size) }}</span>
          <n-button quaternary circle size="tiny" @click="clearFile">
            <template #icon><n-icon :component="CloseOutline" /></template>
          </n-button>
        </div>
      </n-card>

      <!-- Step 2: Settings + Transcribe -->
      <n-card v-if="selectedFile" class="section-card" size="small">
        <div class="settings-grid">
          <div class="setting-item">
            <label class="setting-label">输出格式</label>
            <n-select v-model:value="format" :options="formatOptions" size="small" />
          </div>
          <div class="setting-item">
            <label class="setting-label">语言</label>
            <n-select v-model:value="language" :options="languageOptions" size="small" />
          </div>
          <div class="setting-item setting-switch">
            <label class="setting-label">逆文本规范化</label>
            <n-switch v-model:value="useItn" size="small" />
          </div>
        </div>
        <n-button
          type="primary"
          block
          strong
          :loading="isProcessing"
          :disabled="!backendOnline"
          @click="startTranscribe"
          class="transcribe-btn"
        >
          {{ isProcessing ? "识别中..." : "开始生成字幕" }}
        </n-button>
      </n-card>

      <!-- Step 3: Wave Player -->
      <n-card v-if="audioUrl" class="section-card" size="small">
        <template #header>
          <span class="card-header-text">音频预览</span>
        </template>
        <WavePlayer
          :src="audioUrl"
          :segments="parsedSegments"
          @timeupdate="onTimeUpdate"
        />
      </n-card>

      <!-- Step 4: Subtitle Preview -->
      <n-card v-if="resultContent" class="section-card" size="small">
        <SubtitlePreview
          :segments="parsedSegments"
          :format="format"
          :current-time="playbackTime"
          @download="downloadResult"
          @copy="copyResult"
          @seek="seekPlayer"
        />
      </n-card>

      <!-- Raw SRT (collapsible) -->
      <n-card v-if="resultContent" class="section-card section-card-raw" size="small">
        <n-collapse :default-expanded-names="[]">
          <n-collapse-item title="原始字幕内容" name="raw">
            <n-code :code="resultContent" language="txt" word-wrap />
          </n-collapse-item>
        </n-collapse>
      </n-card>
    </main>

    <!-- Footer -->
    <footer class="app-footer">
      ©2026 时语 Shiyu · Jochen · 完全本地运行
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from "vue";
import {
  NCard, NButton, NTag,
  NUpload, NUploadDragger, NSelect, NSwitch,
  NIcon, NCollapse, NCollapseItem, NCode,
  useMessage,
} from "naive-ui";
import {
  CloudUploadOutline, DocumentTextOutline, CloseOutline,
} from "@vicons/ionicons5";
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

let pollTimer = null;

onMounted(() => {
  pollTimer = setInterval(async () => {
    try {
      await healthCheck();
      backendOnline.value = true;
    } catch {
      backendOnline.value = false;
    }
  }, 2000);
});

onBeforeUnmount(() => {
  clearInterval(pollTimer);
  if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
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
  max-width: 680px;
  margin: 0 auto;
  padding: 28px 20px;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* Header */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.brand-title {
  font-size: 1.6rem;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
  line-height: 1.2;
}

.brand-sub {
  font-size: 0.75rem;
  color: #777;
  margin-top: 2px;
  display: block;
}

/* Main */
.app-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.section-card {
  border-radius: 10px;
}

/* Upload */
.upload-dragger {
  padding: 28px 0 !important;
}

.upload-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 8px;
}

.upload-primary {
  font-size: 0.95rem;
  color: #bbb;
}

.upload-hint {
  font-size: 0.78rem;
  color: #666;
  margin-top: 4px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
  padding: 8px 12px;
  background: rgba(102, 126, 234, 0.06);
  border-radius: 8px;
}

.file-name {
  color: #e0e0e0;
  font-size: 0.9rem;
}

.file-size {
  color: #888;
  font-size: 0.78rem;
}

/* Settings */
.settings-grid {
  display: grid;
  grid-template-columns: 1fr 1fr auto;
  gap: 16px;
  margin-bottom: 16px;
  align-items: end;
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.setting-switch {
  min-width: 120px;
}

.setting-label {
  font-size: 0.78rem;
  color: #999;
}

.transcribe-btn {
  margin-top: 4px;
  height: 40px;
  font-size: 0.95rem;
}

.card-header-text {
  font-weight: 600;
  font-size: 0.9rem;
  color: #e0e0e0;
}

.section-card-raw :deep(.n-collapse-item__header-main) {
  font-size: 0.85rem;
}

/* Footer */
.app-footer {
  text-align: center;
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  color: #555;
  font-size: 0.72rem;
}
</style>
