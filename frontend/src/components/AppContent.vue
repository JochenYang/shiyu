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
                <span class="file-name" :title="selectedFile.name">{{
                  selectedFile.name
                }}</span>
                <span class="file-meta">{{
                  isVideo ? "视频文件" : "音频文件"
                }}</span>
              </div>
            </div>

            <n-upload
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
        <button class="icon-btn" @click="showSettings = true" title="设置">
          <Settings :size="18" />
        </button>
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
        <div class="custom-modal-body">
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
          <div class="setting-item">
            <div class="setting-label">排错日志</div>
            <button class="secondary-btn small-btn" @click="openLogDir">
              打开日志文件夹
            </button>
          </div>
          <div class="copyright-info">
            © 2026 时语 Shiyu Subtitle v1.0.0. All rights reserved.
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from "vue";
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
  Play,
  Pause,
  Rewind,
  FastForward,
  Volume2,
  VolumeX,
  X,
} from "lucide-vue-next";
import { listen } from "@tauri-apps/api/event";
import { convertFileSrc } from "@tauri-apps/api/tauri";
import { save, open } from "@tauri-apps/api/dialog";
import { writeTextFile } from "@tauri-apps/api/fs";
import {
  healthCheck,
  transcribeFile,
  transcribeJson,
  transcribeLocal,
  downloadText,
  openLogFolder,
} from "../api.js";

const message = useMessage();

// State
const selectedFile = ref(null);
const localFilePath = ref(null);
const audioUrl = ref(null);
const isVideo = ref(false);
const isProcessing = ref(false);
const backendOnline = ref(false);

const format = ref("srt");
const language = ref("zh");

const resultSegments = ref([]);
const currentSegIndex = ref(-1);
const currentSegmentText = ref("");

// Settings state
const showSettings = ref(false);
const appSettings = ref({
  autostart: false,
  startMinimized: false,
  defaultSavePath: "",
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
});

// Load settings
onMounted(async () => {
  const saved = localStorage.getItem("shiyu_settings");
  if (saved) {
    appSettings.value = { ...appSettings.value, ...JSON.parse(saved) };
  }

  pollTimer = setInterval(async () => {
    try {
      await healthCheck();
      backendOnline.value = true;
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
  if (audioUrl.value && !localFilePath.value)
    URL.revokeObjectURL(audioUrl.value);
  unlistenDrop?.();
  unlistenHover?.();
  unlistenCancel?.();
});

function handleFileSelected(filePathOrFile, isTauriPath = false) {
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

function handleUpload({ file }) {
  handleFileSelected(file.file, false);
}

async function startTranscribe() {
  if (!selectedFile.value) return;
  isProcessing.value = true;
  resultSegments.value = [];
  currentSegIndex.value = -1;

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

    if (localFilePath.value) {
      const result = await transcribeLocal(localFilePath.value, {
        language: language.value,
        output_format: format.value,
        glossary: glossaryStr,
      });
      if (result.segments) {
        resultSegments.value = result.segments;
      }
    } else {
      const json = await transcribeJson(
        selectedFile.value,
        language.value,
        glossaryStr,
      );
      if (json?.segments) {
        resultSegments.value = json.segments;
      }
    }
    if (resultSegments.value.length > 0) {
      currentSegIndex.value = 0;
    }
    message.success("字幕生成完成");
  } catch (err) {
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

  // Try to find the segment corresponding to the current time
  const idx = parsedSegments.value.findIndex(
    (seg) => time >= seg.start && time <= seg.end,
  );
  if (idx !== -1) {
    // Only auto-update if we are not actively editing a different one
    // or we could just strictly follow the playback head
    currentSegIndex.value = idx;
  }
}

function seekToSegment(idx) {
  lastManualSeekTime = Date.now();
  currentSegIndex.value = idx;
  if (mediaPlayer.value && parsedSegments.value[idx]) {
    mediaPlayer.value.currentTime = parsedSegments.value[idx].start;
    // Don't auto play, just seek
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
  width: 480px;
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
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
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
</style>
