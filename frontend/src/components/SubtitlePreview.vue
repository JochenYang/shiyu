<template>
  <div class="subtitle-preview">
    <div class="preview-header">
      <span class="preview-title">字幕预览</span>
      <div class="preview-actions">
        <n-button size="tiny" secondary @click="$emit('download')">
          下载 {{ format.toUpperCase() }}
        </n-button>
        <n-button size="tiny" secondary @click="$emit('copy')">
          复制
        </n-button>
      </div>
    </div>
    <div class="preview-list" v-if="segments.length">
      <div
        v-for="(seg, i) in segments"
        :key="i"
        class="seg-row"
        :class="{ active: isSegmentActive(seg) }"
        @click="$emit('seek', seg.start)"
      >
        <span class="seg-idx">{{ i + 1 }}</span>
        <span class="seg-time">{{ fmtTime(seg.start) }} → {{ fmtTime(seg.end) }}</span>
        <span class="seg-text">{{ seg.text }}</span>
      </div>
    </div>
    <div class="preview-empty" v-else>暂无字幕</div>
  </div>
</template>

<script setup>
import { NButton } from "naive-ui";

const props = defineProps({
  segments: { type: Array, default: () => [] },
  format: { type: String, default: "srt" },
  currentTime: { type: Number, default: 0 },
});

defineEmits(["download", "copy", "seek"]);

function fmtTime(sec) {
  if (sec == null || sec < 0) return "00:00,000";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 1000);
  const mm = m.toString().padStart(2, "0");
  const ss = s.toString().padStart(2, "0");
  const mss = ms.toString().padStart(3, "0");
  if (h > 0) return `${h}:${mm}:${ss},${mss}`;
  return `${mm}:${ss},${mss}`;
}

function isSegmentActive(seg) {
  const t = props.currentTime;
  return t >= seg.start && t <= seg.end;
}
</script>

<style scoped>
.subtitle-preview {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 10px;
  overflow: hidden;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.preview-title {
  font-weight: 600;
  font-size: 0.88rem;
  color: #e0e0e0;
}

.preview-actions {
  display: flex;
  gap: 6px;
}

.preview-list {
  max-height: 220px;
  overflow-y: auto;
  padding: 4px 0;
}

.seg-row {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 5px 14px;
  cursor: pointer;
  transition: background 0.12s;
  border-radius: 4px;
  margin: 0 4px;
}

.seg-row:hover {
  background: rgba(102, 126, 234, 0.08);
}

.seg-row.active {
  background: rgba(102, 126, 234, 0.16);
}

.seg-idx {
  color: #667eea;
  font-size: 0.72rem;
  min-width: 16px;
  text-align: right;
  flex-shrink: 0;
}

.seg-time {
  color: #888;
  font-size: 0.72rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  flex-shrink: 0;
}

.seg-text {
  color: #ddd;
  font-size: 0.85rem;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preview-empty {
  text-align: center;
  padding: 20px;
  color: #666;
  font-size: 0.85rem;
}
</style>
