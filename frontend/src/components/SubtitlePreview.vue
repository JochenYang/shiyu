<template>
  <div class="subtitle-preview">
    <div class="preview-header">
      <span class="preview-title">字幕预览</span>
      <div class="preview-actions">
        <button class="action-chip" @click="$emit('download')">
          <Download :size="13" />
          下载 {{ format.toUpperCase() }}
        </button>
        <button class="action-chip" @click="$emit('copy')">
          <Copy :size="13" />
          复制
        </button>
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
import { Download, Copy } from "lucide-vue-next";

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
  background: transparent;
  border-radius: 10px;
  overflow: hidden;
}

.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 2px 12px;
  border-bottom: 1px solid rgba(51, 65, 85, 0.2);
}

.preview-title {
  font-weight: 600;
  font-size: 0.85rem;
  color: var(--text-secondary, #94a3b8);
}

.preview-actions {
  display: flex;
  gap: 6px;
}

/* Custom action chips replacing Naive UI buttons */
.action-chip {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 8px;
  border: 1px solid var(--border-subtle, rgba(51, 65, 85, 0.25));
  background: rgba(15, 21, 32, 0.5);
  color: var(--text-secondary, #94a3b8);
  font-family: inherit;
  font-size: 0.72rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}
.action-chip:hover {
  background: rgba(245, 158, 11, 0.08);
  border-color: rgba(245, 158, 11, 0.25);
  color: var(--accent-amber, #f59e0b);
}

.preview-list {
  max-height: 240px;
  overflow-y: auto;
  padding: 6px 0;
}

.seg-row {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 6px 10px;
  cursor: pointer;
  transition: background 0.15s;
  border-radius: 6px;
  margin: 1px 0;
}

.seg-row:hover {
  background: rgba(245, 158, 11, 0.05);
}

.seg-row.active {
  background: rgba(245, 158, 11, 0.1);
}

.seg-idx {
  color: var(--accent-amber, #f59e0b);
  font-size: 0.7rem;
  min-width: 18px;
  text-align: right;
  flex-shrink: 0;
  opacity: 0.7;
}

.seg-time {
  color: var(--text-muted, #475569);
  font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  flex-shrink: 0;
}

.seg-text {
  color: var(--text-primary, #f0f4f8);
  font-size: 0.82rem;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.seg-row.active .seg-text {
  color: var(--accent-amber, #f59e0b);
}

.preview-empty {
  text-align: center;
  padding: 24px;
  color: var(--text-muted, #475569);
  font-size: 0.82rem;
}
</style>
