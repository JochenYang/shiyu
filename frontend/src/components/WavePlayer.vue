<template>
  <div class="wave-player">
    <div ref="waveform" class="waveform"></div>
    <div class="player-bar">
      <n-button quaternary circle size="small" @click="togglePlay" class="play-btn">
        <template #icon>
          <n-icon size="18" :component="isPlaying ? PauseOutline : PlayOutline" />
        </template>
      </n-button>
      <span class="time-current">{{ formatTime(currentTime) }}</span>
      <n-slider
        v-model:value="sliderValue"
        :max="1000"
        :step="1"
        size="small"
        class="time-slider"
        :format-tooltip="formatSliderTooltip"
        @update:value="onSliderChange"
      />
      <span class="time-total">{{ formatTime(duration) }}</span>
    </div>

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from "vue";
import WaveSurfer from "wavesurfer.js";
import { NButton, NIcon, NSlider } from "naive-ui";
import { PlayOutline, PauseOutline } from "@vicons/ionicons5";

const props = defineProps({
  src: { type: [String, Object], default: null },
  segments: { type: Array, default: () => [] },
});

const emit = defineEmits(["timeupdate", "ready"]);

const waveform = ref(null);
const isPlaying = ref(false);
const currentTime = ref(0);
const duration = ref(0);
const sliderValue = ref(0);
let ws = null;
let sliderDragging = false;
let sliderChangePending = false;

onMounted(() => {
  ws = WaveSurfer.create({
    container: waveform.value,
    waveColor: "rgba(148, 163, 184, 0.3)",
    progressColor: "rgba(245, 158, 11, 0.7)",
    cursorColor: "rgba(245, 158, 11, 0.5)",
    cursorWidth: 1,
    height: 48,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    normalize: true,
    interact: true,
    // Let waveform auto-fit container width (no horizontal scroll)
    fillParent: true,
  });

  ws.on("ready", () => {
    duration.value = ws.getDuration();
    emit("ready");
  });

  ws.on("audioprocess", () => {
    // Skip sliderValue updates while a user-initiated seek is pending
    // to prevent the slider from snapping back to the old position
    // before the seeking event fires.
    if (!sliderDragging && !sliderChangePending) {
      currentTime.value = ws.getCurrentTime();
      if (duration.value > 0) {
        sliderValue.value = Math.round((currentTime.value / duration.value) * 1000);
      }
    }
    emit("timeupdate", currentTime.value);
  });

  ws.on("seeking", () => {
    // Clear pending flag so audioprocess can resume updating sliderValue
    sliderChangePending = false;
    const t = ws.getCurrentTime();
    currentTime.value = t;
    if (duration.value > 0) {
      sliderValue.value = Math.round((t / duration.value) * 1000);
    }
  });

  ws.on("play", () => { isPlaying.value = true; });
  ws.on("pause", () => { isPlaying.value = false; });

  if (props.src) loadAudio(props.src);
});

onBeforeUnmount(() => {
  ws?.destroy();
});

watch(() => props.src, (val) => {
  if (val) loadAudio(val);
});

function loadAudio(src) {
  if (!ws) return;
  ws.load(src);
}

function togglePlay() {
  ws?.playPause();
}

function onSliderChange(val) {
  if (ws && duration.value > 0) {
    sliderChangePending = true;
    const time = (val / 1000) * duration.value;
    ws.seekTo(val / 1000);
    currentTime.value = time;
    emit("timeupdate", time);
  }
}

/**
 * Format slider tooltip value (0-1000) as HH:MM:SS.
 * Shows human-readable time instead of raw numeric value.
 */
function formatSliderTooltip(val) {
  if (duration.value <= 0) return "0:00";
  const sec = (val / 1000) * duration.value;
  return formatTime(sec);
}

function formatTime(sec) {
  if (!sec || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

defineExpose({ loadAudio, togglePlay });
</script>

<style scoped>
.wave-player {
  position: relative;
}

.waveform {
  border-radius: 6px;
  overflow: hidden;
}

/* Hide any leftover horizontal scrollbar inside wavesurfer shadow container */
.waveform :deep(div) {
  overflow-x: hidden !important;
}

.player-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}

.play-btn {
  flex-shrink: 0;
}

.time-current,
.time-total {
  font-size: 0.75rem;
  color: #999;
  font-variant-numeric: tabular-nums;
  min-width: 36px;
  flex-shrink: 0;
}

.time-current {
  text-align: right;
}

.time-total {
  text-align: left;
}

.time-slider {
  flex: 1;
}
.time-slider :deep(.n-slider-rail) {
  background: rgba(51, 65, 85, 0.3);
  height: 3px;
  border-radius: 2px;
}
.time-slider :deep(.n-slider-fill) {
  background: linear-gradient(90deg, #f59e0b, #fbbf24);
  border-radius: 2px;
}
.time-slider :deep(.n-slider-thumb) {
  width: 12px;
  height: 12px;
  border: 2px solid #f59e0b;
  box-shadow: 0 0 6px rgba(245, 158, 11, 0.3);
}
.time-slider :deep(.n-slider-thumb:hover) {
  box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
}

</style>
