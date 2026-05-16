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

onMounted(() => {
  ws = WaveSurfer.create({
    container: waveform.value,
    waveColor: "rgba(102, 126, 234, 0.5)",
    progressColor: "rgba(118, 75, 162, 0.85)",
    cursorColor: "rgba(255, 255, 255, 0.6)",
    cursorWidth: 1,
    height: 48,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    normalize: true,
    interact: true,
    minPxPerSec: 50,
  });

  ws.on("ready", () => {
    duration.value = ws.getDuration();
    emit("ready");
  });

  ws.on("audioprocess", () => {
    if (!sliderDragging) {
      currentTime.value = ws.getCurrentTime();
      if (duration.value > 0) {
        sliderValue.value = Math.round((currentTime.value / duration.value) * 1000);
      }
    }
    emit("timeupdate", currentTime.value);
  });

  ws.on("seeking", () => {
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
    const time = (val / 1000) * duration.value;
    ws.seekTo(val / 1000);
    currentTime.value = time;
    emit("timeupdate", time);
  }
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

</style>
