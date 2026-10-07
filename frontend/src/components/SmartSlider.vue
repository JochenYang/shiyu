<template>
  <!--
    气泡滑块 (借鉴 beUI range-slider-bubble 的动效, Vue 重写):
    数值气泡跟随 thumb 移动, 拖动时弹起放大, 松手回落。
    气泡位置按百分比计算, 与原生 input 的 thumb 轨道对齐。
  -->
  <div class="bslider" :class="{ 'is-disabled': disabled, 'is-focused': focused }">
    <div class="bslider-track-wrap">
      <div class="bslider-bubble"
        :style="bubbleStyle"
        :class="{ 'is-active': dragging }"
        aria-hidden="true">
        <span class="bslider-bubble-text">{{ displayValue }}</span>
      </div>
      <div class="bslider-rail"></div>
      <div class="bslider-fill" :style="fillStyle"></div>
      <input
        ref="inputEl"
        type="range"
        class="bslider-input"
        :min="min"
        :max="max"
        :step="step"
        :value="modelValue"
        :disabled="disabled"
        :aria-label="ariaLabel || '滑块'"
        @input="onInput"
        @pointerdown="dragging = true"
        @pointerup="dragging = false"
        @pointercancel="dragging = false"
        @keydown="dragging = true"
        @focus="focused = true"
        @blur="onBlur"
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from "vue";

const props = defineProps({
  modelValue: { type: Number, required: true },
  min: { type: Number, default: 0 },
  max: { type: Number, default: 100 },
  step: { type: Number, default: 1 },
  disabled: { type: Boolean, default: false },
  format: { type: Function, default: null },
  ariaLabel: { type: String, default: "" },
});
const emit = defineEmits(["update:modelValue"]);

const dragging = ref(false);
const focused = ref(false);
const inputEl = ref(null);

const percent = computed(() => {
  const span = props.max - props.min;
  if (span <= 0) return 0;
  return Math.min(100, Math.max(0, ((props.modelValue - props.min) / span) * 100));
});

const displayValue = computed(() =>
  props.format ? props.format(props.modelValue) : String(props.modelValue)
);

// thumb 宽度 16px: 轨道两端各有 8px 的死区, 百分比要映射到
// [8px, calc(100% - 8px)] 才与 thumb 视觉中心重合
const bubbleStyle = computed(() => ({
  left: `calc(${percent.value}% + ${(50 - percent.value) * 0.16}px)`,
}));

const fillStyle = computed(() => ({ width: `${percent.value}%` }));

function onInput(e) {
  emit("update:modelValue", Number(e.target.value));
}

function onBlur() {
  focused.value = false;
  dragging.value = false;
}
</script>

<style scoped>
.bslider {
  flex: 1;
  padding-top: 30px; /* 给气泡留出弹起空间 */
}
.bslider-track-wrap {
  position: relative;
  height: 16px;
  display: flex;
  align-items: center;
}
.bslider-rail {
  position: absolute;
  left: 0;
  right: 0;
  height: 4px;
  border-radius: 2px;
  background: #333;
  pointer-events: none;
}
.bslider-fill {
  position: absolute;
  left: 0;
  height: 4px;
  border-radius: 2px;
  background: #1fbc5b;
  pointer-events: none;
  transition: width 0.05s linear;
}
.bslider-input {
  position: relative;
  z-index: 2;
  width: 100%;
  height: 16px;
  margin: 0;
  background: transparent;
  -webkit-appearance: none;
  appearance: none;
  outline: none;
  cursor: pointer;
}
.bslider-input::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #1fbc5b;
  border: 2px solid #0e2a18;
  box-shadow: 0 0 0 2px rgba(31, 188, 91, 0.25);
  transition: transform 0.15s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.bslider-input:active::-webkit-slider-thumb {
  transform: scale(1.25);
}
.bslider-input::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #1fbc5b;
  border: 2px solid #0e2a18;
  box-shadow: 0 0 0 2px rgba(31, 188, 91, 0.25);
}
.bslider-bubble {
  position: absolute;
  top: -30px;
  transform: translateX(-50%) translateY(4px) scale(0.85);
  opacity: 0;
  padding: 3px 9px;
  border-radius: 7px;
  background: #1fbc5b;
  color: #06130b;
  font-size: 0.72rem;
  font-weight: 600;
  white-space: nowrap;
  pointer-events: none;
  transition:
    left 0.05s linear,
    opacity 0.18s ease,
    transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1);
  z-index: 3;
}
/* 气泡小尾巴 */
.bslider-bubble::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -3px;
  width: 8px;
  height: 8px;
  background: #1fbc5b;
  transform: translateX(-50%) rotate(45deg);
  border-radius: 1px;
}
/* 悬停半显, 拖动/键盘聚焦完全弹起 (用父级状态类, 不依赖兄弟顺序) */
.bslider:hover .bslider-bubble {
  opacity: 0.55;
  transform: translateX(-50%) translateY(2px) scale(0.92);
}
.bslider:hover .bslider-bubble.is-active,
.bslider.is-focused .bslider-bubble {
  opacity: 1;
  transform: translateX(-50%) translateY(0) scale(1);
}

.bslider.is-disabled {
  opacity: 0.5;
}
.bslider.is-disabled .bslider-input {
  cursor: not-allowed;
}
</style>
