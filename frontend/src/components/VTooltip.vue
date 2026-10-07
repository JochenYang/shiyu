<template>
  <!--
    自定义 Tooltip (替代原生 title=""):
    悬停弹出, 弹性浮入 (借鉴 beUI popover 的 spring 曲线, Vue 重写);
    自动判断上下翻转 (靠近视口顶部时显示在下方); 键盘聚焦同样触发。
  -->
  <span
    class="vtip-wrap"
    @mouseenter="show"
    @mouseleave="hide"
    @focusin="show"
    @focusout="hide"
  >
    <slot></slot>
    <Teleport to="body">
      <Transition name="vtip">
        <span
          v-if="visible"
          ref="tipEl"
          class="vtip"
          :style="tipStyle"
          role="tooltip"
        >{{ content }}</span>
      </Transition>
    </Teleport>
  </span>
</template>

<script setup>
import { computed, nextTick, ref } from "vue";

const props = defineProps({
  content: { type: String, required: true },
  placement: { type: String, default: "bottom" }, // bottom | top
});
const visible = ref(false);
const anchorEl = ref(null);
const tipEl = ref(null);
const pos = ref({ left: 0, top: 0, placement: props.placement });

const tipStyle = computed(() => ({
  left: `${pos.value.left}px`,
  top: `${pos.value.top}px`,
}));

function show(e) {
  visible.value = true;
  anchorEl.value = e.currentTarget;
  nextTick(updatePosition);
}

function hide() {
  visible.value = false;
}

function updatePosition() {
  const anchor = anchorEl.value;
  const tip = tipEl.value;
  if (!anchor || !tip) return;
  const a = anchor.getBoundingClientRect();
  const t = tip.getBoundingClientRect();
  let placement = props.placement;
  // 靠近视口边缘时自动翻转
  if (placement === "bottom" && a.bottom + t.height + 10 > window.innerHeight) {
    placement = "top";
  } else if (placement === "top" && a.top - t.height - 10 < 0) {
    placement = "bottom";
  }
  const left = Math.max(
    8,
    Math.min(
      a.left + a.width / 2 - t.width / 2,
      window.innerWidth - t.width - 8
    )
  );
  const top =
    placement === "bottom" ? a.bottom + 8 : a.top - t.height - 8;
  pos.value = { left, top, placement };
}
</script>

<style scoped>
.vtip-wrap {
  display: inline-flex;
}
</style>

<style>
/* Teleport 到 body, 不能用 scoped */
.vtip {
  position: fixed;
  z-index: 9999;
  max-width: 280px;
  padding: 6px 10px;
  border-radius: 7px;
  background: #0d0d0d;
  border: 1px solid #333;
  color: #ddd;
  font-size: 0.74rem;
  line-height: 1.5;
  white-space: pre-line;
  pointer-events: none;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
}
/* 弹性浮入浮出 (spring 感: 过冲后回落) */
.vtip-enter-active {
  transition:
    opacity 0.16s ease,
    transform 0.24s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.vtip-leave-active {
  transition:
    opacity 0.12s ease,
    transform 0.14s ease;
}
.vtip-enter-from,
.vtip-leave-to {
  opacity: 0;
}
.vtip-enter-from {
  transform: translateY(-4px) scale(0.92);
}
.vtip-leave-to {
  transform: translateY(-2px) scale(0.96);
}
</style>
