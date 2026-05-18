<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <n-message-provider>
      <div class="shiyu-layout">
        <AppContent />
      </div>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup>
import { onMounted, onBeforeUnmount } from "vue";
import { NConfigProvider, NMessageProvider, darkTheme } from "naive-ui";
import AppContent from "./components/AppContent.vue";

// 屏蔽全局默认右键菜单（输入框除外），增强应用原生感
function preventContextMenu(e) {
  const target = e.target;
  const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;
  if (!isInput) {
    e.preventDefault();
  }
}

onMounted(() => {
  document.addEventListener('contextmenu', preventContextMenu);
});

onBeforeUnmount(() => {
  document.removeEventListener('contextmenu', preventContextMenu);
});

/**
 * Shiyu Design System Tokens
 * Palette: Deep slate base + warm amber/teal accents.
 * Avoids generic indigo to create a distinctive, professional identity.
 */
const themeOverrides = {
  common: {
    primaryColor: "#1fbc5b",
    primaryColorHover: "#1db155",
    primaryColorPressed: "#189447",
    primaryColorSuppl: "#1fbc5b",
    
    textColor1: "#e0e0e0",
    textColor2: "#a0a0a0",
    textColor3: "#666666",
    
    bodyColor: "#111111",
    cardColor: "#1c1c1c",
    modalColor: "#1c1c1c",
    popoverColor: "#1c1c1c",
    
    borderRadius: "8px",
    fontFamily: '"Outfit", "PingFang SC", "Microsoft YaHei", sans-serif',
  },
  Card: {
    borderRadius: "12px",
    color: "#1c1c1c",
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  Button: {
    borderRadiusMedium: "6px",
    fontWeight: "500",
  },
  Input: {
    borderRadius: "6px",
    color: "#111111",
    colorFocus: "#1c1c1c",
  },
};
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
  color-scheme: dark;
  /* Design tokens */
  --bg-base: #111111;
  --bg-surface: #1c1c1c;
  --bg-elevated: #242424;
  --border-subtle: rgba(255, 255, 255, 0.1);
  --border-medium: rgba(255, 255, 255, 0.2);
  --accent-green: #1fbc5b;
  --text-primary: #e0e0e0;
  --text-secondary: #a0a0a0;
  --text-muted: #666666;
}

html, body, #app {
  margin: 0;
  padding: 0;
  background: var(--bg-base);
  min-height: 100vh;
  color: var(--text-primary);
  font-family: 'Outfit', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Premium scrollbar */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.12);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(148, 163, 184, 0.25);
}

/* Clean layout */
.shiyu-layout {
  background: var(--bg-base);
  min-height: 100vh;
}

/* Naive UI select dropdown override */
.n-base-select-menu {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-subtle) !important;
}
</style>
