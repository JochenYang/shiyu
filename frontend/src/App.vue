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
import { NConfigProvider, NMessageProvider, darkTheme } from "naive-ui";
import AppContent from "./components/AppContent.vue";

/**
 * Shiyu Design System Tokens
 * Palette: Deep slate base + warm amber/teal accents.
 * Avoids generic indigo to create a distinctive, professional identity.
 */
const themeOverrides = {
  common: {
    primaryColor: "#f59e0b",
    primaryColorHover: "#fbbf24",
    primaryColorPressed: "#d97706",
    primaryColorSuppl: "#f59e0b",
    
    textColor1: "#f0f4f8",
    textColor2: "#94a3b8",
    textColor3: "#64748b",
    
    bodyColor: "#0a0e17",
    cardColor: "#0f1520",
    modalColor: "#0f1520",
    popoverColor: "#0f1520",
    
    borderRadius: "12px",
    fontFamily: '"Outfit", "PingFang SC", "Microsoft YaHei", sans-serif',
  },
  Card: {
    borderRadius: "16px",
    color: "rgba(15, 21, 32, 0.6)",
    borderColor: "rgba(51, 65, 85, 0.25)",
  },
  Button: {
    borderRadiusMedium: "10px",
    fontWeight: "600",
  },
  Input: {
    borderRadius: "10px",
    color: "rgba(10, 14, 23, 0.6)",
    colorFocus: "rgba(10, 14, 23, 0.8)",
  },
};
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
  color-scheme: dark;
  /* Design tokens */
  --bg-base: #0a0e17;
  --bg-surface: #0f1520;
  --bg-elevated: #141c2b;
  --border-subtle: rgba(51, 65, 85, 0.25);
  --border-medium: rgba(71, 85, 105, 0.35);
  --accent-amber: #f59e0b;
  --accent-amber-glow: rgba(245, 158, 11, 0.15);
  --accent-teal: #14b8a6;
  --accent-teal-glow: rgba(20, 184, 166, 0.12);
  --text-primary: #f0f4f8;
  --text-secondary: #94a3b8;
  --text-muted: #475569;
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

/* Clean layout with subtle ambient gradients */
.shiyu-layout {
  background:
    radial-gradient(ellipse at 20% 0%, rgba(245, 158, 11, 0.04) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 100%, rgba(20, 184, 166, 0.03) 0%, transparent 40%);
  min-height: 100vh;
}

/* Naive UI select dropdown override */
.n-base-select-menu {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-subtle) !important;
}
</style>
