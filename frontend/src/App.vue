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
 * Premium Design System Tokens
 * Using a refined Slate-Indigo palette for a professional Desktop feel.
 */
const themeOverrides = {
  common: {
    primaryColor: "#818cf8",
    primaryColor: "#6366f1",
    primaryColorHover: "#818cf8",
    primaryColorPressed: "#4f46e5",
    primaryColorSuppl: "#6366f1",
    
    textColor1: "#f1f5f9",
    textColor2: "#94a3b8",
    textColor3: "#64748b",
    
    bodyColor: "#020617",
    cardColor: "#0a0f1d",
    modalColor: "#0a0f1d",
    popoverColor: "#0a0f1d",
    
    borderRadius: "12px",
    fontFamily: '"Inter", "Outfit", "PingFang SC", sans-serif',
  },
  Card: {
    borderRadius: "20px",
    color: "#0a0f1d",
    borderColor: "rgba(51, 65, 85, 0.4)",
  },
  Button: {
    borderRadiusMedium: "8px",
    fontWeight: "500",
  },
  Input: {
    borderRadius: "10px",
    color: "rgba(2, 6, 23, 0.6)",
    colorFocus: "rgba(2, 6, 23, 0.8)",
  },
};
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
  color-scheme: dark;
}

html, body, #app {
  margin: 0;
  padding: 0;
  background: #020617;
  min-height: 100vh;
  color: #f8fafc;
  font-family: 'Outfit', 'PingFang SC', sans-serif;
  -webkit-font-smoothing: antialiased;
}

/* Premium Scrollbar Design */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.1);
  border-radius: 20px;
  border: 3px solid transparent;
  background-clip: content-box;
  transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(148, 163, 184, 0.3);
  border-radius: 20px;
  border: 3px solid transparent;
  background-clip: content-box;
}

.shiyu-layout {
  background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
              radial-gradient(circle at bottom left, rgba(79, 70, 229, 0.03) 0%, transparent 30%);
  min-height: 100vh;
}
</style>
