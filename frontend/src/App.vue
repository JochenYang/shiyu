<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <n-message-provider>
      <AppContent />
    </n-message-provider>
  </n-config-provider>
</template>

<script setup>
import { NConfigProvider, NMessageProvider, darkTheme } from "naive-ui";
import AppContent from "./components/AppContent.vue";

const themeOverrides = {
  common: {
    primaryColor: "#667eea",
    primaryColorHover: "#7b93f0",
    primaryColorPressed: "#5568d6",
    primaryColorSuppl: "#667eea",
    textColor1: "#e8e8e8",
    textColor2: "#c8c8c8",
    textColor3: "#999999",
    cardColor: "#1a1a24",
    modalColor: "#1a1a24",
    popoverColor: "#1a1a24",
    inputColor: "#14141c",
  },
};
</script>

<style>
html, body, #app {
  margin: 0;
  padding: 0;
  background: #101014;
  min-height: 100vh;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
}
</style>
