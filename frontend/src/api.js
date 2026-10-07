/**
 * API client for Subtitle Assistant backend.
 * Communicates with local FastAPI service.
 */

const API_BASE = "http://127.0.0.1:11235";

/**
 * Check if backend is alive.
 */
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

/**
 * Command backend to open logs folder
 */
export async function openLogFolder() {
  const res = await fetch(`${API_BASE}/logs/open`);
  return res.json();
}

/**
 * Transcribe file and get subtitle content.
 * @param {File} file - Audio or video file
 * @param {Object} options - { language, output_format, max_line_length, max_line_count }
 * @returns {Promise<string>} Subtitle text content
 */
export async function transcribeFile(file, options = {}) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("language", options.language || "auto");
  formData.append("output_format", options.output_format || "srt");
  formData.append("glossary", options.glossary || "{}");

  const res = await fetch(`${API_BASE}/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.text();
}

/**
 * Transcribe file and get raw JSON segments.
 * @param {File} file - Audio or video file
 * @param {string} language - Target language
 * @returns {Promise<Object>} { filename, language, duration, segments }
 */
export async function transcribeJson(file, language = "auto", glossary = "{}", engine = "local", subtitleOffset = 0) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("language", language);
  formData.append("glossary", glossary);
  formData.append("engine_mode", engine);
  formData.append("subtitle_offset_sec", String(subtitleOffset));

  const res = await fetch(`${API_BASE}/transcribe/json`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

/**
 * Transcribe a local file by path (Tauri drag-drop, no upload needed).
 * Returns both subtitle content and segments in one request.
 * @param {string} filePath - Absolute path to the file on disk
 * @param {Object} options - { language, output_format }
 * @returns {Promise<Object>} { content, segments, filename, duration, language }
 */
export async function transcribeLocal(filePath, options = {}) {
  const formData = new FormData();
  formData.append("path", filePath);
  formData.append("language", options.language || "auto");
  formData.append("output_format", options.output_format || "srt");
  formData.append("glossary", options.glossary || "{}");
  formData.append("engine_mode", options.engine || "local");
  formData.append("subtitle_offset_sec", String(options.subtitleOffset || 0));

  const res = await fetch(`${API_BASE}/transcribe/local`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

/**
 * Download text content as file.
 */
export function downloadText(content, filename) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Get list of available ASR engines.
 * @returns {Promise<Object>} { engines: [{ name, display_name, requires_network, is_local, available, reason }] }
 */
export async function getEngines() {
  const res = await fetch(`${API_BASE}/engines`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Get model download/loading status.
 * @returns {Promise<Object>} { status, progress, error, engine_loaded, model_path, model_size_mb }
 */
/**
 * 列出模型清单与安装状态 (只读, 不触发下载)。
 * @returns {Promise<Object>} { models: [{id,name,engine,description,languages,installed,sizeBytes,...}], defaultModelID, job, jobs }
 */
export async function getModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 开始下载并安装模型 (非阻塞, 立即返回任务状态)。
 * @param {string} modelId - 清单中的模型 id
 * @param {string} source - "auto" (国内镜像优先) 或 "official" (仅官方源)
 */
export async function installModel(modelId, source = "auto") {
  const formData = new FormData();
  formData.append("source", source);
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/install`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 移除模型 (删除其全部文件)。
 * @param {string} modelId - 清单中的模型 id
 */
export async function removeModel(modelId) {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/remove`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 设置默认识别模型 (持久化到用户配置)。
 * @param {string} modelId
 */
export async function setDefaultModel(modelId) {
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/default`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 取消一个运行中的下载任务 (残余保留, 重试可继续)。
 * @param {string} jobId
 */
export async function cancelModelJob(jobId) {
  const res = await fetch(`${API_BASE}/models/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 列出所有下载任务状态。
 */
export async function getModelJobs() {
  const res = await fetch(`${API_BASE}/models/jobs`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Open model folder in file explorer.
 */
export async function openModelFolder() {
  const res = await fetch(`${API_BASE}/model/open`);
  return res.json();
}

/**
 * Change backend log level dynamically.
 * @param {string} level - "debug" | "info" | "warning" | "error"
 */
export async function setLogLevel(level) {
  const formData = new FormData();
  formData.append("level", level);
  const res = await fetch(`${API_BASE}/settings/log-level`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

/**
 * 云端多配置: 取全部配置 (脱敏) 与激活 id。
 * @returns {Promise<Object>} { engines: [{id, api_url, model, has_key, enabled, request_mode, display_name}], active }
 */
export async function getEngineConfigs() {
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 内置云端 provider 清单 (含 base_url/模型/认证头预填信息)。
 * @returns {Promise<Object>} { providers: [{id, name, api_url, request_mode, models, auth_header, note}] }
 */
export async function getProviders() {
  const res = await fetch(`${API_BASE}/settings/providers`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Apply LLM correction to transcription segments.
 * @param {string} engineMode - "local" or "cloud"
 * @returns {Promise<{segments: Array, llm_corrected: boolean}>}
 */
export async function correctTranscription(segments, engineMode) {
  const res = await fetch(`${API_BASE}/transcribe/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ segments, engine_mode: engineMode }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Get LLM correction settings from backend.
 * @returns {Promise<Object>} { enabled, provider, api_url, has_key, model }
 */
export async function getLlmCorrectionSettings() {
  const res = await fetch(`${API_BASE}/settings/llm-correction`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Save LLM correction configuration to backend.
 * @param {Object} settings - { enabled, provider, api_url, api_key, model }
 */
export async function saveLlmCorrectionSettings(settings) {
  const formData = new FormData();
  formData.append("enabled", settings.enabled ? "true" : "false");
  formData.append("provider", settings.provider || "openai");
  formData.append("api_url", settings.api_url || "");
  formData.append("api_key", settings.api_key || "");
  formData.append("model", settings.model || "");
  const res = await fetch(`${API_BASE}/settings/llm-correction`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * 云端多配置: 整体保存配置列表与激活项。
 * @param {Object} payload - { engines: [{id?, api_url, api_key, model, enabled, request_mode, display_name}], active? }
 */
export async function saveEngineConfigsApi(payload) {
  const res = await fetch(`${API_BASE}/settings/engines`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}