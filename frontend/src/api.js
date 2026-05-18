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
export async function transcribeJson(file, language = "auto") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("language", language);
  if (arguments.length > 2 && arguments[2]) {
      formData.append("glossary", arguments[2]);
  } else {
      formData.append("glossary", "{}");
  }

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