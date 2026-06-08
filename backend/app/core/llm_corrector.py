"""
LLM-based text correction for ASR output.

Supports two provider types:
- "openai": Any OpenAI chat-completions compatible API (OpenAI, 火山引擎, 智谱, MiniMax, DeepSeek, DashScope, etc.)
- "ollama": Local Ollama instance (OpenAI-compatible endpoint)

All providers use the same /chat/completions endpoint format.
Falls back to original text on any error.
"""
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from app.core import text_splitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORRECTION_SYSTEM_PROMPT = """你是语音识别文本修正器。你的工作是：对每行文本修正错别字和断词，然后原样输出。

输入输出格式示例：
输入：
1. 今天天气真好我们一
2. 起去公园散步。
输出：
今天天气真好，我们一
起去公园散步。

绝对禁止的行为（违反将导致输出被丢弃）：
- 禁止添加任何分析、注释、说明、引号、标记
- 禁止在行首添加"被截断"、"注意"、"这里"、"需要"等词
- 禁止改写句式、补充内容、重新组织语言、润色文字
- 禁止在原文基础上加任何词（包括"的"、"了"、"呢"、"吗"等语气词）
- 禁止删除原文中的任何词
- 禁止输出空行
- 禁止输出任何文字以外的内容

必须遵守的规则：
1. 输出行数必须等于输入行数，每行对应输入的相同序号行
2. 只修正错别字、断词、缺失标点
3. 不改变原意和事实内容
4. 如果某行原文正确，原样输出（一个字都不要改）
5. 被截断的半句话不要加「…」前缀或后缀
6. 如果行以完整句子结尾，补句号；如果行明显被截断，末尾不加句号
7. 遇到"en"+"gine"这种断词，在各自行内修正（不要跨行合并）"""

DEFAULT_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Think block stripper (通用：适用于所有支持/不支持关闭思考的模型)
# ---------------------------------------------------------------------------


def _strip_thinking_blocks(text: str) -> str:
    """从 LLM 响应中剥离思维链/推理块。

    处理的模式：
    - <think>...</think>（DeepSeek / MiniMax 等）
    - 行首的标记性思维行（如 "好的"，"我来分析"）
    - 以 "### " 或 "**" 开头的 markdown 分析

    多轮剥离直到不再变化，确保递归嵌套也能清干净。
    """
    if not text:
        return text

    prev = None
    result = text
    while result != prev:
        prev = result
        # 1. <think>...</think>（多行块）
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        # 2. 单行 <think>...</think>（同一行内）
        result = re.sub(r'<think>.*?</think>', '', result)
        # 3. 遗落的 <think> 或 </think> 标签
        result = result.replace('<think>', '').replace('</think>', '')

    return result.strip()


# ---------------------------------------------------------------------------
# 智能修正正文抽取
# ---------------------------------------------------------------------------


def _extract_correction_block(
    lines: List[str],
    expected_count: int,
) -> Optional[List[str]]:
    """从过长的行列表中智能抽取出修正正文。

    当 LLM 输出了分析+修正的混合内容时，修正正文通常在最后。
    尝试从末尾往前找到 *expected_count* 行连续、格式正确的修正行。

    Args:
        lines: 清理后的行列表（已去除 think 块和空行）。
        expected_count: 期望的行数。

    Returns:
        如果找到符合条件的块则返回该块，否则返回 None。
    """
    if not lines:
        return None

    # 策略：从最后一行开始向前扫描，找连续 expected_count 行
    for start in range(len(lines) - expected_count, -1, -1):
        block = lines[start:start + expected_count]

        # 检查候选块是否全部都是"修正风格"（无分析标记）
        looks_like_analysis = 0
        for line in block:
            stripped = line.strip()
            # 分析标记检测（保留与输入相近的文本）
            if not stripped:
                looks_like_analysis += 1
            # Markdown 标题 → 分析
            if re.match(r'^(#{1,6}|-{3,}|={3,})', stripped):
                looks_like_analysis += 2
            # 以"原因"、"分析"、"首先"、"总结"等开头的句子
            if re.match(r'^(原因|分析|首先|其次|总结|注意|提示|解释|说明)[：:]', stripped):
                looks_like_analysis += 2

        # 如果候选块的分析分 <= expected_count * 0.5，认为是有效修正
        if looks_like_analysis <= expected_count * 0.5:
            return block

    return None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CorrectionConfig:
    """Configuration for LLM-based text correction.

    Attributes:
        enabled: Whether correction is enabled.
        provider: Provider name ("openai" | "ollama").
        api_url: Base URL for the API (without /chat/completions suffix).
        api_key: API key for authentication (leave empty for Ollama).
        model: Model name override (provider-specific default if empty).
    """
    enabled: bool = False
    provider: str = "openai"
    api_url: str = ""
    api_key: str = ""
    model: str = ""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_correction_prompt(segment_texts: List[str]) -> str:
    """Build the user prompt for the correction LLM call.

    Formats each text as a numbered line for easy reference,
    with explicit output format instructions.
    """
    lines = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(segment_texts))
    return (
        f"严格修正以下 {len(segment_texts)} 行语音识别错别字和断词。\n\n"
        f"规则（违反将导致输出被丢弃）：\n"
        f"1. 输出严格 {len(segment_texts)} 行，每行对应输入的同序号\n"
        f"2. 只修正错别字和断词，不改句式、不加词、不减词\n"
        f"3. 原文正确就原样输出，一个字都不要改\n"
        f"4. 不要分析、不要注释、不要加引号、不要加标记\n"
        f"5. 禁止在原文基础上添加任何话术或润色\n\n"
        f"{lines}"
    )


def _parse_correction_response(
    response_text: str,
    expected_count: int,
    original_texts: List[str],
) -> List[str]:
    """Parse LLM response into a list of corrected texts.

    Three-tier strategy:
      1. Strip <think> blocks and reasoning artifacts
      2. Try standard format: numbered lines, plain lines, quoted lines
      3. If excess lines remain → try smart extraction of correction block

    If all parsing fails, falls back to *original_texts* (silently).

    Args:
        response_text: Raw response from the LLM.
        expected_count: Number of texts expected.
        original_texts: Fallback texts when parsing fails.

    Returns:
        List of corrected texts (or original_texts on failure).
    """
    if not response_text or not response_text.strip():
        return original_texts

    # ── Step 0: Strip thinking / reasoning blocks ──
    cleaned_text = _strip_thinking_blocks(response_text)
    if not cleaned_text:
        logger.warning("LLM response was all thinking blocks, falling back.")
        return original_texts

    raw_lines = cleaned_text.splitlines()

    # ── Step 1: clean each line ──
    cleaned: List[str] = []
    for line in raw_lines:
        ln = line.strip()
        if not ln:
            continue
        # Strip leading bullet markers: "- ", "* ", "• "
        ln = re.sub(r"^[-*•]\s+", "", ln)
        # Strip leading number prefix: "1. ", "1、", "1:", "1）", "1) "
        ln = re.sub(r"^\d+\s*[.、:）)]?\s*", "", ln)
        # Strip surrounding double quotes
        ln = re.sub(r'^"|"$', "", ln)
        # Strip surrounding 「」
        ln = re.sub(r"^「|」$", "", ln)
        cleaned.append(ln)

    # ── Step 2: try exact match ──
    if len(cleaned) == expected_count:
        return cleaned

    # Try raw lines (before cleaning) as fallback
    raw_stripped = [ln.strip() for ln in raw_lines if ln.strip()]
    if len(raw_stripped) == expected_count:
        return raw_stripped

    # ── Step 3: too many lines → try smart extraction ──
    if len(cleaned) > expected_count:
        extracted = _extract_correction_block(cleaned, expected_count)
        if extracted:
            logger.info(
                "LLM response had %d lines (expected %d) → "
                "smart extraction found correction block at end",
                len(cleaned), expected_count,
            )
            return extracted

        extracted_raw = _extract_correction_block(raw_stripped, expected_count)
        if extracted_raw:
            logger.info(
                "LLM response had %d lines (expected %d) → "
                "smart extraction (raw) found correction block",
                len(cleaned), expected_count,
            )
            return extracted_raw

        logger.warning(
            "LLM correction extra lines: expected %d, got %d (cleaned: %d). "
            "Smart extraction failed — falling back to original.",
            expected_count, len(raw_lines), len(cleaned),
        )
        return original_texts

    # ── Step 4: too few lines — return original ──
    logger.warning(
        "LLM correction response line count mismatch: "
        "expected %d, got %d (cleaned: %d). Falling back to original.",
        expected_count,
        len(raw_lines),
        len(cleaned),
    )
    return original_texts


# ---------------------------------------------------------------------------
# Base corrector
# ---------------------------------------------------------------------------


class BaseCorrector(ABC):
    """Abstract base for LLM-based ASR text correctors."""

    # Override in subclasses
    provider: str = ""

    @abstractmethod
    def correct(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correct text in each segment.

        Args:
            segments: List of dicts with keys "start", "end", "text".

        Returns:
            Same-length list with only ``text`` keys possibly modified.
        """
        ...

    # ------------------------------------------------------------------
    # Shared batch logic
    # ------------------------------------------------------------------

    def _batch_correct(
        self,
        segments: List[Dict[str, Any]],
        api_url: str,
        api_key: str,
        model: str,
    ) -> List[Dict[str, Any]]:
        """Shared batch correction workflow used by all subclasses.

        Splits *segments* into batches of *DEFAULT_BATCH_SIZE*,
        sends each batch to the LLM, parses the response, and
        replaces the ``text`` field in-place.

        Args:
            segments: Input segments to correct.
            api_url: Full chat-completions URL.
            api_key: API key (may be empty for local providers).
            model: Model identifier.

        Returns:
            Corrected segments (same length as input).
        """
        if not segments:
            return []

        if not model:
            logger.warning("LLM correction skipped: model is empty (configure it in settings)")
            return segments

        result = [dict(seg) for seg in segments]  # shallow copy to avoid mutating input

        # Build batch list
        batches = []
        for batch_start in range(0, len(result), DEFAULT_BATCH_SIZE):
            batch_slice = result[batch_start : batch_start + DEFAULT_BATCH_SIZE]
            original_texts = [seg["text"] for seg in batch_slice]
            prompt_text = _build_correction_prompt(original_texts)
            batches.append((batch_start, batch_slice, original_texts, prompt_text))

        # Process batches in parallel (up to 3 concurrent requests to avoid rate limiting)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for batch_start, batch_slice, original_texts, prompt_text in batches:
                fut = executor.submit(
                    self._process_single_batch,
                    api_url, api_key, model, prompt_text, original_texts,
                )
                futures[fut] = (batch_start, batch_slice)

            for fut in concurrent.futures.as_completed(futures):
                batch_start, batch_slice = futures[fut]
                try:
                    corrected = fut.result()
                except Exception as e:
                    logger.warning("Batch correction failed (batch starting at %d): %s", batch_start, e)
                    continue
                if corrected is not None:
                    for seg, new_text in zip(batch_slice, corrected):
                        seg["text"] = new_text

        return result

    def _process_single_batch(self, api_url, api_key, model, prompt_text, original_texts):
        """Process a single batch: call LLM and parse response."""
        response_text = self._call_llm(api_url, api_key, model, prompt_text)
        if response_text is None:
            return None
        _resp_lines = response_text.strip().splitlines()
        logger.info(
            "LLM returned %d lines (expected %d), first line: %.100s",
            len(_resp_lines), len(original_texts),
            _resp_lines[0] if _resp_lines else "(empty)",
        )
        return _parse_correction_response(response_text, len(original_texts), original_texts)

    # ------------------------------------------------------------------
    # Shared HTTP call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt_text: str,
    ) -> Optional[str]:
        """Call the OpenAI-compatible chat completions API.

        Args:
            api_url: Full chat-completions endpoint URL.
            api_key: API key (may be empty for local providers).
            model: Model name.
            prompt_text: User prompt content.

        Returns:
            Response text content on success, or ``None`` on failure.
        """
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.1,  # low temperature for deterministic correction
            # NOTE: 不再传 `thinking: False` — MiniMax 服务端把它当作 ThinkingConfig 对象
            # 期望(bool 类型不匹配, 直接返回 400 bad_request_error)。
            # 客户端 _strip_thinking_blocks 已经能剥离 <think>...</think> 块, 不需要服务端关闭。
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            logger.warning("LLM correction request timed out (url=%s, model=%s)", api_url, model)
            return None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "N/A"
            logger.warning(
                "LLM correction HTTP error (url=%s, model=%s, status=%s): %s",
                api_url, model, status, exc,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "LLM correction request failed (url=%s, model=%s): %s",
                api_url, model, exc,
            )
            return None
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "LLM correction response parse error (url=%s, model=%s): %s",
                api_url, model, exc,
            )
            return None

        # Extract content from standard OpenAI response format
        try:
            content = data["choices"][0]["message"]["content"]
            if content:
                content = _strip_thinking_blocks(content)
                return content.strip() if content else None
            return None
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "LLM correction response has unexpected format (url=%s, model=%s): %s",
                api_url, model, exc,
            )
            return None


# ---------------------------------------------------------------------------
# OpenAI corrector
# ---------------------------------------------------------------------------


class OpenAICorrector(BaseCorrector):
    """Corrector for any OpenAI chat-completions compatible API.

    Covers: OpenAI, 火山引擎, 智谱 GLM, MiniMax, DeepSeek, DashScope 等.
    All use the same ``POST {base_url}/chat/completions`` endpoint format.
    """

    provider: str = "openai"

    # Default model - empty, user must configure their own model ID
    _DEFAULT_MODEL = ""
    # Default base URL (without /chat/completions suffix)
    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def correct(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._batch_correct(
            segments,
            api_url=self._api_url,
            api_key=self._api_key,
            model=self._model,
        )

    def __init__(self, config: CorrectionConfig) -> None:
        self._api_key = config.api_key
        base = config.api_url if config.api_url else self._DEFAULT_BASE_URL
        self._api_url = f"{base.rstrip('/')}/chat/completions"
        self._model = config.model if config.model else self._DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Ollama corrector
# ---------------------------------------------------------------------------


class OllamaCorrector(BaseCorrector):
    """Corrector using a local Ollama instance (OpenAI-compatible endpoint)."""

    provider: str = "ollama"

    _DEFAULT_MODEL = ""
    _DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def correct(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._batch_correct(
            segments,
            api_url=self._api_url,
            api_key=self._api_key,
            model=self._model,
        )

    def __init__(self, config: CorrectionConfig) -> None:
        self._api_key = config.api_key
        base = config.api_url if config.api_url else self._DEFAULT_BASE_URL
        self._api_url = f"{base.rstrip('/')}/chat/completions"
        self._model = config.model if config.model else self._DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Full-text corrector (P2: merge → LLM correct → re-split)
# ---------------------------------------------------------------------------

FULLTEXT_SYSTEM_PROMPT = """你是一个语音识别结果修正工具。你没有思维过程，没有解释，只有输出。

输入是一段语音识别输出的连续文本，可能存在因时间窗口切割导致的词语断裂。
输出必须是修正后的完整文本，保持整体结构不变。

修正规则：
1. 去除因时间窗口切割产生的多余空格，合并断词（如"今天天气真 好" → "今天天气真好，"）
2. 补全缺失的标点符号（句号、逗号、问号等）
3. 修正明显的错别字和识别错误
4. 不改变原意和事实内容
5. 不得添加或删除任何实质性内容（不要总结、不要补充、不要改写句式）
6. 如果有空行，保持分段结构不变
7. 禁止输出任何额外内容：不要思考过程、不要解释、不要备注

你的整个回复必须只包含修正后的文本。这不是对话，是严格的文本转换。"""


def _split_text_into_segments(
    corrected_text: str,
    original_segments: list,
    max_chars: int = 50,
) -> list:
    """Re-split corrected full text into segments with anchored timestamps.

    Splits by sentence-ending punctuation (skipping decimal points like "0.5"),
    then merges short sentences into segments respecting *max_chars*.

    Timestamps are assigned by mapping each new segment's character range
    back to the ORIGINAL segments' positions — this avoids the drift that
    proportional global distribution causes when original segments have
    overlapping timestamps (energy-dip chunking artifact).

    Args:
        corrected_text: The LLM-corrected full text.
        original_segments: Original segments with [start, end] for timing.
        max_chars: Max characters per output segment.

    Returns:
        New segments list with [start, end, text].
    """
    if not corrected_text or not corrected_text.strip():
        return list(original_segments)

    # -- Step 0: Build original char→segment index map --
    orig_text = " ".join(s["text"].strip() for s in original_segments if s.get("text"))
    if not orig_text.strip():
        return list(original_segments)

    orig_char_to_seg: list[int] = []
    for idx, seg in enumerate(original_segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        for _ in text:
            orig_char_to_seg.append(idx)
        # The space between segments belongs to the preceding segment
        orig_char_to_seg.append(idx)
    total_orig_chars = len(orig_char_to_seg)

    # -- Step 1: Split into sentences by terminal punctuation --
    # Terminal punctuation, but SKIP decimal points (e.g., "0.5")
    sentences: list[str] = []
    buf_chars: list[str] = []
    for i, ch in enumerate(corrected_text):
        buf_chars.append(ch)
        if ch in text_splitter.TERMINAL_PUNCT:
            # Skip decimal point: period between two digits
            if ch == '.' and i > 0 and i < len(corrected_text) - 1:
                if corrected_text[i - 1].isdigit() and corrected_text[i + 1].isdigit():
                    continue  # It's a decimal point, not a sentence end
            sentence = "".join(buf_chars).strip()
            if sentence:
                sentences.append(sentence)
            buf_chars = []
    remaining = "".join(buf_chars).strip()
    if remaining:
        sentences.append(remaining)

    if not sentences:
        return list(original_segments)

    # -- Step 2: Merge short sentences into segments --
    merged_texts: list[str] = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + len(s) <= max_chars:
            cur += s
        else:
            merged_texts.append(cur)
            cur = s
    if cur:
        merged_texts.append(cur)

    if not merged_texts:
        return list(original_segments)

    # -- Step 3: Anchored timestamp assignment --
    # Map each new segment's character position in corrected text to
    # a position in the original concatenated text, then find which
    # original segment(s) that range covers and use their timestamps.
    total_corrected_chars = sum(len(t) for t in merged_texts)

    new_segments: list[dict] = []
    cum_corrected = 0
    for text in merged_texts:
        text_len = len(text)
        # Map corrected char range to original text position (ratio)
        end_ratio = (cum_corrected + text_len) / total_corrected_chars
        start_ratio = cum_corrected / total_corrected_chars

        orig_start_pos = int(start_ratio * total_orig_chars)
        orig_end_pos = min(int(end_ratio * total_orig_chars), total_orig_chars - 1)

        start_seg_idx = orig_char_to_seg[orig_start_pos]
        end_seg_idx = orig_char_to_seg[orig_end_pos]

        seg_start = original_segments[start_seg_idx]["start"]
        seg_end = original_segments[end_seg_idx]["end"]

        new_segments.append({
            "start": round(seg_start, 3),
            "end": round(seg_end, 3) if seg_end >= seg_start else round(seg_start + 0.5, 3),
            "text": text,
        })
        cum_corrected += text_len

    return new_segments


class FullTextCorrector(BaseCorrector):
    """Merge all segments → LLM corrects as full text → re-split by sentences.

    Unlike batch per-segment correction, this approach gives the LLM full
    context, allowing it to fix inter-segment word breaks and produce
    natural sentence boundaries.
    """

    provider: str = "fulltext_openai"

    _SYSTEM_PROMPT = FULLTEXT_SYSTEM_PROMPT

    def __init__(self, config: CorrectionConfig) -> None:
        self._api_key = config.api_key
        base = config.api_url if config.api_url else "https://api.openai.com/v1"
        self._api_url = f"{base.rstrip('/')}/chat/completions"
        self._model = config.model if config.model else ""

    def correct(self, segments: list) -> list:
        """Merge → LLM correct → re-split with proportional timestamps."""
        if not segments:
            return []

        if not self._model:
            logger.warning("Full-text correction skipped: model is empty")
            return list(segments)

        # 1. Merge all segment texts into one continuous text
        full_text = " ".join(seg["text"].strip() for seg in segments if seg.get("text"))

        if not full_text.strip():
            return list(segments)

        logger.info(
            "Full-text correction: %d segments, %d chars total",
            len(segments), len(full_text),
        )

        # 2. Call LLM with full text
        response_text = self._call_llm(
            self._api_url, self._api_key, self._model, full_text,
        )

        if not response_text:
            logger.warning("Full-text correction returned empty response, falling back")
            return list(segments)

        # 3. Re-split corrected text
        new_segments = _split_text_into_segments(response_text, segments)

        logger.info(
            "Full-text correction done: %d chars → %d new segments",
            len(response_text), len(new_segments),
        )
        return new_segments

    def _call_llm(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt_text: str,
    ) -> Optional[str]:
        """Call LLM with the full-text system prompt."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.1,
            "thinking": False,
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Full-text LLM call failed: %s", e)
            return None

        try:
            content = data["choices"][0]["message"]["content"]
            if content:
                content = _strip_thinking_blocks(content)
                return content.strip() if content else None
            return None
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Full-text LLM response parse error: %s", e)
            return None


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_MAP: Dict[str, type[BaseCorrector]] = {
    "openai": OpenAICorrector,   # 通用 OpenAI 兼容接口（火山/智谱/MiniMax/DeepSeek/DashScope 等）
    "ollama": OllamaCorrector,   # 本地 Ollama
}

_FULLTEXT_PROVIDER_MAP: Dict[str, type[BaseCorrector]] = {
    "openai": FullTextCorrector,
}


def create_corrector(config: CorrectionConfig) -> Optional[BaseCorrector]:
    """Factory: create a corrector instance based on configuration.

    Supported providers:
    - "openai": OpenAI 兼容接口（火山引擎/智谱/MiniMax/DeepSeek/DashScope/OpenAI 等）
    - "ollama": 本地 Ollama

    Args:
        config: Correction configuration.

    Returns:
        A corrector instance, or ``None`` when correction is disabled
        or the provider is unrecognised / empty.
    """
    if not config.enabled:
        logger.debug("LLM correction is disabled.")
        return None

    provider_key = config.provider.strip().lower() if config.provider else ""
    if not provider_key:
        logger.debug("LLM correction provider is empty; skipping.")
        return None

    corrector_cls = _PROVIDER_MAP.get(provider_key)
    if corrector_cls is None:
        logger.warning(
            "Unknown LLM correction provider '%s'; "
            "supported: %s. Skipping.",
            config.provider,
            ", ".join(sorted(_PROVIDER_MAP)),
        )
        return None

    return corrector_cls(config)


def create_fulltext_corrector(config: CorrectionConfig) -> Optional[BaseCorrector]:
    """Factory: create a full-text corrector instance.

    Unlike ``create_corrector`` which corrects segments in batches,
    the full-text corrector merges all texts → LLM corrects → re-splits
    by sentence boundaries with proportional timestamps.

    Args:
        config: Correction configuration.

    Returns:
        A FullTextCorrector instance, or ``None`` when disabled.
    """
    if not config.enabled:
        return None

    provider_key = config.provider.strip().lower() if config.provider else ""
    if not provider_key:
        return None

    corrector_cls = _FULLTEXT_PROVIDER_MAP.get(provider_key)
    if corrector_cls is None:
        logger.warning(
            "Unknown full-text correction provider '%s'; supported: %s.",
            config.provider,
            ", ".join(sorted(_FULLTEXT_PROVIDER_MAP)),
        )
        return None

    return corrector_cls(config)
