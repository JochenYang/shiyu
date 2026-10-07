"""
豆包录音文件识别 2.0 — 云端 ASR 引擎 (submit/query 两段式 HTTP)。

协议参照火山引擎官方文档 (volcengine.com/docs/6561/1354868) 与 SubPop 的
实测实现 (probes/doubao.py): 音频 base64 wav 直传 (60 秒块), 提交后轮询,
返回 utterances[].words[] 逐词毫秒时间戳 —— 时间精度高于任何本地模型,
且无需 TOS 存储桶, 只需语音服务 API Key。

计费安全: 提交即计费, 因此**绝不自动重试**一次提交; 失败让用户决定。
"""
from __future__ import annotations

import base64
import io
import logging
import math
import time
import wave
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .engine_base import BaseTranscriber, TranscribeResult, Segment

logger = logging.getLogger(__name__)

HOST = "openspeech.bytedance.com"
SUBMIT_PATH = "/api/v3/auc/bigmodel/submit"
QUERY_PATH = "/api/v3/auc/bigmodel/query"
# 豆包用请求头 X-Api-Resource-Id 选模型版本 (官方文档):
#   1.0 = volc.bigasr.auc, 2.0 = volc.seedasr.auc
# 请求体的 model_name 固定为协议常量 bigmodel, 不随版本变。
DEFAULT_RESOURCE_ID = "volc.seedasr.auc"
RESOURCE_IDS = {
    "doubao-asr-2.0": "volc.seedasr.auc",
    "doubao-asr-1.0": "volc.bigasr.auc",
}
MODEL_NAME = "bigmodel"  # 协议常量
RATE = 16000
# 60s 单声道 16bit wav ≈ 1.9MB, base64 后 ≈ 2.6MB —— 官方单请求上限内
CHUNK_SECONDS = 60
QUERY_INTERVAL = 3.0
QUERY_TIMEOUT = 1800.0
# X-Api-Status-Code: 20000000 成功; 20000001 排队; 20000002 处理中;
# 20000003 空音频; 45xxxxxx 鉴权/参数错; 55xxxxxx 服务/限流
_STATUS_OK = "20000000"
_STATUS_PENDING = ("20000001", "20000002")
_STATUS_EMPTY = "20000003"


class DoubaoCloudError(ValueError):
    """豆包云端识别失败 (面向用户的安全消息)。"""


def _chunk_boundaries(samples: np.ndarray) -> List[Tuple[int, int]]:
    """把整段音频切成 ≤60s 的块, 块尾 5 秒内取能量最低点收尾。

    参照 SubPop: 避免在人声中间拦腰截断; 无重叠无间隙, 块偏移即绝对时间。
    """
    chunk = CHUNK_SECONDS * RATE
    bounds: List[Tuple[int, int]] = []
    cursor = 0
    while cursor < len(samples):
        end = min(cursor + chunk, len(samples))
        if end < len(samples):
            # 搜索窗口: 至少留 1s, 到块尾前 0.1s
            lo = max(cursor + RATE, end - 5 * RATE)
            hi = end - quiet
            if hi > lo:
                windows = [samples[i:i + quiet] for i in range(lo, hi, quiet)]
                energies = [float(np.mean(w ** 2)) for w in windows]
                best = lo + int(np.argmin(energies)) * quiet
                end = min(end, best + quiet // 2)
        bounds.append((cursor, end))
        cursor = end
    return bounds

def _pcm_wav_bytes(samples: np.ndarray) -> bytes:
    """float32 [-1,1] 采样 → 16bit PCM WAV 字节。"""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class DoubaoCloudEngine(BaseTranscriber):
    """豆包录音文件识别 2.0 (云端, 按量计费)。"""

    name = "doubao_cloud"
    display_name = "豆包录音文件识别"
    requires_network = True
    is_local = False

    def __init__(self, api_key: str = "", resource_id: str = "", **kwargs) -> None:
        self._api_key = (api_key or "").strip()
        # resource_id 表达模型版本 (1.0/2.0); 未指定时用 2.0
        self._resource_id = resource_id or DEFAULT_RESOURCE_ID

    # ── 配置 ────────────────────────────────────────────────────────

    @staticmethod
    def _config_from_provider() -> Dict[str, Any]:
        """从云端配置里读豆包需要的字段 (KEY + 模型版本)。

        模型版本存在 provider 的 models 清单里 (id → resource_id),
        用户选的 model 字段就是那个 id。
        """
        from .cloud_engine import load_engine_config

        cfg = load_engine_config()
        if cfg.get("provider") != "doubao" and "doubao" not in (cfg.get("model") or ""):
            return {}
        model_id = cfg.get("model") or "doubao-asr-2.0"
        return {
            "api_key": (cfg.get("api_key") or "").strip(),
            "resource_id": RESOURCE_IDS.get(model_id, DEFAULT_RESOURCE_ID),
        }

    def is_available(self) -> Tuple[bool, str]:
        cfg = self._config_from_provider()
        key = self._api_key or cfg.get("api_key", "")
        if not self._resource_id:
            self._resource_id = cfg.get("resource_id", DEFAULT_RESOURCE_ID)
        if not key:
            return False, (
                "豆包云端识别未配置 API Key：请在「设置 → 云端」选择"
                "「豆包录音文件识别」并填入语音服务 API Key"
            )
        return True, ""

    # ── HTTP ────────────────────────────────────────────────────────

    def _post(self, endpoint: str, body: Dict[str, Any], request_id: str) -> Tuple[str, Optional[Dict]]:
        import requests

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self._api_key,
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Request-Id": request_id,
        }
        if endpoint == SUBMIT_PATH:
            headers["X-Api-Sequence"] = "-1"  # 单请求整段音频的固定标记
        try:
            resp = requests.post(
                f"https://{HOST}{endpoint}", json=body, headers=headers, timeout=60
            )
        except requests.RequestException as e:
            raise DoubaoCloudError(
                "豆包连接中断或超时。未自动重试；已提交的音频可能仍会计费。"
            ) from e
        code = resp.headers.get("X-Api-Status-Code", "")
        if resp.status_code in (401, 403) or code in ("45000010", "45000011"):
            raise DoubaoCloudError(
                "豆包鉴权失败：请检查语音服务 API Key，并已开通「录音文件识别 2.0」。"
            )
        if resp.status_code == 429 or code in ("55000031", "45000131"):
            raise DoubaoCloudError("豆包请求受限或服务繁忙，请稍后重试。")
        if resp.status_code != 200:
            raise DoubaoCloudError(
                "豆包服务未成功响应。未自动重试，请到火山引擎控制台检查服务状态与用量。"
            )
        if endpoint == SUBMIT_PATH or code != _STATUS_OK:
            return code, None
        try:
            return code, resp.json()
        except ValueError as e:
            raise DoubaoCloudError("豆包返回了无法解析的识别结果。") from e

    # ── 识别 ────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language: str = "auto", **kwargs) -> TranscribeResult:
        import requests

        ok, reason = self.is_available()
        if not ok:
            raise DoubaoCloudError(reason)
        # is_available() 已把 KEY 与版本补齐, 这里不再重复读配置

        from .audio_processor import extract_audio_for_transcribe
        import soundfile as sf

        wav16k = extract_audio_for_transcribe(Path(audio_path))
        audio, sr = sf.read(wav16k, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=RATE)

        segments: List[Segment] = []
        for start_sample, end_sample in _chunk_boundaries(audio):
            chunk = audio[start_sample:end_sample]
            if chunk.size == 0 or float(np.max(np.abs(chunk))) <= 1e-7:
                continue  # 纯静音块: 不提交, 不计费
            offset = start_sample / RATE
            duration = len(chunk) / RATE
            value = self._transcribe_chunk(chunk, duration)
            segments.extend(self._parse_result(value, duration, offset))

        return TranscribeResult(segments=self._finish(segments))

    def _transcribe_chunk(self, chunk: np.ndarray, duration: float) -> Dict[str, Any]:
        """提交一个 60s 块并轮询到完成。提交即计费, 绝不自动重试。"""
        request_id = str(uuid.uuid4())
        body = {
            "user": {"uid": "shiyu"},
            "audio": {
                "data": base64.b64encode(_pcm_wav_bytes(chunk)).decode("ascii"),
                "format": "wav",
                "rate": RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": MODEL_NAME,  # 协议常量, 版本由 resource_id 头决定
                "show_utterances": True,   # 逐词时间戳必需
                "enable_itn": True,        # 数字归一, 与本地 ITN 一致
                "enable_punc": True,
                "enable_ddc": False,
            },
        }
        self._post(SUBMIT_PATH, body, request_id)
        logger.info("[Doubao] 已提交 %.1fs 音频 (request_id=%s)", duration, request_id)

        deadline = time.monotonic() + QUERY_TIMEOUT
        while time.monotonic() < deadline:
            code, value = self._post(QUERY_PATH, {}, request_id)
            if code == _STATUS_OK:
                return value or {}
            if code == _STATUS_EMPTY:
                return {"result": {"text": "", "utterances": []}}
            if code not in _STATUS_PENDING:
                raise DoubaoCloudError("豆包未返回有效任务状态。")
            time.sleep(QUERY_INTERVAL)
        raise DoubaoCloudError(
            "豆包识别等待超时。已提交部分可能计费，请到控制台确认用量后重试。"
        )

    # ── 结果解析 ────────────────────────────────────────────────────

    @staticmethod
    def _parse_result(value: Dict[str, Any], duration: float, offset: float) -> List[Segment]:
        """豆包 utterances → Segment (逐词时间, 拒绝插值)。"""
        if not isinstance(value, dict) or not isinstance(value.get("result"), dict):
            raise DoubaoCloudError("豆包结果结构不完整。")
        utterances = value["result"].get("utterances")
        if not isinstance(utterances, list):
            raise DoubaoCloudError("豆包未返回逐词时间戳，无法生成准确的字幕时间。")

        segments: List[Segment] = []
        for utterance in utterances:
            text = (utterance or {}).get("text")
            words = (utterance or {}).get("words")
            if not isinstance(text, str) or not isinstance(words, list):
                raise DoubaoCloudError("豆包句子数据不完整。")
            if not text.strip() and not words:
                continue
            if not words:
                raise DoubaoCloudError("豆包未返回逐词时间戳，无法生成准确的字幕时间。")

            word_rows: List[Tuple[str, float, float]] = []
            previous = -1.0
            for word in words:
                w_text = (word or {}).get("text")
                w_start, w_end = (word or {}).get("start_time"), (word or {}).get("end_time")
                if not isinstance(w_text, str):
                    raise DoubaoCloudError("豆包词语数据不完整。")
                if (
                    not isinstance(w_start, (int, float))
                    or not isinstance(w_end, (int, float))
                    or not (math.isfinite(w_start) and math.isfinite(w_end))
                    or not 0 <= w_start <= w_end <= duration * 1000 + 50
                    or w_start < previous
                ):
                    raise DoubaoCloudError("豆包返回的时间戳异常，已停止生成字幕。")
                previous = w_start
                word_rows.append(
                    (w_text.strip(), offset + min(w_start / 1000, duration),
                     offset + min(w_end / 1000, duration))
                )

            # 逐词时间 → Segment + char_times (字符继承所属词 start,
            # 与 forced_aligner.char_times_for 同一契约); 行末取末词 end,
            # 豆包给了真实词尾 —— 比本地 CTC 只有起点更好。
            full_text = "".join(w[0] for w in word_rows)
            if not full_text.strip():
                continue
            char_times: List[float] = []
            last_end = offset + duration
            for w_text, w_start, w_end in word_rows:
                for _ch in w_text:
                    char_times.append(w_start)
                last_end = w_end
            segments.append(
                Segment(
                    start=round(char_times[0], 3),
                    end=round(last_end, 3),
                    text=full_text,
                    char_times=char_times,
                )
            )
        return segments
