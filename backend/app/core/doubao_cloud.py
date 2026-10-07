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
    quiet = RATE // 10          # 100ms 一个候选窗
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

        # 与 cloud_engine 一致: 直接用 librosa 读音频并重采样到 16kHz。
        # (不依赖 main.py 的私有辅助函数 —— 这里曾因导入不存在的
        #  extract_audio_for_transcribe 导致整个豆包引擎报错。)
        import librosa

        try:
            audio, sr = librosa.load(audio_path, sr=RATE, mono=True)
        except Exception as e:
            # 上层应已把视频抽成音频; 读不了就明确报错而不是抛原始异常
            raise DoubaoCloudError(
                f"无法读取音频文件: {type(e).__name__}。请确认文件是音频,"
                "或已安装 ffmpeg 以从视频中抽取音轨。"
            ) from e
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise DoubaoCloudError("音频为空, 无内容可识别。")
        segments: List[Segment] = []
        for start_sample, end_sample in _chunk_boundaries(audio):
            chunk = audio[start_sample:end_sample]
            if chunk.size == 0 or float(np.max(np.abs(chunk))) <= 1e-7:
                continue  # 纯静音块: 不提交, 不计费
            offset = start_sample / RATE
            duration = len(chunk) / RATE
            value = self._transcribe_chunk(chunk, duration)
            # 诊断: 记录豆包原始返回的时间范围。真实素材出现时间戳越界时,
            # 光看异常消息无法判断是基准差异还是单个词异常。
            try:
                _utts = (value or {}).get("result", {}).get("utterances") or []
                _ends = [w.get("end_time", 0) for u in _utts
                         for w in (u.get("words") or [])]
                _starts = [w.get("start_time", 0) for u in _utts
                           for w in (u.get("words") or [])]
                logger.info(
                    "[Doubao] 块 offset=%.2fs 时长=%.2fs: %d 句 %d 词, "
                    "start 范围 [%s, %s]ms, end 范围 [%s, %s]ms",
                    offset, duration, len(_utts), len(_ends),
                    min(_starts) if _starts else "-", max(_starts) if _starts else "-",
                    min(_ends) if _ends else "-", max(_ends) if _ends else "-",
                )
            except Exception:
                pass
            segments.extend(self._parse_result(value, duration, offset))
        # 豆包已给出真实词尾时间, 不需要 sherpa 的 _finish (那是 CTC 后处理);
        # 段已按块顺序生成, 时间天然递增。
        return TranscribeResult(segments=segments, language=language)

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

            # 收集词。豆包对空格/标点这类不发声的词返回 start=-1 end=-1,
            # 这是"无时间"的合法信号而不是异常 —— 直接报错会让英文素材
            # 整体失败 (英文词间都是空格)。这里先记 None, 后面回填。
            word_rows: List[Tuple[str, Optional[float], Optional[float]]] = []
            previous: Optional[float] = None
            for word in words:
                w_text = (word or {}).get("text")
                w_start, w_end = (word or {}).get("start_time"), (word or {}).get("end_time")
                if not isinstance(w_text, str):
                    raise DoubaoCloudError("豆包词语数据不完整。")
                if w_start == -1 or w_end == -1:
                    word_rows.append((w_text, None, None))
                    continue
                if (
                    not isinstance(w_start, (int, float))
                    or not isinstance(w_end, (int, float))
                    or not (math.isfinite(w_start) and math.isfinite(w_end))
                    or not 0 <= w_start <= w_end <= duration * 1000 + 50
                    or (previous is not None and w_start < previous)
                ):
                    # 报出具体数值, 否则无法定位是哪一条越界
                    raise DoubaoCloudError(
                        f"豆包返回的时间戳异常（词={w_text!r} "
                        f"start={w_start} end={w_end}ms，"
                        f"本块时长={duration:.2f}s 上限={duration * 1000 + 50:.0f}ms，"
                        f"上一词 start={previous}ms）"
                    )
                previous = w_start
                word_rows.append((w_text, w_start, w_end))

            if not word_rows:
                continue
            # 无时间的词继承前一个有时间词; 开头的则用第一个有时间词
            first_start = next((s for _, s, _ in word_rows if s is not None), None)
            if first_start is None:
                continue  # 整句都没有时间, 无法定位

            prev_time = first_start
            last_end_ms = first_start
            chars: List[str] = []
            times: List[float] = []
            for w_text, w_start, w_end in word_rows:
                if w_start is None:
                    t = prev_time
                else:
                    t = w_start
                    prev_time = w_start
                    last_end_ms = w_end
                for ch in w_text:
                    chars.append(ch)
                    times.append(t)

            # 去掉首尾空白, 时间同步裁剪 (英文素材首尾常带空格);
            # 保留中间的空白字符 —— 否则英文词会粘连成一片。
            lo_i, hi_i = 0, len(chars)
            while lo_i < hi_i and chars[lo_i].isspace():
                lo_i += 1
            while hi_i > lo_i and chars[hi_i - 1].isspace():
                hi_i -= 1
            if lo_i >= hi_i:
                continue

            full_text = "".join(chars[lo_i:hi_i])
            char_times = [
                round(offset + min(t / 1000, duration), 3) for t in times[lo_i:hi_i]
            ]
            segments.append(
                Segment(
                    start=char_times[0],
                    end=round(offset + min(last_end_ms / 1000, duration), 3),
                    text=full_text,
                    char_times=char_times,
                )
            )
        return segments
