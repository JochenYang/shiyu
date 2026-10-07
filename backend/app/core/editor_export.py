"""
剪辑软件工程导出 — 把字幕行写成各剪辑软件可直接导入的工程文件。

导出格式的设计目标 (与 SRT 的区别):
  * 每行字幕是**独立轨道片段**, 用户可单独位移、加效果、改样式 —— 这正是
    「分块」的意义: 逐句包装而不是一整条文本轨。
  * 时间用剪辑软件的原生单位 (FCPXML 用有理数秒/帧, 剪映用微秒),
    避免毫秒取整误差随长度累积。

支持:
  * FCPXML  — Final Cut Pro (Title 片段, 每句可独立加效果)
  * 剪映草稿 — draft_content.json (文本轨道, 每句独立 segment)
  * EDL     — Premiere / DaVinci 通用 (CMX3600, 最保守的兼容格式)
"""
from __future__ import annotations

import json
import uuid
from fractions import Fraction
from typing import Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET

# FCP 基本字幕效果的 UID (官方 Basic Title.moti)
FCP_BASIC_TITLE_UID = (
    ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"
)
DEFAULT_FPS = 25


def _fps_fraction(fps: float) -> Fraction:
    """帧率 → 有理数 (23.976/29.97 这类非整数帧率也要精确)。"""
    common = {
        23.976: Fraction(24000, 1001),
        29.97: Fraction(30000, 1001),
        59.94: Fraction(60000, 1001),
    }
    for value, frac in common.items():
        if abs(fps - value) < 0.001:
            return frac
    return Fraction(int(round(fps)), 1)


def _frames(seconds: float, fps: Fraction) -> int:
    """秒 → 帧号 (四舍五入到最近帧, 避免累积漂移)。"""
    return int(round(Fraction(seconds).limit_denominator(100000) * fps))


# ── FCPXML ──────────────────────────────────────────────────────────


def generate_fcpxml(
    segments: Sequence[Dict],
    *,
    fps: float = DEFAULT_FPS,
    width: int = 1920,
    height: int = 1080,
    title: str = "时语字幕",
    font: str = "PingFang SC",
    font_size: float = 48,
) -> str:
    """生成 FCPXML: 每个字幕行是一个独立 title 片段。

    每句独立成片段意味着用户可以在 FCP 时间线上单独选中任意一句做
    包装 (缩放、描边、位移、动效), 而不是共享一条文本轨。
    """
    frame = _fps_fraction(fps)
    if not segments:
        raise ValueError("没有字幕内容可导出")

    rows: List[Dict] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = _frames(float(seg.get("start", 0.0)), frame)
        end = _frames(float(seg.get("end", 0.0)), frame)
        if end <= start:
            end = start + 1  # 至少一帧, 剪辑软件不接受零长片段
        rows.append({"text": text, "start": start, "end": end})
    if not rows:
        raise ValueError("没有字幕内容可导出")

    total = rows[-1]["end"]

    def t(frames: int) -> str:
        return f"{frames}/{frame.numerator}s" if frame.denominator == 1 else f"{frames * frame.numerator}/{frame.denominator}s"

    root = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources, "format", id="r1", name="FFVideoFormatRateUndefined",
        frameDuration=f"{frame.numerator}/{frame.denominator}s",
        width=str(width), height=str(height), colorSpace="1-1-1 (Rec. 709)",
    )
    ET.SubElement(resources, "effect", id="r2", name="基本字幕", uid=FCP_BASIC_TITLE_UID)

    clip = ET.SubElement(
        root, "clip", name=title, format="r1", start="0s",
        duration=t(total), tcFormat="NDF",
    )
    spine = ET.SubElement(clip, "spine")
    cursor = 0
    for i, row in enumerate(rows, 1):
        if row["start"] > cursor:
            ET.SubElement(
                spine, "gap", name="Gap", offset=t(cursor), start="0s",
                duration=t(row["start"] - cursor),
            )
        node = ET.SubElement(
            spine, "title", ref="r2", offset=t(row["start"]),
            name=row["text"], start="0s", duration=t(row["end"] - row["start"]),
        )
        ET.SubElement(ET.SubElement(node, "text"), "text-style", ref=f"ts{i}").text = row["text"]
        style = ET.SubElement(node, "text-style-def", id=f"ts{i}")
        ET.SubElement(
            style, "text-style", font=font, fontSize=str(font_size),
            fontFace="Regular", fontColor="1 1 1 1", alignment="center",
        )
        cursor = row["end"]
    if cursor < total:
        ET.SubElement(
            spine, "gap", name="Gap", offset=t(cursor), start="0s",
            duration=t(total - cursor),
        )
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + xml


# ── 剪映草稿 ────────────────────────────────────────────────────────


def generate_jianying_draft(
    segments: Sequence[Dict],
    *,
    fps: float = DEFAULT_FPS,
    width: int = 1920,
    height: int = 1080,
    duration_sec: Optional[float] = None,
) -> str:
    """生成剪映 draft_content.json: 文本轨道上每句一个独立 segment。

    剪映时间单位是**微秒**; 每条文本 segment 挂一个 text material。
    导入方式: 新建草稿目录, 把本文件放进去 (剪映 5.x 的 draft_content.json)。
    """
    us = 1_000_000
    frame = _fps_fraction(fps)
    materials: List[Dict] = []
    tracks_segments: List[Dict] = []

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = int(round(float(seg.get("start", 0.0)) * us))
        end = int(round(float(seg.get("end", 0.0)) * us))
        if end <= start:
            end = start + us // int(frame) if frame else start + us // 25
        mid = uuid.uuid4().hex.upper()
        materials.append({
            "id": mid,
            "type": "text",
            "content": json.dumps({
                "text": text,
                "styles": [{"fill": {"content": {"solid": {"color": [1, 1, 1]}}}}],
            }, ensure_ascii=False),
            "font_size": 8.0,
            "text_color": "#FFFFFF",
            "alignment": 1,
            "letter_spacing": 0.0,
            "line_spacing": 0.02,
            "typesetting": 0,
        })
        tracks_segments.append({
            "id": uuid.uuid4().hex.upper(),
            "material_id": mid,
            # 剪映的渲染目标区 (画布内文本位置, 首帧居中偏下)
            "render_index": 0,
            "source_timerange": {"start": 0, "duration": end - start},
            "target_timerange": {"start": start, "duration": end - start},
            "extra_material_refs": [],
            "clip": {"scale": {"x": 1.0, "y": 1.0},
                     "transform": {"x": 0.0, "y": -0.8},
                     "rotation": 0.0,
                     "flip": {"horizontal": False, "vertical": False}},
            "visible": True,
            "volume": 1.0,
            "speed": 1.0,
            "is_placeholder": False,
            "reverse": False,
            "intensifies_audio": False,
            "last_nonzero_volume": 1.0,
        })

    if not tracks_segments:
        raise ValueError("没有字幕内容可导出")

    total_us = int(round((duration_sec if duration_sec else tracks_segments[-1]["target_timerange"]["start"] / us + tracks_segments[-1]["target_timerange"]["duration"] / us) * us))

    draft: Dict = {
        "draft_id": uuid.uuid4().hex.upper(),
        "draft_name": "时语字幕",
        "draft_fold_path": "",
        "draft_root_path": "",
        "draft_removable_storage_device": "",
        "draft_timeline_materials_size": 0,
        "draft_cover": "",
        "draft_deeplink_path": "",
        "draft_type": "",
        "draft_materials": [],
        "draft_materials_copied_info": [],
        "draft_enterprise_info": {},
        "canvas_config": {"width": width, "height": height, "ratio": "original"},
        "duration": total_us,
        "fps": float(fps),
        "version": 360000.0,
        "new_version": "110.0.0",
        "platform": {"app_id": 3704, "app_source": "lv", "app_version": "5.9.0",
                     "device_id": uuid.uuid4().hex, "os": "windows"},
        "materials": {
            "videos": [], "audios": [], "texts": materials,
            "stickers": [], "images": [], "effects": [],
            "transitions": [], "filters": [], "canvases": [],
            "speeds": [], "placeholders": [], "sound_channel_mappings": [],
            "common_keyframes": [], "text_templates": [],
            "material_animations": [], "audio_effects": [], "audio_fades": [],
            "beats": [], "vocal_separations": [],
        },
        "tracks": [{
            "id": uuid.uuid4().hex.upper(),
            "type": "text",
            "attribute": 0,
            "flag": 0,
            "segments": tracks_segments,
        }],
        "keyframes": {"videos": [], "audios": [], "texts": [],
                      "stickers": [], "filters": [], "transitions": []},
        "keyframe_refs": [],
        "relationships": [],
        "materials_extra_info": {},
        "smart_crops": [],
        "free_render_index_mode_on": False,
        "color_space": 0,
    }
    return json.dumps(draft, ensure_ascii=False, indent=2)


# ── EDL (Premiere / DaVinci 通用) ───────────────────────────────────


def generate_edl(segments: Sequence[Dict], *, fps: float = DEFAULT_FPS,
                 title: str = "SHIYU_SUBTITLE") -> str:
    """生成 CMX3600 EDL: Premiere / DaVinci Resolve / Avid 通用。

    EDL 是纯文本, 只带时间码与注释; 每句一条事件 —— 导入后是独立的
    字幕事件, 可单独加效果。它是**兼容性最好**的兜底格式。
    """
    frame = _fps_fraction(fps)
    fps_value = float(frame)

    def tc(seconds: float) -> str:
        total = int(round(seconds * fps_value))
        f = total % int(round(fps_value))
        s = (total // int(round(fps_value))) % 60
        m = (total // (int(round(fps_value)) * 60)) % 60
        h = total // (int(round(fps_value)) * 3600)
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    n = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip().replace("\n", " ")
        if not text:
            continue
        n += 1
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end <= start:
            continue
        lines.append(f"{n:03d}  AX       V     C        {tc(start)} {tc(end)} {tc(start)} {tc(end)}")
        lines.append(f"* FROM CLIP NAME: {text}")
        lines.append("")
    if n == 0:
        raise ValueError("没有字幕内容可导出")
    return "\n".join(lines)
