{
  "conflict_points": [
    "标点最小长度与 PUNCT_MIN_LEN_FOR_PERIOD 共享冲突:A 提议在 text_splitter 新增常量 PUNCT_MIN_LEN_FOR_PERIOD=10 并由 asr_corrector 复用;B 反对跨模块常量耦合(应由 corrector 自治)。真相是 10 这个数字当前硬编码在 asr_corrector._apply_punct_rules 第 793 行,B 担忧合理但 Phase 1 不必过度抽象。裁决:常量放 asr_corrector 内部,命名 PUNCT_MIN_LEN=10;text_splitter 不依赖 corrector 配置。",
    "_assign_timestamps 比例分配冲突:A 提议 OVERLAP=0.05s 防止段间出现字幕跳动;B 主张 TimeAuthority 接管、seg.end 强制对齐到 VAD span 边界;C 强调长 ASMR 段(30-60s)用单比例分配会产生 ±2s 误差。真相是 OVERLAP 是 UX 小补丁(连续感),不是边界真相来源;B 的 align_text 接口预留但 Phase 1 不实装。裁决:Phase 1 实施 OVERLAP=0.05 作为连续性补丁,Phase 2 用 CTC forced alignment 替代比例分配作为权威来源。",
    "DEFAULT_MAX_CHARS 冲突:A 主张 35→25(与 cloud/sensevoice 的 max_len=25 对齐);B 主张抽到 TimeAuthority 抽象后由 ScenePreset 注入(lecture=20 / mixed=45 / rap=12 / podcast=35 / asmr=35);C 主张按场景 20-45 范围动态决定。真相是 A 的 25 对齐只解决了 ASR 引擎原生切分场景;C 暴露的中英混说切碎英文单词是真问题,必须按 word 边界切;B 的抽象必要但应推到 Phase 2。裁决:Phase 1 先把对齐值由 35 降到 25(单一变量收益最高),Phase 2 引入 ScenePreset 时再决定是否上浮。",
    "三红线边界冲突:B 方案(TimeAuthority 抽象)严格遵守不破坏接口;但 C 方案(场景化参数)要求前端传 scene_preset,会扩展公共参数集合。这与红线 1「不破坏现有接口」看似冲突,实则是「不破坏」指签名兼容,而「扩展」是新增可选参数。裁决:Phase 2 的 scene_preset 在 main.py 接收处设为可选 kwargs,默认 None 走 DEFAULT_* 路径,实现「签名兼容 + 行为可选」双兼容。"
  ],
  "phased_plan": [
    {
      "phase": "Phase 1 — 立即可动的最小变更(本日交付)",
      "scope": "对齐 ASR 引擎原生切分阈值、修补时间戳连续性、提取 corrector 内部常量。零架构变化,零接口破坏,零测试失效。",
      "deliverables": "1) text_splitter.DEFAULT_MAX_CHARS: 35→25;2) text_splitter.DEFAULT_MAX_CHARS_EN: 50→42;3) 新增 _assign_timestamps_with_overlap(OVERLAP=0.05s) 包装原 _assign_timestamps;4) asr_corrector 内部 PUNCT_MIN_LEN=10 常量化;5) 单元测试 5 个新用例覆盖以上变更",
      "exit_criteria": "现有 32 个单测全部通过 + 新增 5 个用例通过;cloud/sensevoice 引擎切分片段数与字符数一致;时间戳连续无回退;PUNCT_MIN_LEN 修改后标点插入行为符合预期"
    },
    {
      "phase": "Phase 2 — 场景化参数体系(2 周内)",
      "scope": "引入 ScenePreset 枚举(lecture/mixed_language/rap/podcast/asmr),由前端 UI 选择并透传到后端;保留 DEFAULT_* 兼容路径。不动 TimeAuthority 抽象层(留到 Phase 3)。",
      "deliverables": "1) backend/app/core/scene_preset.py 新增枚举与切分参数映射表;2) main.py 接收 scene_preset 参数并透传到 cloud_engine / sensevoice_engine / segment_merger / asr_corrector;3) text_splitter 增加 scene_aware 版本 split_text_for_subtitles_v2(支持 word_boundary 模式);4) is_mostly_cjk 阈值在 mixed 场景放宽到 0.5;5) 前端 UI 新增场景选择下拉(lecture/mixed/rap/podcast/asmr/auto);6) 5 个场景各 1 个端到端测试",
      "exit_criteria": "5 种场景各跑 1 个 5 分钟样本音频,人工评审字幕可读性 4/5 分以上;word_boundary 模式下无英文单词被切碎;rap 模式单行 ≤12 字;asmr 模式单段可达 15s"
    },
    {
      "phase": "Phase 3 — TimeAuthority 抽象 + CTC 对齐权威化(1 个月内)",
      "scope": "实现 B 方案的核心构想,但仅在 CTC forced alignment 可用时启用,保留 VAD 作为 fallback。speaker label 暂不实装(留为 Phase 4)。",
      "deliverables": "1) backend/app/core/time_authority.py:TimeAuthority 抽象类 + VadTimeAuthority(默认)+ ForcedAlignmentTimeAuthority(SenseVoice 启用时);2) align_text(spans, text) 接口在 ForcedAlignmentTimeAuthority 中实装,直接调用 ctc_aligner.align_text_to_timestamps;3) segment_merger / text_splitter / asr_corrector 改为通过 TimeAuthority 实例取时间戳;4) 删除 cloud_engine._generate_timestamps_vad 内部的硬编码 libROSA 调用,改走 VadTimeAuthority;5) 删除 _assign_timestamps 比例分配路径,改为 TimeAuthority 强制对齐;6) 端到端对齐精度测试:CTC 路径与 Whisper verbose_json 时间戳 MAE < 0.3s",
      "exit_criteria": "CTC 路径时间戳 MAE < 0.3s;VAD 路径在 CTC 不可用时行为与 Phase 1 一致;TimeAuthority 接口通过 mock 测试覆盖;无现有调用方编译失败"
    }
  ],
  "three_red_lines_decision": {
    "item": [
      "红线 1:「不破坏现有接口」 — 保留(三个方案均不违反;B 的 TimeAuthority 抽象在 Phase 3 通过适配器兼容旧接口)",
      "红线 2:「不修改共享状态」 — 保留(asr_corrector 已遵守不修改输入;Phase 1 不触碰;Phase 3 TimeAuthority 设计为无状态)",
      "红线 3:「L1 确定性纠错优先于 L2 LLM」 — 保留并强化(Phase 2 的 ScenePreset 对 L1 注入参数,L2 LLM 只在 L1 之后运行,与现有架构一致)",
      "打破的红线:无。三个红线在本方案下全部保留,设计严格遵守「变更最小化」与「向后兼容」原则",
      "Phase 3 TimeAuthority 设计时受红线 1 约束的应对:红线 1 的「不破坏」指公共入口签名不破坏,内部 _assign_timestamps / _generate_timestamps_vad 等私有函数允许替换;TimeAuthority 通过 BaseTranscriber.transcribe 接口对外保持兼容"
    ]
  },
  "final_interfaces": "# Phase 1 新增/修改的最终接口签名\n\n# === text_splitter.py ===\nDEFAULT_MAX_CHARS: int = 25          # 由 35 下调,与 cloud/sensevoice 引擎 max_len 对齐\nDEFAULT_MAX_CHARS_EN: int = 42       # 由 50 下调,英文单词级切分更紧凑\nDEFAULT_MIN_CHARS: int = 15          # 不变\nDEFAULT_MAX_DURATION: float = 7.0    # 不变\n# PUNCT_MIN_LEN_FOR_PERIOD 不在本模块定义(避免与 asr_corrector 耦合)\n\ndef split_text_at_boundaries(\n    segments: List[Dict],\n    max_chars: int = DEFAULT_MAX_CHARS,        # 默认值已变更\n    min_chars: int = DEFAULT_MIN_CHARS,\n    max_duration: float = DEFAULT_MAX_DURATION,\n) -> List[Dict]:                                # 签名不变,只改默认值\n\ndef _assign_timestamps(\n    seg: Dict,\n    merged_parts: List[str],\n    duration: float,\n    result: List[Dict],\n) -> None:\n    # 内部委托给 _assign_timestamps_with_overlap,行为完全向后兼容\n    _assign_timestamps_with_overlap(seg, merged_parts, duration, result)\n\ndef _assign_timestamps_with_overlap(\n    seg: Dict,\n    parts: List[str],\n    duration: float,\n    result: List[Dict],\n    overlap_sec: float = 0.05,                  # 新增可选参数,默认值即新行为\n) -> None:",
  "algorithm_pseudocode": "# Phase 1 — _assign_timestamps_with_overlap 核心算法\n\nfunction _assign_timestamps_with_overlap(seg, parts, duration, result, overlap_sec=0.05):\n    total_chars = sum(len(p) for p in parts)\n    if total_chars == 0:\n        result.append(copy(seg))\n        return\n\n    cursor = seg.start\n    n = len(parts)\n    for idx, part_text in enumerate(parts):\n        is_last = (idx == n - 1)\n        part_ratio = len(part_text) / total_chars\n        raw_end = seg.end if is_last else cursor + duration * part_ratio\n\n        if is_last:\n            part_end = seg.end\n        else:\n            part_end = raw_end\n            next_cursor = raw_end - overlap_sec\n        else:\n            part_end = raw_end\n            next_cursor = raw_end\n\n        result.append({\n            \"start\": round(cursor, 3),\n            \"end\": round(part_end, 3),\n            \"text\": part_text,\n        })\n        cursor = next_cursor if not is_last else raw_end\n\n# 关键不变式:\n#   - parts[0].start == seg.start\n#   - parts[-1].end == seg.end\n#   - parts[i].end + overlap_sec >= parts[i+1].start (连续无回退,允许微小重叠)\n#   - overlap_sec 越小越接近严格按字符比例,越大越能掩盖时间戳抖动\n\n# Phase 3 — TimeAuthority.aligned_split 核心算法(预留)\n# function aligned_split(text, audio_waveform, sr, base_spans) -> List[Segment]:\n#     if ctc_model_available:\n#         logits = run_sensevoice_encoder(audio_waveform, sr)\n#         char_timestamps = ctc_aligner.align_text_to_timestamps(logits, text, sp_model)\n#         return group_chars_to_subtitles(char_timestamps, scene_preset)\n#     else:\n#         return vad_time_authority.aligned_split(text, audio_waveform, sr, base_spans)",
  "test_strategy": "# Phase 1 测试策略(单测为主,无需 GPU/网络)\n\n1. 单元测试(backend/tests/core/test_text_splitter.py 已有):\n   - test_default_max_chars_changed_to_25: 断言 text_splitter.DEFAULT_MAX_CHARS == 25\n   - test_split_text_at_boundaries_default_uses_25: 35 字输入不再触发切分\n   - test_split_text_at_boundaries_splits_at_25: 40 字输入切分为 [≤25, 余]\n   - test_assign_timestamps_with_overlap_continuous: 验证连续 3 段 end[i] + 0.05 >= start[i+1]\n   - test_assign_timestamps_with_overlap_preserves_first_start: parts[0].start == seg.start\n   - test_assign_timestamps_with_overlap_preserves_last_end: parts[-1].end == seg.end\n   - test_assign_timestamps_with_overlap_default_overlap_is_005: 默认值正确\n   - test_assign_timestamps_with_overlap_custom_overlap_0: overlap_sec=0 退化为旧行为\n   - test_punct_min_len_constant_in_asr_corrector: asr_corrector 内部 PUNCT_MIN_LEN == 10\n\n2. 集成测试(backend/tests/integration/):\n   - test_cloud_engine_segments_use_new_default: mock DashScope, 验证返回的 segment 内部子段 ≤ 25 字符\n   - test_segment_merger_split_long_segments_default: 验证 split_long_segments 默认走 25 阈值\n\n3. 回归保护(必须,不可省):\n   - 跑现有 32 个 text_splitter / segment_merger / asr_corrector 用例,确保 0 失败\n   - 跑现有 5 个 cloud_engine 集成用例,确保 0 失败\n   - 手动跑 1 个 1 分钟中文样本,确认字幕可读性未退化\n\n4. 覆盖率门槛:\n   - text_splitter.py 整体 ≥ 90%(已 95%+,变更不应拉低)\n   - _assign_timestamps_with_overlap 新分支 100% 覆盖\n\n5. 跳过/不验证项:\n   - Phase 1 不跑端到端 GPU 测试(CTC 模型在 Phase 3 验证)\n   - Phase 1 不验性能(字符数下降只会让切分更轻量)\n   - 不验前端 UI(Phase 2 才动 UI)",
  "risks_and_mitigations": [
    {
      "risk": "DEFAULT_MAX_CHARS 从 35 降到 25 后,某些 LLM 修正输出的长句(35-40 字)会被切到 2 行,可能与 LLM 期望的 1 行不匹配",
      "probability": "medium",
      "impact": "low",
      "mitigation": "Phase 1 只影响 ASR 引擎原生的初始切分,LLM 修正路径走的是 segment_merger.split_long_segments(可独立传参);在 main.py 拼接 pipeline 时确认 split_long_segments 仍传 DEFAULT_MAX_CHARS;若发现 LLM 路径字幕变碎,Phase 2 在 ScenePreset 中为 lecture 场景单独设回 30"
    },
    {
      "risk": "OVERLAP=0.05s 在快速语流下可能让相邻段字幕同时显示 50ms,前端 SRT 播放器偶发闪一下",
      "probability": "low",
      "impact": "low",
      "mitigation": "0.05s 是行业默认值(YouTube auto-caption 也用类似余量);若用户反馈,Phase 2 把 overlap_sec 纳入 ScenePreset 可调参数;在 _assign_timestamps_with_overlap 注释中说明:overlap_sec=0 即可禁用"
    },
    {
      "risk": "PUNCT_MIN_LEN 从硬编码 10 提到模块常量,改名 PUNCT_MIN_LEN 后,grep 历史 bug 报告可能找不到引用",
      "probability": "low",
      "impact": "low",
      "mitigation": "在常量定义处加注释「原硬编码于 _apply_punct_rules line 793,Phase 1 提为常量」,git blame 可追溯"
    },
    {
      "risk": "Phase 3 TimeAuthority 重构会触碰 _generate_timestamps_vad / _assign_timestamps 两处,影响面较大,可能引入回归",
      "probability": "medium",
      "impact": "medium",
      "mitigation": "Phase 2 末预留 1 周做 TimeAuthority 适配器灰度:旧路径加 feature flag,新路径加监控(段数差、时间戳 MAE),灰度 1 周后全量切换"
    },
    {
      "risk": "ScenePreset 5 个场景的切分参数取值(20/45/12/35/35)是基于 C 方案经验值,缺乏量化数据支撑",
      "probability": "medium",
      "impact": "medium",
      "mitigation": "Phase 2 上线后 1 周内收集 5 个场景各 3 个真实样本的人工可读性评分与时间戳 MAE,按数据回校参数;不追求一步到位,允许 Phase 2.1 微调"
    },
    {
      "risk": "三个红线全保留,可能导致 Phase 3 TimeAuthority 设计时束手束脚(尤其红线 1「不破坏接口」)",
      "probability": "low",
      "impact": "low",
      "mitigation": "红线 1 的「不破坏」指公共入口签名不破坏;内部 _assign_timestamps / _generate_timestamps_vad 等私有函数允许替换;TimeAuthority 通过 BaseTranscriber.transcribe 接口对外保持兼容"
    }
  ],
  "rollback_plan": "# Phase 1 回滚方案(5 分钟内可回退)\n\n## 触发条件(任一):\n1. 现有 32 个单测中 ≥ 3 个失败\n2. 端到端 sample 字幕出现半截单词被切或长句异常碎化\n3. cloud/sensevoice 引擎返回段数较修改前 +50% 以上\n\n## 回滚操作(零数据风险,纯代码):\n\n1. text_splitter.py:\n   - DEFAULT_MAX_CHARS = 35           (恢复 35)\n   - DEFAULT_MAX_CHARS_EN = 50        (恢复 50)\n   - _assign_timestamps 直接恢复为旧实现,删除 _assign_timestamps_with_overlap\n   - 预计 diff: 8 行\n\n2. asr_corrector.py:\n   - 删除 PUNCT_MIN_LEN 常量\n   - _apply_punct_rules line 793 改回 if len(trimmed) > 10\n   - 预计 diff: 3 行\n\n3. 验证:\n   - pytest backend/tests/core/test_text_splitter.py -v\n   - pytest backend/tests/core/test_asr_corrector.py -v\n   - git revert <phase1-commit-sha>  (一键)\n\n## 风险等级:低\n- Phase 1 仅修改默认值与新增 1 个私有函数\n- 无 schema 变更、无 API 变更、无配置变更\n- 公共函数签名 100% 兼容\n- git revert 一行命令,5 分钟内恢复\n\n## 数据备份:无需\n- 无数据库变更\n- 无用户配置变更\n- 无音频/字幕产物变更",
  "phase1_immediate_patches": [
    {
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\text_splitter.py",
      "location": "line 25-28 常量区",
      "change": "DEFAULT_MAX_CHARS = 35 → 25;DEFAULT_MAX_CHARS_EN = 50 → 42;DEFAULT_MIN_CHARS / DEFAULT_MAX_DURATION 不变",
      "difficulty": "trivial"
    },
    {
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\text_splitter.py",
      "location": "line 231-253 _assign_timestamps 函数体",
      "change": "将原 _assign_timestamps 函数体提取为 _assign_timestamps_with_overlap(seg, parts, duration, result, overlap_sec=0.05),新函数在非最后段做 cursor = part_end - overlap_sec 修正;原 _assign_timestamps 改为对 _assign_timestamps_with_overlap 的薄包装(签名不变)",
      "difficulty": "easy"
    },
    {
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\asr_corrector.py",
      "location": "line 793 之前新增模块级常量",
      "change": "新增 PUNCT_MIN_LEN: int = 10 常量(带注释「原硬编码于 _apply_punct_rules,Phase 1 提为常量」);line 793 的 if len(trimmed) > 10 改为引用 PUNCT_MIN_LEN",
      "difficulty": "trivial"
    },
    {
      "file": "D:\\codes\\Shiyu\\backend\\tests\\core\\test_text_splitter.py",
      "location": "追加 4 个测试函数到文件末尾",
      "change": "test_default_max_chars_is_25; test_assign_timestamps_with_overlap_default_005; test_assign_timestamps_with_overlap_continuous_no_gap; test_assign_timestamps_with_overlap_disable_overlap",
      "difficulty": "easy"
    },
    {
      "file": "D:\\codes\\Shiyu\\backend\\tests\\core\\test_asr_corrector.py",
      "location": "追加 1 个测试函数",
      "change": "test_punct_min_len_constant_exported_and_used,验证 asr_corrector.PUNCT_MIN_LEN == 10 且 _apply_punct_rules 实际引用该常量",
      "difficulty": "trivial"
    }
  ]
}