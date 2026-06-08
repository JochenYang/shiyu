{
  "user_scenarios": [
    {
      "$text": "场景 1: 中文长视频（知识口播/讲座）— 单人语速中等偏慢, 句法完整, 标点丰富, 用户在 1-2 倍速下回看, 需要清晰可定位的句首时间锚点。痛点: 当前 MAX_SENTENCE_CHARS=40 仍偏长, 标点密度高时容易出现 “半句话换行”, 影响扫读。完美形态: 每行 16-22 字 (1-2 行), 句末标点处必切, 逗号处可选切, 严格 6.5s 时长上限, 时间戳精度 ±0.1s。预期参数: chars_per_line=20, pause_threshold=0.45s, max_duration=6.5s, language_mode=cjk_strict, scene_preset=lecture。"
    },
    {
      "$text": "场景 2: 中英混说（vlog / 跨境分享 / 脱口秀）— 同一句话里中英文交替, 英文占比 20-60%, 专有名词/品牌名/代码片段多。痛点: 当前 is_mostly_cjk 阈值 0.3 一刀切, 中英混说时频繁按 char 硬切, 切出 “app / le / ta / 还是” 这种半截单词, 阅读体验崩坏。完美形态: 切分时按语言边界优先 (空白 + 标点), 英文按 word 切, 中文按 char 切, 切点绝不落在英文单词中间, 允许单行比纯中文场景略长 (40-50 字符)。预期参数: chars_per_line=45, pause_threshold=0.4s, max_duration=7s, language_mode=word_boundary, scene_preset=mixed_language。"
    },
    {
      "$text": "场景 3: 说唱歌词（嘻哈/快节奏念白）— 高密度押韵, 语速极快 (10+ 字/秒), 句尾停顿极短 (<0.2s), VAD 容易把整段识别为一个长 interval。痛点: 当前按 VAD 切, 整段会变 1 条超长字幕, text_splitter 再按标点切, 但说唱极少标点, 落到 Level 3 硬切, 切在押韵中间。完美形态: 强制 4-6s 节奏切分 (节拍网格对齐), 允许每行较短 (8-12 字), 优先在押韵词尾/呼吸位切, 切点附上节拍高亮提示供前端调参。预期参数: chars_per_line=12, pause_threshold=0.15s, max_duration=4s, language_mode=auto, scene_preset=rap, break_at_comma=false。"
    },
    {
      "$text": "场景 4: 播客对话（双人/多人访谈）— 多说话人快速接话, 插话/抢话/重叠常见, 停顿语义化 (短停是思考, 长停是换人)。痛点: 当前 VAD 把抢话段合并成一条, 失去说话人切换节奏; 又或者把短停切碎, 一句话三段。完美形态: 停顿阈值分档 — 短停 (<0.3s) 不切, 中停 (0.3-0.8s) 仅在标点处切, 长停 (>0.8s) 必切; 说话人切换处优先切; 引入 “说话人标签 [A]: ” 前缀 (可选开关)。预期参数: chars_per_line=35, pause_threshold=0.3-0.8s (双阈值), max_duration=8s, language_mode=mixed, scene_preset=podcast, enable_speaker_label=true。"
    },
    {
      "$text": "场景 5: ASMR / 气声独白（无停顿连续耳语）— 极长气声, 几乎无 VAD 停顿, 单条 interval 可达 30-60s, 用户期望 “跟读用” 而非 “扫读用” 字幕, 偶尔需要长上下文。痛点: 当前 MAX_SENTENCE_CHARS=40 + VAD-first 强制把长段切碎, 但切点语义化不足, 容易断在助词/虚词。完美形态: ASMR 模式开关打开后, 停顿阈值放宽到 1.0s, 每行放宽到 30-40 字, 时长上限 15s, 在 “气口/句末助词” (呢/啊/吧/嘛) 处优先切, 提供 “长段模式” UI 提示。预期参数: chars_per_line=35, pause_threshold=1.0s, max_duration=15s, language_mode=cjk_strict, scene_preset=asmr, break_at_comma=false。"
    }
  ],
  "perfect_srt_examples": [
    "1\\n00:00:00,320 --> 00:00:03,180\\n大家好, 欢迎来到本期频道。\\n今天我们聊一聊 AI 字幕。\\n\\n2\\n00:00:03,450 --> 00:00:07,820\\n先说结论, 完美的分句要做到\\n三件事: 看得清、跟得上、找得到。\\n\\n3\\n00:00:08,100 --> 00:00:11,560\\n第一, 视觉上每行不超过二十二个字,\\n断行不切断语义。\\n\\n4\\n00:00:11,900 --> 00:00:15,230\\n第二, 时间上每条不超过六点五秒,\\n留出眼睛呼吸的余量。",
    "1\\n00:00:00,200 --> 00:00:03,400\\n最近我在用一个新的 productivity app,\\n\\n2\\n00:00:03,400 --> 00:00:05,900\\n它支持 mixed-language mode。\\n\\n3\\n00:00:05,900 --> 00:00:08,800\\n它的 LLM corrector 一次能处理\\n差不多 50 个字符的 chunk。\\n\\n4\\n00:00:08,800 --> 00:00:12,000\\n我录了一段中文带英文 brand name 的视频,\\n测了一下分句效果。",
    "1\\n00:00:00,000 --> 00:00:02,800\\nYo yo yo check it out\\nverse one 开始\\n\\n2\\n00:00:02,800 --> 00:00:05,400\\n我把节拍踩在脚下 flow 不停\\n像 train 不需要 station\\n\\n3\\n00:00:05,400 --> 00:00:08,000\\n麦克风是我的武器 押韵是我的 armor\\n上膛就 fire\\n\\n4\\n00:00:08,000 --> 00:00:10,600\\nVerse 2 加速 double time\\n你跟不上别怪我",
    "1\\n00:00:00,000 --> 00:00:02,500\\n[A]: 大家好, 欢迎收听本期播客。\\n\\n2\\n00:00:02,500 --> 00:00:05,200\\n[B]: 我是主持人 Michael, 今天我们请到了\\nGPT-5 团队的产品负责人。\\n\\n3\\n00:00:05,200 --> 00:00:09,800\\n[A]: 先请嘉宾做个自我介绍, 然后我们聊聊\\n最近 AI 字幕工具的发展。\\n\\n4\\n00:00:09,800 --> 00:00:12,500\\n[C]: 好的, 我是 Sarah, 一直在做\\nmultimodal 方向的研究。",
    "1\\n00:00:00,000 --> 00:00:12,500\\n大家好呀, 今天我们继续上期的话题,\\n讲一讲睡眠与声音的关系, 嗯,\\n其实很多人不知道, 我们的大脑\\n在浅睡阶段对低频声音特别敏感,\\n\\n2\\n00:00:12,500 --> 00:00:25,000\\n白噪声之所以有效, 是因为它能\\n盖住突发的环境噪音, 让大脑放松下来。\\n\\n3\\n00:00:25,000 --> 00:00:38,000\\n接下来我们进入今天的正题, 讲讲\\nASMR 场景下的分句策略和参数推荐, 呢。"
  ],
  "tunable_params": [
    {
      "key": "chars_per_line_cjk",
      "label": "中文每行字符数",
      "type": "int",
      "default": "20",
      "range": "8-40",
      "unit": "字符",
      "scene_binding": "中文长视频 / ASMR",
      "description": "单行目标字符数。偏长利于扫读, 偏短利于跟读。建议 16-22 区间。"
    },
    {
      "key": "chars_per_line_en",
      "label": "英文每行单词数",
      "type": "int",
      "default": "10",
      "range": "4-18",
      "unit": "单词",
      "scene_binding": "中英混说 / 播客",
      "description": "英文按 word 切, 默认 10 词/行 (约 50-60 字符)。"
    },
    {
      "key": "pause_threshold_sec",
      "label": "停顿阈值 (VAD 切分)",
      "type": "float",
      "default": "0.5",
      "range": "0.15-1.2",
      "unit": "秒",
      "scene_binding": "所有场景",
      "description": "超过此停顿即视为句间。预设三档: 紧凑 0.3 / 标准 0.5 / 宽松 0.8。说唱 0.15, ASMR 1.0。"
    },
    {
      "key": "max_segment_duration_sec",
      "label": "单条字幕最大时长",
      "type": "float",
      "default": "7",
      "range": "3-15",
      "unit": "秒",
      "scene_binding": "所有场景",
      "description": "硬上限。超过即按标点或硬切。说唱 4s, ASMR 15s, 讲座 6.5s。"
    },
    {
      "key": "min_segment_duration_sec",
      "label": "单条字幕最小时长",
      "type": "float",
      "default": "1.2",
      "range": "0.6-3",
      "unit": "秒",
      "scene_binding": "所有场景",
      "description": "短于此则向后合并, 避免闪屏。"
    },
    {
      "key": "language_mode",
      "label": "语言模式",
      "type": "enum",
      "default": "auto",
      "options": "auto / cjk_strict / en_strict / word_boundary / mixed",
      "scene_binding": "中英混说",
      "description": "auto 走 is_mostly_cjk 启发式; mixed 场景推荐 word_boundary, 切点绝不落在英文单词内。"
    },
    {
      "key": "scene_preset",
      "label": "场景预设",
      "type": "enum",
      "default": "standard",
      "options": "lecture / mixed_language / rap / podcast / asmr / standard / custom",
      "scene_binding": "所有场景",
      "description": "一键套用场景参数包, 降低用户认知负担。custom 暴露细粒度滑块。"
    },
    {
      "key": "enable_speaker_label",
      "label": "说话人标签",
      "type": "bool",
      "default": "false",
      "scene_binding": "播客对话",
      "description": "开启后在字幕前加 [A]: [B]: 前缀, 依赖说话人分离结果。"
    },
    {
      "key": "break_at_comma",
      "label": "逗号处是否切分",
      "type": "bool",
      "default": "true",
      "scene_binding": "中文长视频",
      "description": "默认开启, 利于扫读; ASMR/说唱建议关闭。"
    },
    {
      "key": "prefer_terminal_punct",
      "label": "硬切时倾向标点",
      "type": "bool",
      "default": "true",
      "scene_binding": "所有场景",
      "description": "硬切时是否在 ±3 字符内搜索标点优先, 避免语义破碎。当前实现已部分支持, 建议显式开关。"
    }
  ],
  "bad_case_definitions": [
    "bad-1 英文单词被腰斩 (high) — 触发: 中英混说场景, language_mode=auto, 字符硬切命中英文单词中间。检测规则: 切点位置左右 3 字符内包含英文小写字母且切点不在空格/标点处。例: “最近我在用一个新的 productivity | app, 它支持 mixed mode” → 应为 “productivity app,” 一体。降级策略: 切到 word_boundary 模式重切; 若仍失败, 保留原 ASR 段落, 前端标红 + tooltip 提示 “检测到英文被腰斩, 已合并”, metadata.fallback_reason=en_word_clipped。",
    "bad-2 无停顿超长段 (high) — 触发: ASMR/长气声场景, VAD 无停顿, 单 interval > 30s, MAX_SENTENCE_CHARS=40 切出 8+ 条无意义短句。检测规则: 单个 VAD interval duration > 20s 且切分后字幕数 > 6。例: 35s 气声独白被切成 8 段 4-5s 短字幕, 切点全在虚词。降级策略: 检测 interval 长度, > 20s 自动启用 ASMR 模式 (停顿 1.0s, 时长 15s), metadata.fallback_reason=long_speech_interval。",
    "bad-3 说唱过度切碎 (medium) — 触发: 说唱场景 VAD 误识别 0.1-0.2s 呼吸停顿, 整段被切碎成 20+ 短条, 押韵节奏全毁。检测规则: 相邻字幕间隔 < 1.2s 的条数 / 总条数 > 0.6。例: 8s 说唱段被切成 “Yo | yo | yo | check | it | out | verse | one” 8 段。降级策略: 自动降级到 rap preset, 合并 1.2s 内的短条, metadata.fallback_reason=rap_over_split。",
    "bad-4 多人说话人合并丢失 (high) — 触发: 播客抢话场景, 3 人接话被 VAD 识别为单段, 字幕合并成一条无主语长段。检测规则: 单条字幕包含 ≥ 2 个不同说话人 (依赖 diarization), 或单条 > 4s 且无句末标点。例: 5s 内 A-B-C 三人抢话合并成一条。降级策略: 依赖说话人分离结果; 缺失时退化为停顿切分, 标记 fallback_reason=no_diarization; 抢话段在元数据打 overlap_warning。",
    "bad-5 ASR 重叠导致文本重复 (medium) — 触发: sliding window ASR 输出含重复 segment, 去重失败导致字幕出现重复字串。检测规则: 单条字幕内 n-gram 重复率 > 30%。例: “今天天气真 / 天气真好 / 好我们出去” → “今天天气真天气真好我们出去”。降级策略: _deduplicate_segments 已实现, 若检测到重复率 > 30% 仍残留, 标记 fallback_reason=dedup_failed, 降级到 segment_merger.merge_segments 重跑。",
    "bad-6 字幕跳变时间戳错位 (low) — 触发: VAD 边界与 ASR 时间戳不对齐, 字幕在 0.3s 内切换内容, 视觉跳变。检测规则: 相邻字幕 start 间隔 < 0.3s 且文本不同时, 触发跳变警告。例: 句A 00:00:05.000 → 句B 00:00:05.150, 跳变 0.15s。降级策略: 前端最小显示时间强制 1.2s; 后端 min_segment_duration 默认 1.2s 兜底, 同时把 start 推后 0.3s。",
    "bad-7 中文硬切在词中间 (medium) — 触发: 长句无标点, 字符硬切命中双字词中间。检测规则: 切点位置左右 2 字符同时为 CJK 汉字 (即切在双字词中间) 且该位置非标点。例: “北京市朝阳区三 / 里屯街道” → 应为 “朝阳区三里屯” 一体。降级策略: 硬切时 ±3 字符内优先找标点, 找不到则按常见双字词后缀 (的/了/是/在/和/与/里/上/下) 切; 仍失败则标记 fallback_reason=hard_cut_no_boundary, 前端允许用户手动合并。",
    "bad-8 切分后产生空段 (low) — 触发: 切分逻辑边界条件处理不当, 产生空文本段, 渲染时空字幕占位。检测规则: 输出字幕中 text.strip() == '' 的条数 > 0。例: 含连续标点 “。。。” 或纯空白的中文段落切分后产生空段。降级策略: 在 subtitle_formatter.generate_srt 前增加 filter: text.strip() == '' 的段丢弃, 同时打 warning 计数到 metrics, metadata.fallback_reason=empty_segment_dropped。"
  ],
  "frontend_debug_needs": [
    "debug-1 VAD 边界高亮 (P0): 在字幕预览面板叠加 VAD 切点, 蓝色竖线 + 时间戳, 帮助用户理解 “为什么这里切了”。鼠标悬停显示 (停顿时长, 是否触发了 fallback)。数据来源: 后端 sentence_splitter 输出新增 vad_boundaries 字段 [{time, pause_sec, is_terminal, is_fallback}]。实现: 前端 timeline overlay layer。",
    "debug-2 句子决策树可视化 (P0): 每一行字幕可展开, 显示它的决策路径: VAD interval → 标点启发 → max_chars 强制 → 碎片合并。可点击重新跑该段不同参数。数据来源: 每条 subtitle 增加 decision_trace 数组 [{stage, action, params, before, after}]。实现: 前端折叠面板。",
    "debug-3 “为什么没切” 提示 (P1): 长段 (>15s) 但被识别为单条时, 显示原因: 停顿不够 / 没标点 / 阈值过高。给用户调参依据。数据来源: sentence_splitter 输出 reason 字段, 枚举 no_pause / no_terminal_punct / merged_by_threshold / fallback_disabled。实现: 字幕条 hover 提示。",
    "debug-4 参数实时预览 (P0): 用户调整 chars_per_line / pause_threshold 滑块时, 前端 debounce 300ms 重跑分句, 立即看到效果对比 (左右分屏: 旧 vs 新)。数据来源: 后端提供 /api/preview/split 接口, 接收 segments + params, 返回新 segments, 不写入存储。实现: 滑块 + 双栏 diff 视图。",
    "debug-5 坏例警示浮层 (P1): 检测到 bad case (英文腰斩 / 词中切 / 跳变) 时, 在对应字幕上显示红色警示图标 + hover 提示降级方案。数据来源: 后端 metadata.bad_cases [{type, position, fallback_applied}]。实现: 前端根据 type 渲染不同图标。",
    "debug-6 音频波形同步 (P2): 字幕条与音频波形 1:1 对齐, 点击字幕可定位音频位置, 拖动字幕边界可调时间戳 (写回后端)。实现: 使用 wavesurfer.js 渲染波形, 双击字幕条进入编辑模式。",
    "debug-7 场景预设自动推荐 (P2): 上传音频后, 后端基于 (语速, 停顿密度, 中英比) 推测最佳 scene_preset, 前端给 “推荐使用 rap 模式” 提示。数据来源: 在主流程最早期 (ASR 之前) 跑一段 30s 音频分析, 输出 preset_hint。实现: 一键应用按钮。"
  ],
  "feedback_loop": "反馈闭环设计 (6 个子通道, 周聚合 + 月迭代): 通道 1 — 每行字幕点赞/点踩: 用户在字幕预览面板对每条字幕 thumbs up/down, 收集 “哪些句子用户觉得切得不好” 的原始信号。数据: segment_id, rating (±1), user_id, applied_params snapshot。存储: feedback.db 表 subtitle_feedback。用途: 聚合分析, 在某 scene_preset + 参数组合下差评率 > 20% 的文件 → 提示调参。通道 2 — 手动编辑轨迹: 用户在前端拖动字幕边界/合并/拆分时, 上报 edit diff 作为 “分句改对答案” 的金标准。数据: original_segments, edited_segments, edit_type (split/merge/shift), edit_position。存储: 表 subtitle_edits。用途: 构造 fine-tuning 数据集, 训练 regression model 调整默认参数。通道 3 — 坏例一键反馈: 在 debug-5 浮层基础上, 用户可一键 “这是 bug”, 触发回传 segment_id + bad_case_type + 当前参数 + 原始 ASR 输出。数据: bad_case_type (从 7 个枚举), segment_text + timestamps, params_snapshot, raw_asr_segments。存储: 表 bad_case_reports (含 status: new/triaged/fixed)。用途: 开发者看板周汇总 Top 5 bad case, 优先修高频项。通道 4 — 参数变更历史: 用户每次保存导出, 把最终采纳的参数写入历史, 支持 “上次那种切法” 一键回滚。数据: file_id, params_json, scene_preset, scene_auto_detected, user_kept_or_reverted。存储: 表 params_history。用途: 跨文件复用参数; 统计 “自定义参数 vs 预设” 采用率, 决定预设是否要改。通道 5 — A/B 参数实验: 对同一音频, 随机用两套参数 (A=当前默认, B=新候选) 各切一次, 让用户盲选。数据: file_id, param_set_a vs b, user_choice (A/B/tie), choice_reason。存储: 表 ab_experiments。用途: 新参数上线前的灰度验证; B 胜出率 > 60% 持续 2 周即替换默认。通道 6 — 闭环总览: 收集 (thumbs + 编辑 + 坏例) → 周聚合 (Top bad case + 调参 pattern) → 算法侧 PR (新增坏例防御) → A/B 灰度 → 全量 + 用户通知 (“v1.3 修复了您常上报的 ‘英文腰斩’ 问题”)。节奏: 每周一次反馈 review, 每月一次算法迭代。"
}