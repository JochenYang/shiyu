{
  "phase1_changes": [
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\text_splitter.py",
      "function": "module-level constant",
      "change": "DEFAULT_MAX_CHARS: 35 -> 25",
      "lines_changed": "1",
      "verification": "grep \"DEFAULT_MAX_CHARS = \" returns 25; rerun existing split_text_at_boundaries tests",
      "dependency_order": "1"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\text_splitter.py",
      "function": "module-level constant",
      "change": "DEFAULT_MAX_CHARS_EN: 50 -> 42",
      "lines_changed": "1",
      "verification": "grep \"DEFAULT_MAX_CHARS_EN = \" returns 42",
      "dependency_order": "1"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\text_splitter.py",
      "function": "_assign_timestamps_with_overlap (new wrapper)",
      "change": "Add module-level constant OVERLAP_SECONDS = 0.05 and new public function _assign_timestamps_with_overlap(seg, parts, duration, result) that calls _assign_timestamps and subtracts OVERLAP from each non-first segment.start; wraps only the internal helper, not the public split_text_at_boundaries signature.",
      "lines_changed": "~20",
      "verification": "Unit test: for input 2-part split [start=0,end=10] -> seg1 [0, 5), seg2 [5.05, 10) when duration>=OVERLAP",
      "dependency_order": "2"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\asr_corrector.py",
      "function": "ASRDeterministicCorrector module-level + _apply_punct_rules",
      "change": "Add module-level constant PUNCT_MIN_LEN = 10; replace literal `10` on line 793 with PUNCT_MIN_LEN",
      "lines_changed": "2",
      "verification": "grep \"len(trimmed) > PUNCT_MIN_LEN\" present; no change in punctuation insertion behavior for 10/11-char inputs",
      "dependency_order": "3"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\tests\\test_text_splitter_phase1.py",
      "function": "new test file",
      "change": "Add 5 unit tests: (1) DEFAULT_MAX_CHARS==25, (2) split_text_at_boundaries respects new max_chars for 25-char long text, (3) _assign_timestamps_with_overlap produces overlapping timestamps for >=2 parts, (4) overlap constant is 0.05, (5) PUNCT_MIN_LEN exists in asr_corrector and is referenced.",
      "lines_changed": "~80",
      "verification": "pytest backend/tests/test_text_splitter_phase1.py -v shows 5 passed",
      "dependency_order": "4"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\tests\\test_asr_corrector_phase1.py",
      "function": "new test file",
      "change": "Add 3 unit tests: (1) punctuation rule 3 fires for text >10 chars and skips for <=10, (2) PUNCT_MIN_LEN constant is accessible, (3) existing 32 unit tests still pass (regression).",
      "lines_changed": "~50",
      "verification": "pytest backend/tests/test_asr_corrector_phase1.py -v",
      "dependency_order": "4"
    }
  ],
  "phase2_changes": [
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\scene_preset.py",
      "function": "new module",
      "change": "Define ScenePreset enum (LECTURE, MIXED_LANGUAGE, RAP, PODCAST, ASMR) and SCENE_PARAMS dict mapping each preset to {max_chars, max_chars_en, max_duration, min_chars, word_boundary:bool}. Provide resolve_scene_params(preset: str|None) -> dict helper that returns DEFAULT_* when preset is None/auto.",
      "lines_changed": "~80",
      "verification": "Unit test: resolve_scene_params(None) returns DEFAULT dict; resolve_scene_params('rap') returns max_chars=12; resolve_scene_params('mixed_language') returns word_boundary=True.",
      "dependency_order": "1"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\text_splitter.py",
      "function": "split_text_for_subtitles_v2 + is_mostly_cjk",
      "change": "Add word_boundary parameter to split_text_for_subtitles_v2 (English hard-cut uses word-boundary loop regardless of language flag). Loosen is_mostly_cjk threshold to accept an override threshold param (default 0.3, 0.5 for mixed). Keep split_text_for_subtitles as v1 alias.",
      "lines_changed": "~60",
      "verification": "Unit test: v2 with word_boundary=True keeps \"internationalization\" intact even when len>max_len.",
      "dependency_order": "2"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\segment_merger.py",
      "function": "merge_segments / split_long_segments",
      "change": "Accept optional scene_params dict, default None. When None, use text_splitter.DEFAULT_*; when provided, pass max_chars/max_duration into split_text_at_boundaries.",
      "lines_changed": "~15",
      "verification": "Existing call sites unchanged; new optional kwarg does not break API.",
      "dependency_order": "3"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\cloud_engine.py",
      "function": "transcribe entry points (lines ~268, 640, 813, 1100)",
      "change": "Accept optional scene_preset: str|None parameter; pass into split_text_for_subtitles_v2 and is_mostly_cjk. Default None keeps current behavior.",
      "lines_changed": "~30",
      "verification": "All 4 call sites receive scene_preset kwarg with default None.",
      "dependency_order": "3"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\sensevoice_engine.py",
      "function": "transcribe (line ~357)",
      "change": "Accept optional scene_preset parameter; thread into split_text_for_subtitles_v2.",
      "lines_changed": "~10",
      "verification": "Local SenseVoice transcribe() default args unchanged.",
      "dependency_order": "3"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\core\\asr_corrector.py",
      "function": "ASRDeterministicCorrector.__init__",
      "change": "Accept optional scene_preset: str|None; load SCENE_PARAMS and pass min_chars/PUNCT_MIN_LEN override when computing punctuation (Phase 2 keeps PUNCT_MIN_LEN but allows scene-level override for ASMR long segments).",
      "lines_changed": "~15",
      "verification": "Existing deterministic_correct() callers work without arg.",
      "dependency_order": "3"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\app\\main.py",
      "function": "transcribe endpoint(s)",
      "change": "Add optional Form/JSON field scene_preset: str|None = Form(None). Thread into engine.transcribe() and segment_merger.merge_segments(). Signature-compatible: omitting the field keeps the old behavior.",
      "lines_changed": "~20",
      "verification": "OpenAPI schema shows scene_preset as optional; existing POST calls work.",
      "dependency_order": "4"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\frontend\\src\\components\\TranscribeForm.vue",
      "function": "UI form",
      "change": "Add scene preset dropdown (lecture/mixed_language/rap/podcast/asmr/auto) and send scene_preset field.",
      "lines_changed": "~30",
      "verification": "Manual: select \"rap\" and confirm max line length <=12.",
      "dependency_order": "5"
    },
    {
      "file_path": "D:\\codes\\Shiyu\\backend\\tests\\test_scene_preset.py",
      "function": "new test file",
      "change": "5 end-to-end style tests: one per ScenePreset, feeding canned segments, asserting max segment char count <= preset max_chars.",
      "lines_changed": "~120",
      "verification": "pytest test_scene_preset.py -v shows 5 passed",
      "dependency_order": "6"
    }
  ],
  "execution_order": [
    "Phase 1 - 拉分支 feat/phase1-splitter-thresholds，主分支基线，复制 32 个现有测试 fixture",
    "Phase 1.1 - 改 text_splitter.DEFAULT_MAX_CHARS 与 DEFAULT_MAX_CHARS_EN，立即跑 pytest 确认无回归",
    "Phase 1.2 - 在 text_splitter 增加 OVERLAP_SECONDS 与 _assign_timestamps_with_overlap，保持旧函数入口不变",
    "Phase 1.3 - 在 asr_corrector 引入 PUNCT_MIN_LEN 常量并替换字面量 10",
    "Phase 1.4 - 添加 test_text_splitter_phase1.py 与 test_asr_corrector_phase1.py（5 + 3 个用例）",
    "Phase 1.5 - 跑全量 pytest：32 旧 + 8 新 = 40 个全通过；记录耗时；准备提交信息（feat(splitter): align defaults and add overlap patch）",
    "Phase 1.6 - 请主人审阅 diff，签字确认后合并 main",
    "Phase 2 - 拉分支 feat/phase2-scene-preset；先把现有接口快照保存（main.py 现有 transcribe 路由签名、cloud_engine / sensevoice_engine / segment_merger 的 transcribe/merge 签名）",
    "Phase 2.1 - 新建 scene_preset.py：枚举 + SCENE_PARAMS + resolve_scene_params helper，加单元测试",
    "Phase 2.2 - text_splitter 增加 v2 函数 + word_boundary 选项 + is_mostly_cjk 阈值 override，保持 v1 入口兼容",
    "Phase 2.3 - segment_merger / cloud_engine / sensevoice_engine / asr_corrector 全部加 scene_preset 可选 kwargs，默认 None 走 DEFAULT_*；不改任何调用方既有行为",
    "Phase 2.4 - main.py 路由加 Form(None) scene_preset 字段，透传到下层；OpenAPI 自动出现新可选字段",
    "Phase 2.5 - 前端下拉控件提交 scene_preset；本地 5 个场景各跑 5 分钟样本，主人人工打分 ≥ 4/5",
    "Phase 2.6 - 5 个 e2e 测试（test_scene_preset.py）落地；跑全量 pytest；提交 feat(scene-preset): introduce ScenePreset for downstream tunables",
    "Phase 2.7 - 主人签字后合 main；Phase 3 留给 B 方案 + CTC forced alignment（不在本次范围）"
  ],
  "test_cases": [
    {
      "id": "TXT-01",
      "module": "text_splitter",
      "name": "DEFAULT_MAX_CHARS equals 25 after Phase 1",
      "type": "unit",
      "assertion": "from app.core.text_splitter import DEFAULT_MAX_CHARS; assert DEFAULT_MAX_CHARS == 25"
    },
    {
      "id": "TXT-02",
      "module": "text_splitter",
      "name": "DEFAULT_MAX_CHARS_EN equals 42 after Phase 1",
      "type": "unit",
      "assertion": "from app.core.text_splitter import DEFAULT_MAX_CHARS_EN; assert DEFAULT_MAX_CHARS_EN == 42"
    },
    {
      "id": "TXT-03",
      "module": "text_splitter",
      "name": "split_text_at_boundaries honors new max_chars=25 for CJK",
      "type": "unit",
      "assertion": "Build a 60-char CJK string with three 。 punctuation; assert max len(seg['text']) <= 25"
    },
    {
      "id": "TXT-04",
      "module": "text_splitter",
      "name": "_assign_timestamps_with_overlap applies 0.05s overlap to non-first parts",
      "type": "unit",
      "assertion": "seg={start:0,end:10}; parts=['A','B','C']; assert result[1]['start']==(10/3)+0.05 and result[2]['start']==(10/3)*2+0.05"
    },
    {
      "id": "TXT-05",
      "module": "text_splitter",
      "name": "OVERLAP_SECONDS constant is 0.05",
      "type": "unit",
      "assertion": "from app.core.text_splitter import OVERLAP_SECONDS; assert OVERLAP_SECONDS == 0.05"
    },
    {
      "id": "ASR-01",
      "module": "asr_corrector",
      "name": "PUNCT_MIN_LEN constant exists and is 10",
      "type": "unit",
      "assertion": "from app.core.asr_corrector import PUNCT_MIN_LEN; assert PUNCT_MIN_LEN == 10"
    },
    {
      "id": "ASR-02",
      "module": "asr_corrector",
      "name": "_apply_punct_rules appends 。 for >10-char text without trailing punct",
      "type": "unit",
      "assertion": "text='这是一段超过十字的中文文本' -> ends with '。'"
    },
    {
      "id": "ASR-03",
      "module": "asr_corrector",
      "name": "_apply_punct_rules does NOT append 。 for <=10-char text",
      "type": "unit",
      "assertion": "text='十个字符abcde' (10 chars) -> text unchanged"
    },
    {
      "id": "ASR-04",
      "module": "asr_corrector",
      "name": "Regression: 32 existing tests still pass",
      "type": "regression",
      "assertion": "pytest backend/tests/ -k 'asr_corrector' -v shows zero failures"
    },
    {
      "id": "SCN-01",
      "module": "scene_preset",
      "name": "resolve_scene_params(None) returns DEFAULT values",
      "type": "unit",
      "assertion": "params['max_chars']==25 (== text_splitter.DEFAULT_MAX_CHARS); params['word_boundary'] is False (default non-mixed)"
    },
    {
      "id": "SCN-02",
      "module": "scene_preset",
      "name": "rap preset caps line at 12 chars",
      "type": "unit",
      "assertion": "params['max_chars']==12"
    },
    {
      "id": "SCN-03",
      "module": "scene_preset",
      "name": "mixed_language preset enables word_boundary and looser cjk threshold",
      "type": "unit",
      "assertion": "params['word_boundary'] is True; params['cjk_threshold']==0.5"
    },
    {
      "id": "SCN-04",
      "module": "text_splitter v2",
      "name": "split_text_for_subtitles_v2 with word_boundary=True keeps long English word intact",
      "type": "unit",
      "assertion": "text='internationalization is great'; v2(max_len=10, word_boundary=True) keeps 'internationalization' as one chunk"
    },
    {
      "id": "SCN-05",
      "module": "scene_preset end-to-end",
      "name": "rap preset e2e: 5-min sample produces segments with max 12 chars",
      "type": "integration",
      "assertion": "max(len(s['text']) for s in segments) <= 12"
    },
    {
      "id": "SCN-06",
      "module": "scene_preset end-to-end",
      "name": "lecture preset e2e: 5-min sample produces segments with max 20 chars",
      "type": "integration",
      "assertion": "max(len(s['text']) for s in segments) <= 20"
    },
    {
      "id": "SCN-07",
      "module": "scene_preset end-to-end",
      "name": "mixed preset e2e: no English word is broken mid-word",
      "type": "integration",
      "assertion": "No segment text contains whitespace inside a long English word (>8 letters)"
    },
    {
      "id": "SCN-08",
      "module": "scene_preset end-to-end",
      "name": "podcast preset e2e: max line <=35 chars",
      "type": "integration",
      "assertion": "max(len(s['text']) for s in segments) <= 35"
    },
    {
      "id": "SCN-09",
      "module": "scene_preset end-to-end",
      "name": "asmr preset e2e: max segment duration <=15s while chars <=35",
      "type": "integration",
      "assertion": "all (s['end']-s['start']) <= 15.0 and len(s['text']) <= 35"
    },
    {
      "id": "API-01",
      "module": "main API contract",
      "name": "POST transcribe without scene_preset still works (signature-compatible)",
      "type": "contract",
      "assertion": "Send existing payload; assert 200 OK and same response shape as Phase 1 baseline."
    },
    {
      "id": "API-02",
      "module": "main API contract",
      "name": "POST transcribe with scene_preset=rap routes to rap params",
      "type": "contract",
      "assertion": "Send scene_preset='rap'; assert response segments have max line 12 chars."
    }
  ],
  "definition_of_done": [
    "Phase 1: text_splitter.DEFAULT_MAX_CHARS=25, DEFAULT_MAX_CHARS_EN=42, OVERLAP_SECONDS=0.05, asr_corrector.PUNCT_MIN_LEN=10，全部已落地且被测试断言。",
    "Phase 1: 现有 32 个单测全部通过；新增 5 个 text_splitter 用例 + 3 个 asr_corrector 用例全部通过（总计 40 个）。",
    "Phase 1: cloud_engine 与 sensevoice_engine 4 + 1 个调用点的最大行长度与新 DEFAULT_MAX_CHARS=25 对齐（验证：插入 50 字 CJK 文本，max(seg['text']) == 25）。",
    "Phase 1: 时间戳连续性验证：连续 3 段 [0,5), [5,10), [10,15) 经 _assign_timestamps_with_overlap 包装后变成 [0,5), [5.05,10), [10.05,15)，无回退。",
    "Phase 1: 标点插入行为不变：10/11 字符边界行为与基线一致。",
    "Phase 1: 主人审阅 diff 并签字后合并 main，commit 信息遵循 Conventional Commits（feat/fix/refactor），不含 AI 签名。",
    "Phase 2: scene_preset.py 落地，5 个枚举值与 SCENE_PARAMS 映射表、resolve_scene_params helper 全部存在。",
    "Phase 2: 5 个下游模块（text_splitter、segment_merger、cloud_engine、sensevoice_engine、asr_corrector）全部接受 scene_preset 可选 kwargs，默认 None 走 DEFAULT_*，签名兼容。",
    "Phase 2: main.py transcribe 路由新增 scene_preset Form 字段为可选；既有 POST 请求（不传 scene_preset）行为零变化（API-01 测试通过）。",
    "Phase 2: 前端下拉控件提交 scene_preset；5 种场景各 1 个 5 分钟样本跑通；人工评审可读性 4/5 以上。",
    "Phase 2: 5 个 e2e 用例（SCN-05 ~ SCN-09）全部通过；2 个 API 契约测试（API-01、API-02）全部通过。",
    "Phase 2: word_boundary 模式开启时，无英文单词被切碎（SCN-04、SCN-07 验证）。",
    "Phase 2: rap 模式单行 ≤12 字、asmr 模式单段 ≤15s（SCN-05、SCN-09 验证）。",
    "Phase 2: 主人审阅 diff 并签字后合并 main；Phase 3（TimeAuthority + CTC）不在本计划执行范围。",
    "全局: 全量 pytest 通过；不破坏 sensevoice_engine、cloud_engine、engine_base、llm_corrector、subtitle_formatter、vad_detector、audio_processor、ffmpeg_downloader、main 任何既有导入。",
    "全局: 所有 commit 信息符合 `feat(scope): subject` 格式，subject 不超过 50 字符，无 AI 工具标识。"
  ]
}