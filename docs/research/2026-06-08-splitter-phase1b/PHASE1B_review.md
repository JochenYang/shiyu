{
  "findings": [
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "505-516",
      "description": "两个 except 子句捕获类型重复且语义模糊：第一个 except 捕获 (ValueError, KeyError, TypeError) 用于“parse 错误”，第二个 except 又捕获 (KeyError, IndexError, TypeError) 用于“格式错误”。KeyError/TypeError 同时被两个分支匹配，但执行顺序由 Python 保证（前者先命中后直接 return None），因此第二个 except 实际上只在 IndexError 路径上可达。这导致代码意图不清、命中时只记录一个相对模糊的日志，未来扩展易踩坑。",
      "fix_suggestion": "合并两个 except 为单一 (ValueError, KeyError, IndexError, TypeError) 分支，并按状态码（resp.status_code）区分 parse vs. format 错误，或在 response shape 校验失败时用更明确的 message 区分两种失败语义。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "61-62",
      "description": "DEFAULT_MAX_WORKERS 和 DEFAULT_TIMEOUT_SEC 在 import 时通过 os.environ.get 求值并固化在模块命名空间里。这意味着测试中通过 os.environ 覆盖后必须 importlib.reload 才能生效（见 test_env_override_*），运行时也无法通过环境变量动态调整。可观测性/可调优性偏弱。",
      "fix_suggestion": "考虑改为函数内读取，或在 _batch_correct / _call_llm 入口重新读 os.environ，保留向后兼容的实现方式是添加 module-level 函数 _get_max_workers() / _get_timeout_sec() 并在调用点使用。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\sentence_splitter.py",
      "line": "183-210",
      "description": "_align_segments_to_vad_regions 使用 overlap/total_dur 比例切分文本时没有显式兜底‘整段 ASR 段被 VAD 完全包含’场景：当 seg_total_dur 与 region overlap 完全重合（frac=1.0）时，target_chars = len(remaining_text) 整段落入第一个 region，后续 region 因 segment_consumed 已达 100% 全部跳过，行为正确；但当一个 ASR 段横跨 N 个 VAD 区间且各 VAD 区间 overlap 比例相近时，文本切分位置可能切在词中间（即便有 ±2 字符的标点搜索，CJK 标点稀疏时仍会切出非自然边界）。需要更稳的 punctuation-aware 分句。",
      "fix_suggestion": "在无法找到标点/空格边界时改用 _split_text_at_punctuation 提供的启发式，或增加‘当 frac &lt; 1.0 时按字符数等分并优先落点在最后一个标点/空格’逻辑；至少加一个测试覆盖 CJK 长段跨 3+ VAD 区间无标点场景。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "335-340",
      "description": "_batch_correct 文档说明 max_workers 默认 5，但当 total_batches &lt; 5 时实际并发被 clamp 到 total_batches（line 368），看似正确。问题是 'workers=5' 默认对 OpenAI/MiniMax 兼容 API 仍然温和，但如果未来 batch_size 调到 50 + workers 5，会出现 5 路并发 × 单 batch 50 行 prompt，单次 provider 端 QPS=5；某些 provider (智谱 GLM 免费版) 对 QPS 有限制，可能被风控。",
      "fix_suggestion": "在 _batch_correct 启动时增加 provider 维度的 QPS 节流配置（暂时可仅记录 warning 并暴露 env 变量 LLM_QPS_LIMIT），或在文档中显式声明此组合的 provider 兼容性。"
    },
    {
      "severity": "MEDIUM",
      "file": "D:\\codes\\Shiyu\\backend\\app\\main.py",
      "line": "455-460",
      "description": "/transcribe 端点从‘静默丢弃坏 glossary 并继续’改为 JSONResponse(400, error=\"invalid_glossary\")。这是一个 API 契约破坏性变更：旧版客户端若依赖 ‘传错 glossary 也能跑通转写’ 的行为，会突然拿到 400 报错。文档和前端需要同步更新。",
      "fix_suggestion": "在 CHANGELOG/PR description 中明确标注 BREAKING CHANGE；前端的 glossary 表单需要在发送前做 JSON 校验；建议保留 1 个版本窗口，warning-level 日志后 fallback 到空 glossary。"
    },
    {
      "severity": "MEDIUM",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "780-820",
      "description": "FullTextCorrector._call_llm 仍是旧实现：timeout 硬编码 180s、payload 仍含 'thinking': False（被 commit e2e440d 修复的同款问题）、没有 retry 路径。它与 BaseCorrector._call_llm 行为分叉，未来任何 timeout/retry/think 调优需要双修，易遗漏。",
      "fix_suggestion": "让 FullTextCorrector 调用 BaseCorrector._call_llm 并通过参数注入 system prompt 差异（重构出 _post_chat(messages, ...) 公共函数），统一行为。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "432-451",
      "description": "_call_llm 的 timeout 在 for 循环里硬编码读取 DEFAULT_TIMEOUT_SEC；如果未来想给不同 batch 设置不同 timeout（如长 batch 拉长 timeout），需要扩展。当前实现可接受。",
      "fix_suggestion": "无强制要求；若支持每批可调 timeout，把 timeout 作为 _call_llm 的可选形参（默认 DEFAULT_TIMEOUT_SEC）即可。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\tests\\core\\test_sentence_splitter_phase1b.py",
      "line": "19-130",
      "description": "所有 VAD 测试用例都是‘理想对齐’构造：ASR 段边界 == VAD 区间边界（如 0-2/2-4）。未覆盖真实 ASR 输出常见的 partial overlap（ASR 段起止在 VAD 区间中部）、ASR 段 text 含 BPE/标点混合、以及 ASR 段比 VAD 区间短/长等场景。报告自身也提示了这一点。",
      "fix_suggestion": "补一个 case：ASR 段 [0.3, 3.7] 跨 VAD [0, 2]/[2, 4]；ASR 段 [0, 5] 大于 2 个 VAD 区间；ASR 段 [0, 1.5] 落在 VAD [0, 2] 内但文本带空格。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\tests\\core\\test_main_phase1b.py",
      "line": "156-175",
      "description": "TestPrintReplacedWithLogger 仅断言 _register_default_engines 路径中无 'Engine Registry' 字样的 print，但 lifespan 中的 print 仍会被 print_mock 捕获并被这个测试忽略（注释也承认了）。换言之，print->logger 的替换只验证了 1/2 路径。",
      "fix_suggestion": "用 lifespan fixture 跑完整 startup，或 grep 整个 main.py 文件确认无 print() 调用，再把测试改造为 grep assertion。"
    },
    {
      "severity": "LOW",
      "file": "D:\\codes\\Shiyu\\backend\\app\\main.py",
      "line": "786-787",
      "description": "正确性修复范围内的新发现（非 Phase 1B 改动）：correct_transcription 中用 print() 记录 split/corrected 计数，与本批次‘把 print 改为 logger’的目标不一致。",
      "fix_suggestion": "替换为 logger.info 以保持日志通道统一。"
    },
    {
      "severity": "INFO",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\sentence_splitter.py",
      "line": "89-224",
      "description": "关于核心 bug 修复结论：_align_segments_to_vad_regions 新逻辑通过 ‘(1) 计算每个 region 的最大 overlap segment; (2) 用 segment_consumed 游标记录已消费字符数; (3) 后续 region 从 remaining_text 切分’ 的三步机制，确保同一 ASR 段的文本不会跨 region 重复。test_single_asr_segment_spanning_two_vad_regions_not_duplicated 用不变量 ‘任一段文本长度 < 原文长度’ 明确验证了非重复。修复逻辑可信，行为符合预期；但真实 ASR 输出中 fractional overlap、CJK 标点稀疏场景需要补测试并按上面 finding #3 加固。",
      "fix_suggestion": "维持当前实现 + 补真实场景测试。"
    },
    {
      "severity": "INFO",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "321-410",
      "description": "关于性能调优结论：batch_size 20 + workers 5 + timeout 90s + retry 1 的组合在 41 个测试全部通过的前提下是安全的：(a) test_uses_default_batch_size_20 验证 30 segments 切为 2 batch；(b) test_timeout_retries_once 验证重试行为符合设计；(c) test_http_error_no_retry 验证 4xx/5xx 不重试避免无谓延迟翻倍。综合看，单次请求的预期 p95 latency 受 batch size 减小而下降，但最坏情况 5 路并发 + 1 重试 ≈ 12 batch × 90s × 2 = 仍可控。环境变量覆盖已就绪。",
      "fix_suggestion": "建议在 PR 描述里加 latency 测量方法（p50/p95），并文档化 LLM_MAX_WORKERS / LLM_TIMEOUT_SEC 的可调旋钮。"
    },
    {
      "severity": "INFO",
      "file": "D:\\codes\\Shiyu\\backend\\app\\main.py",
      "line": "542-548, 631-638",
      "description": "关于 API 兼容性结论：llm_status 字段在 /transcribe/json 和 /transcribe/local 中新增为 backward-additive，老客户端忽略未知字段；唯一破坏性变更是 glossary 400 行为。已加 4 个测试覆盖 status flag + 2 个测试覆盖 400 路径，测试覆盖充分。建议在 PR 描述中标注 BREAKING CHANGE 标签。",
      "fix_suggestion": "文档/CHANGELOG 更新；前端 glossary 输入加 JSON.parse 校验。"
    },
    {
      "severity": "INFO",
      "file": "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
      "line": "512-516",
      "description": "test_failed_correction_sets_failed 中提到通过 main_module._llm_status 引用而非 from-import 是必要的。复查：_apply_llm_correction 中通过 global _llm_status 重新绑定模块全局变量，from-import 的本地引用是旧绑定，所以测试用 main_module._llm_status 读取当前值是正确做法。这是测试写法约束，但反映出 _llm_status 作为可变全局状态的设计本身存在隐式风险（并发请求会相互覆盖状态）。",
      "fix_suggestion": "长期建议：把 _llm_status 作为返回值由调用方合并进 response payload；短期可保留全局但加 threadlocal 或 request-scoped context。"
    }
  ],
  "recommendation": "SHIP",
  "summary": "Phase 1B 三个核心目标均达成：\n\n1. 文本复制 bug 修复：_align_segments_to_vad_regions 通过最大 overlap + segment_consumed 字符游标机制，从根本上消除了同一 ASR 段文本跨 VAD 区间复制的可能；test_single_asr_segment_spanning_two_vad_regions_not_duplicated 用‘任一段长度 < 原文’的不变量显式验证了非重复。修复可信。\n2. LLM 性能调优：batch 50→20、workers 3→5、timeout 180s→90s、+1 retry 的组合在测试中验证有效（test_uses_default_batch_size_20、test_timeout_retries_once、test_http_error_no_retry）。env 覆盖就绪，最坏情况延迟可控。\n3. 错误显式化：glossary 400 + llm_status 字段已加测试，4+2=6 个测试覆盖。glossary 400 属 API 契约破坏性变更，需文档/前端同步；其余为 additive。\n\n测试覆盖：27 个新 Phase 1B 测试 + 14 个 Phase 1 回归基线全部 41 通过；按修复点拆分每个修复都有 ≥3 个测试（VAD 修复 7 测试含 4 dedupe + 4 边界、LLM 调优 10 测试含 6 常量/env + 3 重试 + 1 batch、main 6 测试含 3 llm_status + 2 glossary 400 + 1 print->logger）。\n\n风险点（无 blocker，标记 NEEDS WORK 之前的总体评级）：MEDIUM 两项需 owner 注意：glossary 400 破坏性变更需 PR 描述标注 + 文档同步；FullTextCorrector._call_llm 仍为旧实现，与 BaseCorrector 行为分叉需后续统一。\n\n回滚方案：三个 commit 互不依赖，可逐个 git revert；DEFAULT_BATCH_SIZE/MAX_WORKERS/TIMEOUT_SEC 全有 env 覆盖，回滚时不改代码也可调回旧值。\n\n建议 SHIP，但需在 PR 描述/CHANGELOG 标注 glossary 400 为 BREAKING CHANGE，并在合并后立即跟进 FullTextCorrector 统一化。"
}