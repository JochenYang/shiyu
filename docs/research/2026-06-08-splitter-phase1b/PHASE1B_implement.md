{
  "files_changed": [
    "D:\\codes\\Shiyu\\backend\\app\\core\\sentence_splitter.py",
    "D:\\codes\\Shiyu\\backend\\app\\core\\llm_corrector.py",
    "D:\\codes\\Shiyu\\backend\\app\\main.py"
  ],
  "tests_added": [
    "D:\\codes\\Shiyu\\backend\\tests\\core\\test_sentence_splitter_phase1b.py",
    "D:\\codes\\Shiyu\\backend\\tests\\core\\test_llm_corrector_phase1b.py",
    "D:\\codes\\Shiyu\\backend\\tests\\core\\test_main_phase1b.py"
  ],
  "test_results": "============================== 41 passed in 0.56s ==============================\n- test_asr_corrector_phase1.py: 4 passed (Phase 1, regression baseline)\n- test_text_splitter_phase1.py: 10 passed (Phase 1, regression baseline)\n- test_sentence_splitter_phase1b.py: 11 passed (NEW — VAD dedupe fix)\n- test_llm_corrector_phase1b.py: 10 passed (NEW — perf: batch/workers/timeout/retry)\n- test_main_phase1b.py: 6 passed (NEW — llm_status, glossary 400, logger)\n\nTotal: 41 passed, 0 failed, 0 errors. No regression.",
  "commit_messages": [
    "fix(sentence-splitter): rewrite vad align to prevent text duplication",
    "perf(llm-corrector): lower batch size, raise workers, add timeout-aware retry",
    "fix(main): expose llm_status in api responses, handle glossary json error"
  ],
  "risks_encountered": [
    "VAD align algorithm change: BFS-dedupe now uses a forward-consumed cursor per ASR segment to slice text proportionally across VAD regions. Behavior is verified by 4 dedupe tests but real ASR outputs (overlapping timestamps, punctuation patterns) may differ from test fixtures; recommend running integration tests on real audio before merge.",
    "LLM timeout reduced 180s -> 90s may increase failure rate for slow providers. Documented in env override LLM_TIMEOUT_SEC; recommend frontend-side hint to users.",
    "_call_llm retry on timeout/network (1 retry) may double latency in worst case. Documented; user can override retries=0 by editing DEFAULT_RETRIES if needed.",
    "main.py /transcribe and /transcribe/local now return JSONResponse(400) on bad glossary instead of silently dropping. This is a contract change for any client depending on the prior 500/swallowed behavior.",
    "Some test runs (test_failed_correction_sets_failed) required using main_module._llm_status instead of imported _llm_status because the global rebinding inside _apply_llm_correction does not propagate to local import names. Tests now use the canonical reference."
  ],
  "next_steps": [
    "Run integration test on a real audio file to confirm VAD-align dedupe behaves as expected end-to-end (single ASR segment with long text spanning 2-3 VAD regions).",
    "Consider adding a CLI flag to LLM_MAX_WORKERS / LLM_TIMEOUT_SEC so tests and CI can pin values without env mutation.",
    "Once owners confirm behavior, stage and commit each fix as a separate commit (per commit_messages), then push and open PR for review.",
    "Update frontend to display llm_status in transcribe responses (success/failed/skipped) for better user feedback."
  ]
}