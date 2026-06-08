{
  "dependency_order": [
    "fix-error-explicit (P0) — make silent error paths observable first; provides the diagnostic signals needed to validate the other two fixes",
    "fix-llm-perf (P0) — reduce batch size + raise worker pool + add timeout-aware fallback; depends on error-explicit for correct LLM failure reporting",
    "fix-vad-dedupe (P0) — rewrite _align_segments_to_vad_regions with forward-assignment (VAD-first) and BFS-based dedupe; independent of the above but verified last because it produces the most user-visible symptom (duplicate text)"
  ],
  "per_fix_interfaces": {
    "item": [
      {
        "id": "fix-error-explicit",
        "scope": "backend/app/core/llm_corrector.py + backend/app/main.py + backend/app/core/engine_base.py",
        "signatures": {
          "item": [
            "class LLMErrorKind(str, Enum): TIMEOUT | NETWORK | HTTP_4XX | HTTP_5XX | PARSE | EMPTY | UNKNOWN",
            "@dataclass class LLMCallResult: ok: bool; error_kind: LLMErrorKind | None; error_message: str; corrected_lines: list[str] | None; elapsed_ms: int; provider: str; model: str",
            "def _apply_llm_correction(segments: list[dict], settings: dict) -> dict  # returns {\"applied\": bool, \"segments\": list[dict], \"llm_meta\": {\"ok\": bool, \"error_kind\": ..., \"elapsed_ms\": int, \"attempted\": bool, \"skipped_reason\": str | None}}",
            "class EngineRegistry: def register(self, name: str, factory: Callable[[], BaseTranscriber]) -> None; def available_engines(self) -> list[dict]  # returns [{name, available, error_kind?, error_message?}]",
            "GET /health -> {\"ok\": bool, \"engines\": {name: {\"available\": bool, \"error\": str|None}}, \"vad_model_loaded\": bool, \"llm_configured\": bool}",
            "POST /transcribe -> ResponseModel { ..., llm_meta: LLMCallResult, vad_used: bool, vad_error: str|None }",
            "logger = logging.getLogger(__name__) replacement of all print() calls in lifespan _register_default_engines _apply_vad_sentence_segmentation",
            "json.JSONDecodeError handler in 3 glossary endpoints returns HTTP 400 with {\"error\": \"invalid_glossary\", \"detail\": str} instead of silently dropping"
          ]
        }
      },
      {
        "id": "fix-llm-perf",
        "scope": "backend/app/core/llm_corrector.py",
        "signatures": {
          "item": [
            "DEFAULT_BATCH_SIZE: int = 20  # was 50",
            "DEFAULT_MAX_WORKERS: int = 6  # was 3; env override LLM_MAX_WORKERS",
            "DEFAULT_TIMEOUT_SEC: int = 90  # was 180; env override LLM_TIMEOUT_SEC",
            "DEFAULT_RETRIES: int = 1  # was 0; one immediate retry on timeout/network only",
            "@dataclass class BatchResult: batch_index: int; ok: bool; lines: list[str]; error_kind: LLMErrorKind; error_message: str; elapsed_ms: int; retries: int",
            "def _call_llm_batch(batch_lines: list[str], settings: dict) -> BatchResult  # single batch; pure function, no shared state",
            "def correct_segments(segments: list[dict], settings: dict) -> dict  # orchestrates ThreadPoolExecutor + collects BatchResult; on partial failure, returns original lines for failed batches (no chunk-split fallback in v1)",
            "def _http_session() -> requests.Session  # shared Session with HTTPAdapter pool_connections=DEFAULT_MAX_WORKERS*2, pool_maxsize=DEFAULT_MAX_WORKERS*2 for connection reuse",
            "def on_batch_done(future, batch_index, total_batches, callback)  # progress callback for streaming front-end updates (optional use)"
          ]
        }
      },
      {
        "id": "fix-vad-dedupe",
        "scope": "backend/app/core/asr_corrector.py (or sentence_splitter.py where alignment lives)",
        "signatures": {
          "item": [
            "@dataclass class VADRegion: idx: int; start: float; end: float; text: str  # immutable, text is empty until assigned",
            "def build_vad_regions(speech_intervals: list[tuple[float,float]], min_dur: float = 0.3) -> list[VADRegion]  # clip, filter, materialize region list (no text yet)",
            "def assign_segments_to_regions(segments: list[dict], regions: list[VADRegion]) -> list[VADRegion]  # forward assignment: for each region, find the ASR segment with maximum overlap whose center falls inside the region; assign text once; tie-break by earlier start time",
            "def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float",
            "def _stitch_adjacent_regions(regions: list[VADRegion], gap_threshold: float = 0.2) -> list[VADRegion]  # optional: merge VAD regions with sub-threshold gap to reduce fragmentation",
            "def _format_subtitle_from_regions(regions: list[VADRegion]) -> list[dict]  # produce {start, end, text} for subtitle output, drops regions with empty text",
            "def _deduplicate_segments(segments: list[dict]) -> list[dict]  # kept as helper, but only applied to overlapping ASR inputs (not after VAD assignment)",
            "REMOVED: _align_segments_to_vad_regions  # replaced by the two-step build + assign above"
          ]
        }
      }
    ]
  },
  "per_fix_tests": {
    "item": [
      {
        "id": "fix-error-explicit",
        "tests": {
          "item": [
            "test_llm_timeout_returns_error_kind_timeout: monkeypatch requests.post to sleep 200s, assert LLMCallResult.error_kind == TIMEOUT, applied == False, error_message contains \"timeout\"",
            "test_llm_http_500_classified: monkeypatch response.status_code=500, assert error_kind == HTTP_5XX, llm_meta.ok == False, log line uses logger.error not print",
            "test_engine_registry_reports_unavailable: force ImportError for sensevoice_local in _register_default_engines, call /health, assert engines.sensevoice_local.available == False and engines.sensevoice_local.error contains \"import\"",
            "test_glossary_invalid_json_returns_400: POST /glossary with body '{\"entries\":\"not json\"}', assert HTTP 400, body has error=invalid_glossary, custom_glossary not silently empty",
            "test_vad_failure_flagged: monkeypatch vad_detector to raise, assert response.vad_error is not None and vad_used == False, original segments preserved",
            "test_llm_corrector_failure_does_not_set_applied: monkeypatch LLM to raise, assert response.llm_meta.applied == False, segments unchanged, NOT indistinguishable from LLM success"
          ]
        }
      },
      {
        "id": "fix-llm-perf",
        "tests": {
          "item": [
            "test_batch_split_uses_20: 150 segments produce 8 batches of size <= 20 (last may be smaller); assert DEFAULT_BATCH_SIZE == 20 in module import check",
            "test_three_workers_runs_concurrent: 6 batches with mock 30s latency, measure wall_clock < 60s (proves parallel execution), assert max_workers default == 6",
            "test_timeout_90s_returns_fallback: monkeypatch requests.post to sleep 100s, assert BatchResult.error_kind == TIMEOUT, original lines preserved for that batch, no exception bubbles",
            "test_retry_on_timeout_only: monkeypatch to fail with Timeout once then succeed, assert BatchResult.retries == 1, ok == True; HTTP_400 does NOT trigger retry",
            "test_connection_reuse: use mock to count TCP connections in requests.Session, 6 batches should share session, assert <= 2 connections (vs 6 in old code)",
            "test_partial_failure_preserves_unchanged_segments: 3 batches, batch 1 ok, batch 2 timeout, batch 3 ok; assert output length == input length; failed-batch lines == original lines; successful-batch lines == corrected",
            "test_wall_clock_under_3x_serial_budget: 6 batches × 30s mock = 180s serial; with 6 workers and 20-line batches, assert wall_clock < 90s (5x speedup)"
          ]
        }
      },
      {
        "id": "fix-vad-dedupe",
        "tests": {
          "item": [
            "test_no_duplicate_when_asr_segments_overlap_vad_region: two ASR segments both overlap VAD region 0 with text \"这里还有一个变量\" and \"这里还有一个变量根据项目\"; assert output region 0 text == \"这里还有一个变量根据项目\" (no \"这里还有一个变量...这里还有一个变量\")",
            "test_forward_assignment_one_text_per_region: 3 ASR segments, 2 VAD regions, all overlap region 0; assert exactly one ASR segment's text is assigned to region 0 (max-overlap-center tiebreak)",
            "test_uncovered_region_uses_nearest_segment: VAD region with no overlap to any ASR segment, region.start in middle of segment A; assert region gets A's text via nearest-center rule",
            "test_empty_region_dropped: VAD region with no ASR coverage within max-distance threshold; assert _format_subtitle_from_regions omits that region",
            "test_subtitle_timestamps_contiguous: 5 regions, assert output is sorted by start, end_i == start_{i+1} (no gaps > 0.05s tolerance)",
            "test_regression_existing_dedup: two ASR segments where seg B fully contains seg A, after _deduplicate_segments then VAD assign; assert no text duplication in final subtitle",
            "test_no_allocation_when_vad_disabled: speech_intervals empty, build_vad_regions returns [], pipeline short-circuits to original segments unchanged, no empty text in output"
          ]
        }
      }
    ]
  },
  "unified_risks": [
    "R1 (HIGH): fix-vad-dedupe changes the alignment contract — downstream subtitle_formatter and SRT/JSON exporters may depend on region order or count. Mitigate: keep VAD region ordering stable; expose _format_subtitle_from_regions with same return shape; golden-file test against current 3 sample outputs in backend/tests/.",
    "R2 (HIGH): fix-llm-perf changes DEFAULT_BATCH_SIZE 50→20 changes prompt cache hit rate (if provider caches prompts). Mitigate: measure before/after with provider response headers; document in CHANGELOG; env override LLM_BATCH_SIZE keeps current value as fallback option.",
    "R3 (MEDIUM): max_workers=6 increases concurrent TLS handshakes to LLM provider; risk of 429 rate-limit on providers with strict QPS. Mitigate: env LLM_MAX_WORKERS default 6 but cap at 8; add explicit retry-with-backoff on 429 in v1.1; log warning on 429.",
    "R4 (MEDIUM): Connection pool via requests.Session requires Session lifecycle management; if session closed prematurely, second batch fails. Mitigate: session created per correct_segments() call (cheap), closed via context manager.",
    "R5 (MEDIUM): fix-error-explicit changes /health and /transcribe response shape; frontend may parse old shape. Mitigate: add new fields additive (engines, llm_meta, vad_error); keep old fields (ok, segments, subtitles) unchanged; document in docs/api-changes.md.",
    "R6 (LOW): glossary 400 response breaks any client that silently retries on 4xx. Mitigate: keep custom_glossary default {} in body so client can still see what was loaded; only the malformed entries are rejected.",
    "R7 (LOW): VAD forward-assignment changes the heuristic for \"which ASR segment owns this region\"; if ASR produces out-of-order segments (rare), order may differ. Mitigate: stable sort by segment.start ascending before assignment.",
    "R8 (LOW): fix-error-explicit adds logger.warning/error where print() was used; existing log handlers may double-log. Mitigate: confirm logging configured in main.py lifespan with single handler; tests assert log count.",
    "R9 (CROSS-CUTTING): all three fixes are in llm_corrector.py/asr_corrector.py; merge conflicts likely if branches run in parallel. Mitigate: serialize commits as below, do not parallelize; run code-reviewer after each commit.",
    "R10 (TESTING): real LLM tests require API key and are slow/flaky. Mitigate: mock requests.post in unit tests; keep one smoke test in backend/tests/integration/ that runs only when LLM_API_KEY is set."
  ],
  "definition_of_done": [
    "[ ] fix-error-explicit: all 6 unit tests pass; print() calls in lifespan/_register_default_engines/_apply_llm_correction/_apply_vad_sentence_segmentation replaced with logger; /health returns engines availability map; /transcribe returns llm_meta with error_kind on failure; glossary 400 on bad JSON; code-reviewer recommendation == SHIP",
    "[ ] fix-llm-perf: DEFAULT_BATCH_SIZE==20, DEFAULT_MAX_WORKERS==6, DEFAULT_TIMEOUT_SEC==90, DEFAULT_RETRIES==1, requests.Session reused; all 7 unit tests pass; integration smoke test on 150-segment fixture completes in < 90s (down from ~180s+); code-reviewer recommendation == SHIP",
    "[ ] fix-vad-dedupe: _align_segments_to_vad_regions removed; build_vad_regions + assign_segments_to_regions implemented; all 7 unit tests pass; golden-file regression test for 3 existing samples passes (zero duplicate substrings, contiguous timestamps); code-reviewer recommendation == SHIP",
    "[ ] Cross-cutting: docs/api-changes.md updated; CHANGELOG.md updated for 1.2.0; full test suite green; coverage on changed files >= 80%; no new lint warnings; manual E2E: transcribe 3-min sample with VAD on + LLM on, confirm no duplicate text in output SRT; manual E2E: kill LLM key mid-run, confirm frontend shows \"LLM 失败 (timeout)\" toast",
    "[ ] Rollback: each fix is an independent commit, revertible via git revert <sha> without affecting others; documented in PR description"
  ],
  "commit_split": [
    "commit 1: feat(error): replace silent except/print with structured LLMCallResult + engine registry diagnostics (fix-error-explicit) — files: backend/app/core/llm_corrector.py, backend/app/core/engine_base.py, backend/app/main.py, backend/app/core/vad_detector.py (logger only), backend/app/api/glossary.py; +tests backend/tests/test_error_explicit.py; ~280 lines diff",
    "commit 2: perf(llm): shrink batch 50->20, raise workers 3->6, timeout 180->90s, add session reuse + retry (fix-llm-perf) — files: backend/app/core/llm_corrector.py only; +tests backend/tests/test_llm_perf.py; ~120 lines diff (replaces lines 55, 322, 346-354, 430 area)",
    "commit 3: fix(asr): rewrite VAD alignment to forward-assignment, eliminate text duplication (fix-vad-dedupe) — files: backend/app/core/asr_corrector.py (or sentence_splitter.py per current owner of _align_segments_to_vad_regions); +tests backend/tests/test_vad_dedupe.py; +docs docs/vad-alignment.md; ~260 lines diff (replaces lines 28-86 _deduplicate and 150-209 _align)",
    "commit 4 (optional, docs-only): docs: API changes for /health, /transcribe llm_meta field, glossary 400 behavior — files: docs/api-changes.md, CHANGELOG.md; ~60 lines diff"
  ],
  "total_diff_lines": 720
}