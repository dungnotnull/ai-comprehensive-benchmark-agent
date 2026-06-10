# ai-benchmark-agent — Test Scenarios

## Scenario 1: Golden Path — Proxy Intercepts and Records Anthropic Call

**Trigger:** Developer calls `anthropic.messages.create(model="claude-sonnet-4-6", ...)` with base_url pointed at proxy.

**Setup:**
- `ANTHROPIC_API_KEY` set; proxy running on port 8022
- Client configured: `client = anthropic.Anthropic(base_url="http://localhost:8022/anthropic")`
- Session header: `X-Session-ID: test-session-001`

**Expected agent actions:**
1. Proxy receives POST `/anthropic/v1/messages`
2. MetricsCollector records `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`, `task_type`
3. Proxy forwards request to real Anthropic API and returns response unmodified
4. Background judge task enqueued: quality score + hallucination check
5. LLMCall saved to SQLite with all 8 metrics (quality_score/task_success populated within 2s)

**Expected output:**
- Client receives identical response to calling Anthropic directly
- `GET /api/v1/metrics` returns `total_calls: 1`, non-zero `avg_latency_ms`, non-zero `cost_usd`
- SQLite `llm_calls` table has 1 row with valid `latency_ms > 0` and `cost_usd > 0`

**Success criteria:**
- Proxy overhead ≤ 5ms (measured via perf_counter difference)
- Both Anthropic and OpenAI format requests proxied correctly
- quality_score populated within 3 seconds

---

## Scenario 2: Correction Detection — Multi-Turn Session

**Trigger:** User has a 5-turn conversation. Turns 3 and 5 contain correction intent.

**Setup:**
- Turn 1: "Write a Python function to sort a list"
- Turn 2: LLM responds with bubble sort
- Turn 3: "That's wrong, use quicksort not bubble sort" ← correction
- Turn 4: LLM responds with quicksort
- Turn 5: "Not what I asked, add type hints please" ← correction

**Expected agent actions:**
1. MetricsCollector detects "That's wrong" in turn 3 → marks turn 2 call as `is_correction=True`
2. MetricsCollector detects "Not what I asked" in turn 5 → marks turn 4 call as `is_correction=True`
3. Session `correction_count` incremented to 2 after full conversation

**Expected output:**
- `GET /api/v1/sessions` shows session with `correction_count: 2`
- Report shows `correction_rate: 0.40` (2 of 5 turns were corrections)
- "Top Correction Sessions" section lists this session

**Success criteria:**
- Correction detection precision ≥ 80% on these labeled turns
- Session correction_count correctly incremented in SQLite

---

## Scenario 3: HELM Score Computation

**Trigger:** After accumulating 20+ calls for a model, request HELM evaluation.

**Setup:**
- 25 synthetic LLMCall records inserted into SQLite with quality_scores 1–5
- Mix of task types: coding (8), writing (7), reasoning (5), qa (5)
- Varying latencies: 200–8000ms; costs: $0.0001–$0.002

**Expected agent actions:**
1. BenchmarkEvaluator.evaluate_with_helm(calls) invoked
2. accuracy = mean(quality_score)/5.0
3. calibration = 1.0 - variance(quality_score)/4.0
4. efficiency = normalized (1/latency × 1/cost)
5. composite = weighted sum (35% accuracy + 20% calibration + 20% robustness + 15% efficiency + 10% bias)
6. HELMScore saved to benchmark_results table

**Expected output:**
```json
{
  "accuracy": 0.72,
  "calibration": 0.68,
  "robustness": 0.75,
  "efficiency": 0.54,
  "composite": 0.69,
  "sample_count": 25
}
```

**Success criteria:**
- HELM composite ∈ [0.0, 1.0]
- All 4 individual scores are non-negative
- Returns "N/A — insufficient data" for < 10 calls

---

## Scenario 4: ELO Rating — Multi-Model Pairwise Comparison

**Trigger:** Two models (claude-sonnet-4-6 and gpt-4o) have both answered 5 identical prompts (same prompt_hash).

**Setup:**
- 5 matched pairs: same prompt_hash, one call from each model
- claude-sonnet-4-6 quality scores: [4.2, 3.8, 4.5, 4.1, 3.9]
- gpt-4o quality scores: [3.9, 4.1, 4.0, 3.7, 4.3]
- Initial ELO: both 1000

**Expected agent actions:**
1. compute_elo_ratings finds 5 matched pairs
2. Bradley-Terry pairwise comparison for each pair
3. ELO updated: claude-sonnet wins 3/5 → should gain ~+19 ELO points
4. Ratings saved to elo_ratings table

**Expected output:**
- claude-sonnet-4-6 ELO > 1000
- gpt-4o ELO < 1000
- Difference approximately 30–50 ELO points (not dramatic for close results)

**Success criteria:**
- ELO ratings are valid floats
- Higher quality model has higher ELO
- Tie results in minimal ELO change (< 5 points)

---

## Scenario 5: Markdown Report Generation

**Trigger:** `python -m agent.main report --days 30 --model claude-sonnet-4-6`

**Setup:**
- 50 LLM calls in SQLite from the last 30 days for claude-sonnet-4-6
- Mix of task types; 5 correction turns; avg quality 3.8/5.0

**Expected agent actions:**
1. ReportGenerator.generate_rolling_report(model="claude-sonnet-4-6", days=30)
2. 9 sections generated (header, executive summary, usage, metrics, comparison, task breakdown, cost, failures, knowledge note)
3. LLM synthesizes 3-4 sentence executive summary
4. Report written to `data/BENCHMARK-RESULTS.md`
5. Entry appended to `data/benchmark_log.json`

**Expected output (Markdown excerpt):**
```markdown
# AI Benchmark Report

*Generated: 2026-06-09 08:00 UTC | Scope: model: `claude-sonnet-4-6` | Period: last 30 days*

## Executive Summary

Over the last 30 days, claude-sonnet-4-6 handled 50 tasks with an average quality 
score of 3.8/5.0 and a total cost of $0.0142. The model excelled at coding tasks 
(avg quality 4.2) and struggled with math (avg quality 3.1). With a 10% correction 
rate, consider switching to claude-opus-4-8 for complex reasoning tasks.

## 8-Metric Summary

| # | Metric | Mean | Notes |
...
```

**Success criteria:**
- Report is valid Markdown
- All 9 sections present
- `data/BENCHMARK-RESULTS.md` written successfully
- `benchmark_log.json` appended with valid JSON

---

## Scenario 6: Model Comparison — Best Model Per Task

**Trigger:** `GET /api/v1/compare?days=30`

**Setup:**
- 3 models in SQLite: claude-opus-4-8, gpt-4o, gpt-4o-mini
- Claude: best quality (4.3) but highest cost ($0.002/call)
- GPT-4o: medium quality (3.9), medium cost ($0.0005/call)
- GPT-4o-mini: lowest quality (3.3), cheapest ($0.00005/call)
- Task breakdown: coding 40%, writing 30%, qa 30%

**Expected agent actions:**
1. ModelComparator.compare_models(days=30)
2. Efficiency scores computed for each model
3. best_overall = claude-opus-4-8 (highest quality composite)
4. best_value = gpt-4o (best quality/cost ratio for this workload)
5. best_by_task maps each task to best model
6. LLM generates 4-sentence recommendation

**Expected output:**
```json
{
  "best_overall": "claude-opus-4-8",
  "best_value": "gpt-4o",
  "best_by_task": {
    "coding": "claude-opus-4-8",
    "writing": "gpt-4o"
  },
  "recommendation": "For your workload dominated by coding (40%), claude-opus-4-8..."
}
```

**Success criteria:**
- best_overall has highest efficiency composite score
- best_value has best quality/cost ratio
- Recommendation is non-empty prose

---

## Scenario 7: Graceful Degradation — All LLM Providers Down

**Trigger:** ANTHROPIC_API_KEY invalid, OPENAI_API_KEY invalid, Ollama not running.

**Expected agent actions:**
1. Judge calls fail for all providers
2. MetricsCollector falls back to heuristic quality scoring
3. Proxy still forwards requests to real providers (proxy layer unaffected by judge failures)
4. quality_score filled with heuristic (based on output_tokens and latency)
5. Report generated with note: "Quality scores are heuristic estimates (judge unavailable)"

**Expected output:**
- quality_score is a valid float (not null)
- Proxy continues to function and intercept calls
- No crashes or unhandled exceptions

**Success criteria:**
- proxy_overhead still ≤ 5ms
- All calls still recorded to SQLite
- quality_score field populated (heuristic, not null)

---

## Scenario 8: Privacy Mode — No Prompt Data Leaves Machine

**Trigger:** `PRIVACY_MODE=true` set; user makes 10 calls.

**Expected agent actions:**
1. All judge calls routed to Ollama only (regardless of other API keys set)
2. Only prompt_hash (SHA-256 first 16 chars) stored — not full prompt text
3. Ollama performs quality judging locally
4. Report generated from locally-judged scores

**Expected output:**
- `llm_calls.prompt_preview` contains only first 200 chars of prompt
- Zero calls to Anthropic or OpenAI APIs from the judge layer
- Quality scores still populated (via Ollama)

**Success criteria:**
- No outbound HTTP to anthropic.com or api.openai.com from judge worker
- Prompt content stays local (verify via network capture or mock)
- quality_score is valid float from Ollama judging
