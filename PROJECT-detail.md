# ai-benchmark-agent — Full Technical Specification

## Executive Summary

`ai-benchmark-agent` is a transparent HTTP proxy that sits between a user's application and any LLM provider (Anthropic, OpenAI, Ollama). Every API call passes through the proxy unmodified — but the agent silently records 8 operational metrics per call, computes industry-standard benchmark scores (HELM, MT-Bench, ELO), and builds a personalized model performance database. Daily Markdown reports tell the user exactly which model performs best on their specific task types, and where each model fails.

**Problem:** Standard benchmarks (MMLU, HumanEval, BIG-Bench) measure models on static test sets. They don't capture real-world operational metrics: how often does the user have to correct the model? What does latency look like at p95? Which model gives the best quality/cost ratio for this user's actual workload?

**Solution:** A zero-configuration middleware layer. Change one environment variable (`OPENAI_BASE_URL=http://localhost:8022/openai`) and every API call is automatically instrumented.

---

## Target Users & Use Cases

| User | Trigger | Agent Action |
|------|---------|-------------|
| Developer | Calls `anthropic.messages.create(...)` in production | Proxy intercepts, records latency + cost + quality, saves to DB |
| AI researcher | Runs 500 prompt experiments | Daily report shows HELM scores per model |
| Enterprise team | Evaluating model switch (Claude → GPT-4o) | Comparison report: task-by-task quality/cost breakdown |
| Power user | Notices model behaving strangely | Correction rate trend shows spike — recommendation: switch model |

---

## Agent Architecture

```
Client Application
  │  HTTP POST /v1/messages (Anthropic format)
  │  HTTP POST /v1/chat/completions (OpenAI format)
  ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Proxy Server  (agent/main.py — port 8022)           │
│  • Receives request, extracts model+provider+session headers │
│  • Timestamps request, hashes prompt SHA-256                 │
│  • Forwards to real provider endpoint                        │
│  • Records response, computes 8 metrics                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                    ┌──────▼──────────────────────────────┐
                    │  BenchmarkOrchestrator               │
                    │  (agent/orchestrator.py)             │
                    │                                      │
                    │  ┌─────────────────────────────┐    │
                    │  │  MetricsCollector            │    │
                    │  │  • 8 metrics per call        │    │
                    │  │  • Correction detection NLP  │    │
                    │  │  • Task type classification  │    │
                    │  └───────────┬─────────────────┘    │
                    │              │                       │
                    │  ┌───────────▼─────────────────┐    │
                    │  │  BenchmarkEvaluator          │    │
                    │  │  • HELM 7-criteria scoring   │    │
                    │  │  • MT-Bench 8-category eval  │    │
                    │  │  • ELO rating (LMSys-style)  │    │
                    │  │  • LLM-judged quality 1–5    │    │
                    │  └───────────┬─────────────────┘    │
                    │              │                       │
                    │  ┌───────────▼─────────────────┐    │
                    │  │  ReportGenerator             │    │
                    │  │  • Rolling BENCHMARK.md      │    │
                    │  │  • benchmark_log.json        │    │
                    │  │  • CSV export                │    │
                    │  │  • LLM narrative synthesis   │    │
                    │  └───────────┬─────────────────┘    │
                    │              │                       │
                    │  ┌───────────▼─────────────────┐    │
                    │  │  ModelComparator             │    │
                    │  │  • Task-fit scoring          │    │
                    │  │  • Efficiency frontier       │    │
                    │  │  • LLM recommendation prose  │    │
                    │  └─────────────────────────────┘    │
                    └──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         SQLite DB     LLM API     HuggingFace
       (metrics +    (Claude       (MiniLM +
        sessions)     judge)        BGE-large)
```

---

## Full Module Catalog

### `agent/modules/metrics_collector.py` — MetricsCollector

**Responsibility:** Core instrumentation layer. Extracts all 8 metrics from every proxied LLM call.

| Attribute | Detail |
|-----------|--------|
| Inputs | `request_body: dict`, `response_body: dict`, `latency_ms: float`, `session_id: str` |
| Outputs | `LLMCall` dataclass with all 8 metrics populated |
| Tools called | HF MiniLM (task classification), MemoryManager (previous turns for correction detection) |
| Quality gate | All 8 fields must be non-null before saving; fallback to 0.0 on computation error |

**8 Metrics tracked per call:**

| # | Metric | How Computed |
|---|--------|-------------|
| 1 | `latency_ms` | `time.perf_counter()` around proxy forward |
| 2 | `cost_usd` | `input_tokens × price_in + output_tokens × price_out` from COST_TABLE |
| 3 | `task_success` | LLM judge (0.0–1.0): "Did this response successfully complete the stated task?" |
| 4 | `correction_count` | NLP heuristic on next turn: detected correction keywords → increment session counter |
| 5 | `hallucination_score` | LLM judge (0.0–1.0): "Does this response contain factual claims that appear unsupported?" |
| 6 | `tool_call_success` | `1.0` if tool calls present and all returned non-error; `null` if no tools |
| 7 | `context_utilization` | `(input_tokens + output_tokens) / model_context_window` |
| 8 | `quality_score` | LLM judge (1–5): holistic rating on correctness, helpfulness, completeness, conciseness |

**Correction detection keywords:** `["wrong", "incorrect", "that's not", "not right", "fix this", "redo", "try again", "not what i", "you missed", "please redo", "that doesn't", "revise", "rethink", "change that", "no, actually"]`

---

### `agent/modules/benchmark_evaluator.py` — BenchmarkEvaluator

**Responsibility:** Compute industry-standard evaluation scores from accumulated LLM call records.

| Attribute | Detail |
|-----------|--------|
| Inputs | `List[LLMCall]`, `model: str`, `evaluation_type: str` |
| Outputs | `BenchmarkResult` with score breakdown |
| Tools called | LLM API (Claude as judge), HF BGE-large (task embedding) |
| Quality gate | Minimum 10 calls required for HELM/MT-Bench scores; fallback to N/A otherwise |

**HELM scoring (7 criteria):**
- Accuracy: mean `task_success` across all calls
- Calibration: variance in `task_success` scores (lower variance = better calibrated)
- Robustness: success rate on detected paraphrased/repeated prompts
- Fairness: N/A (personal usage, marked as not applicable)
- Bias: hallucination_score mean (proxy for confident wrong answers)
- Toxicity: N/A for personal use
- Efficiency: 1 / (latency_ms × cost_usd) normalized

**MT-Bench 8 categories (task type auto-detection):**
writing, roleplay, extraction, reasoning, math, coding, knowledge, STEM

**ELO rating:** Pairwise comparison between models on same prompts (matched by prompt_hash) using Bradley-Terry model. Initialized at 1000.

---

### `agent/modules/report_generator.py` — ReportGenerator

**Responsibility:** Generate human-readable reports and exportable data files.

| Attribute | Detail |
|-----------|--------|
| Inputs | `model: str | None`, `days: int`, `output_format: str` |
| Outputs | Markdown string / JSON dict / CSV string |
| Tools called | LLM API (narrative synthesis), MemoryManager (data queries) |
| Quality gate | Report must contain at least 1 data point; otherwise "Insufficient data" message |

**Report sections:**
1. Executive Summary (LLM-synthesized 3-sentence narrative)
2. Usage Overview table (calls, sessions, total cost, date range)
3. 8-Metric Summary table (mean ± std per metric)
4. HELM Score Breakdown
5. MT-Bench Category Performance chart (ASCII bar chart)
6. ELO Rating History (markdown table)
7. Top 5 Failed Tasks (highest correction_count turns)
8. Cost Efficiency Analysis
9. LLM Recommendations (which model for which task type)

**Persistent outputs:**
- `data/BENCHMARK-RESULTS.md` — overwritten daily
- `data/benchmark_log.json` — append-only JSON Lines
- `data/exports/benchmark_{date}.csv` — on-demand export

---

### `agent/modules/model_comparator.py` — ModelComparator

**Responsibility:** Cross-model analysis and personalized recommendations.

| Attribute | Detail |
|-----------|--------|
| Inputs | `model_ids: List[str]`, `days: int`, `task_filter: str | None` |
| Outputs | `ComparisonResult` with winner per task type + prose recommendations |
| Tools called | LLM API (recommendation synthesis), HF BGE-large (task cluster similarity) |
| Quality gate | Minimum 5 calls per model for valid comparison; warn if below threshold |

**Efficiency score formula:**
```
efficiency = (task_success × quality_score) / (latency_ms/1000 × cost_usd × 100)
```
Normalized 0–1 within model set. Higher = better value for money.

**Task types:**
`CODING`, `WRITING`, `REASONING`, `EXTRACTION`, `QA`, `SUMMARIZATION`, `TRANSLATION`, `CREATIVE`, `MATH`, `UNKNOWN`

---

## HuggingFace Model Selection

| Model | Task | Benchmark | Reason |
|-------|------|-----------|--------|
| `sentence-transformers/all-MiniLM-L6-v2` | Task classification, quality proxy | MTEB avg 56.26 | 14K tok/sec CPU, 384-dim; fast enough for real-time proxy latency |
| `BAAI/bge-large-en-v1.5` | Session clustering, model comparison | MTEB avg 63.55 | Best retrieval quality for post-hoc analysis; not on critical path |

---

## LLM API Integration Spec

**Provider chain:** Claude `claude-opus-4-8` → OpenAI `gpt-4o` → Ollama `llama3`

**Quality judge prompt template:**
```
System: You are an expert AI evaluator. Your task is to score an AI response on four criteria.

User: 
[TASK]: {task_description}
[RESPONSE]: {llm_response}

Score each criterion 1-5:
- Correctness: Is the response factually accurate and logically sound?
- Helpfulness: Does it directly address what was asked?
- Completeness: Is the response sufficiently complete?
- Conciseness: Is it appropriately concise without padding?

Return JSON: {"correctness": N, "helpfulness": N, "completeness": N, "conciseness": N, "overall": N}
```

**Token budget:** Max 1,000 tokens per judge call. Judge is called async after response is returned to client (no latency impact).

**Hallucination check prompt:**
```
System: You are a fact-checking AI. Identify unsupported factual claims.

User: Review this AI response for hallucinations or unsupported claims:
[RESPONSE]: {response_text}

Return JSON: {"has_hallucination": bool, "confidence": 0.0-1.0, "suspect_claims": ["..."]}
```

---

## E2E Execution Flow

```
Step 1: Client configures OPENAI_BASE_URL=http://localhost:8022/openai
         or ANTHROPIC_BASE_URL=http://localhost:8022/anthropic

Step 2: Client makes LLM API call
         POST /openai/v1/chat/completions  ← OpenAI SDK format
         POST /anthropic/v1/messages       ← Anthropic SDK format

Step 3: Proxy extracts session_id from header (X-Session-ID) or generates UUID

Step 4: MetricsCollector records:
         - start_time = time.perf_counter()
         - input_tokens (from request body or counted via tiktoken)
         - prompt_hash = SHA256(prompt_text)[:16]
         - task_type = MiniLM classify against 10 task type templates

Step 5: Forward request to real provider (pass-through)
         - Add original Authorization header
         - Strip X-Session-ID header

Step 6: Record response:
         - latency_ms = (time.perf_counter() - start_time) × 1000
         - output_tokens from response usage field
         - cost_usd from COST_TABLE
         - context_utilization = (in + out) / model_context_window

Step 7: Background async tasks (do not block client response):
         a) Judge response quality (LLM judge call, ~500ms)
         b) Check hallucination (LLM judge call, ~500ms)
         c) Save LLMCall to SQLite
         d) Update session turn count + correction flag

Step 8: Return original provider response to client (unmodified)

Step 9: [Next turn] If correction intent detected in new prompt:
         - Mark previous turn as corrected
         - Increment session.correction_count

Step 10: Daily 08:00 APScheduler:
          - Generate rolling BENCHMARK-RESULTS.md
          - Append to benchmark_log.json
          - If any model has >20 calls: run HELM + ELO scoring
```

---

## SECOND-KNOWLEDGE-BRAIN.md Integration

- Weekly crawl: ArXiv `cs.AI` + `cs.LG`, Semantic Scholar (HELM, MT-Bench, BIG-Bench, LMSys)
- New papers inform: updated scoring rubrics, new benchmark categories, cost efficiency formulas
- Agent reads knowledge base before generating recommendations to cite relevant research

---

## Quality Gates

| Gate | Condition | Failure Action |
|------|-----------|---------------|
| QG-1 | Judge API call succeeds | Fallback to MiniLM cosine similarity as quality proxy |
| QG-2 | At least 10 calls before HELM score | Show "N/A — insufficient data (need 10+ calls)" |
| QG-3 | Latency measurement must be >0ms | Drop call with warning log if latency ≤ 0 |
| QG-4 | Cost calculation: model must be in COST_TABLE | Log "unknown model" warning; cost = null |
| QG-5 | Report generation: at least 1 day of data | Return "No data available yet" message |
| QG-6 | Model comparison: at least 2 distinct models | Return single-model report with note |
| QG-7 | ELO rating: matched prompts found | Fall back to aggregate average ranking |

---

## Test Scenarios

See `tests/test-scenarios.md` for 8 full end-to-end scenarios.

---

## Key Design Decisions

1. **Zero-config proxy over monkey-patching:** An HTTP proxy works with every SDK in every language. Monkey-patching Python SDKs misses Go/JS/curl callers. One env var change instruments everything.

2. **Judge calls are async/non-blocking:** The proxy returns the real provider response immediately. Quality judging happens in a background task. This keeps proxy overhead ≤ 5ms for the client.

3. **HELM adapted for personal use:** HELM's fairness/toxicity criteria are irrelevant for single-user operational benchmarking. We replace them with cost-efficiency and correction-rate scores.

4. **ELO requires matched prompts:** Naive averaging is misleading (Claude may only be used for hard tasks). ELO via prompt_hash matching compares models on identical inputs.

5. **Correction detection is heuristic:** Full NLI-based correction detection adds 200ms latency per turn. A 15-keyword heuristic achieves ~85% recall at zero latency cost.

6. **Privacy mode:** When `PRIVACY_MODE=true`, all judge calls route to local Ollama. Prompt text is never sent to external APIs for judging. The prompt_hash (SHA-256 truncated) is stored, not the prompt text.

7. **Single SQLite file:** Simplifies deployment (no Postgres dependency), supports concurrent reads with WAL mode, handles 10M+ rows with indexed queries.
