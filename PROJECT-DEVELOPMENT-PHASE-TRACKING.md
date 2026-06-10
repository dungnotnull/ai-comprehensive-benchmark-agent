# ai-benchmark-agent — Development Phase Tracking

## Quantified Improvement Targets

| # | Target | Baseline | Goal | Measurement |
|---|--------|----------|------|-------------|
| T1 | Proxy overhead latency | N/A (no proxy) | ≤ 5ms p95 added latency | `time.perf_counter()` before/after proxy logic |
| T2 | Quality judge accuracy | N/A | ≥ 80% agreement with human ratings on 100-call sample | Cohen's κ ≥ 0.6 |
| T3 | Correction detection recall | N/A | ≥ 85% recall on labeled correction turns | Manual label 50 correction examples |

---

## Phase 0 — Research & Architecture (Week 1–2)

**Goal:** Understand HELM, MT-Bench, LMSys ELO methodology. Design proxy architecture. Define 8 metrics.

### Tasks
- [x] Read HELM paper (Liang et al. 2022) — extract 7 scoring criteria
- [x] Read MT-Bench paper (Zheng et al. 2023) — extract 8 task categories + judge prompt templates
- [x] Read LMSys Chatbot Arena paper — understand Bradley-Terry ELO methodology
- [x] Read BIG-Bench paper — identify applicable tasks for personal benchmarking
- [x] Design 8-metric schema (LLMCall dataclass fields)
- [x] Prototype HTTP proxy with `httpx` async forwarding
- [x] Design SQLite schema (5 tables)
- [x] Baseline test: proxy overhead on localhost (target: ≤ 5ms)

**Deliverables:** Architecture diagram (above), LLMCall schema, SQLite schema DDL

**Success criteria:** Proxy forwards requests with ≤ 5ms overhead on loopback

**Estimated effort:** 4 person-days

**Status: DONE**

---

## Phase 1 — Core Agent Modules (Week 3–5)

**Goal:** Implement MetricsCollector and BenchmarkEvaluator.

### Tasks
- [x] `agent/modules/metrics_collector.py`
  - [x] LLMCall dataclass (all 8 fields + metadata + response_preview)
  - [x] Latency measurement via `time.perf_counter()`
  - [x] Token counting: use `tiktoken` for OpenAI; response.usage for Anthropic
  - [x] Cost calculation: COST_TABLE for 15 models
  - [x] Task type classification: MiniLM + 10 template embeddings
  - [x] Correction detection: 15 keyword heuristic
  - [x] Context utilization: (in+out)/context_window
  - [x] Response text extraction for LLM judge (Anthropic/OpenAI/Ollama)
- [x] `agent/modules/benchmark_evaluator.py`
  - [x] HELM 7-criteria scoring (adapted for personal use)
  - [x] MT-Bench task categorization (8 categories)
  - [x] ELO rating: Bradley-Terry pairwise comparison on prompt_hash matches
  - [x] LLM judge integration: quality_score + hallucination_score async
  - [x] Response text retrieval for judging via response_preview field
- [x] `agent/memory/memory_manager.py` — SQLite 5-table schema with WAL mode
  - [x] response_preview column + migration for existing DBs

**Deliverables:** metrics_collector.py, benchmark_evaluator.py, memory_manager.py

**Success criteria:** 100 synthetic calls processed; all 8 metrics populated; latency ≤ 5ms proxy overhead

**Estimated effort:** 7 person-days

**Status: DONE**

---

## Phase 2 — Orchestrator + Proxy Server (Week 6–8)

**Goal:** Build the FastAPI proxy server and orchestration loop.

### Tasks
- [x] `agent/main.py` — FastAPI proxy
  - [x] Route `/anthropic/{path}` (Anthropic SDK format)
  - [x] Route `/openai/{path}` (OpenAI SDK format)
  - [x] Route `/ollama/{path}` (Ollama format)
  - [x] CLI commands: start-proxy, report, compare, export, update-knowledge, cost
  - [x] REST API endpoints: /metrics, /report, /compare, /sessions, /health
  - [x] X-Session-ID header extraction
  - [x] .env loading via python-dotenv
- [x] `agent/orchestrator.py`
  - [x] BenchmarkOrchestrator class
  - [x] `async intercept_and_record(request, provider, response)`
  - [x] `async evaluate_session(session_id)`
  - [x] `async generate_report(model, days)`
  - [x] `async compare_models(model_ids, days)`
  - [x] APScheduler: daily 08:00 report generation
  - [x] Prometheus metrics (Gauge/Counter/Histogram for all 8 metrics)
  - [x] Background judge queue worker
- [x] `tools/prometheus_client.py` — Full Prometheus exposition with per-label metrics
- [x] Background task queue: asyncio Queue for judge calls (no blocking)

**Deliverables:** main.py, orchestrator.py, prometheus_client.py

**Success criteria:** Proxy accepts both Anthropic and OpenAI format requests; background judge tasks complete within 2s

**Estimated effort:** 6 person-days

**Status: DONE**

---

## Phase 3 — Report Generator + Model Comparator (Week 9–10)

**Goal:** Implement ReportGenerator and ModelComparator.

### Tasks
- [x] `agent/modules/report_generator.py`
  - [x] 9-section Markdown report template
  - [x] ASCII bar chart for MT-Bench categories
  - [x] JSON Lines log appender
  - [x] CSV exporter
  - [x] LLM narrative synthesis (3-sentence executive summary)
  - [x] Write to `data/BENCHMARK-RESULTS.md` atomically
- [x] `agent/modules/model_comparator.py`
  - [x] Efficiency score formula
  - [x] Task-type winner determination
  - [x] ELO-based overall ranking
  - [x] LLM recommendation prose generation

**Deliverables:** report_generator.py, model_comparator.py

**Success criteria:** Report generated from 100-call synthetic dataset; LLM recommendation matches expected best model

**Estimated effort:** 5 person-days

**Status: DONE**

---

## Phase 4 — HuggingFace Model Integration (Week 11–12)

**Goal:** Integrate MiniLM for task classification and BGE-large for comparison clustering.

### Tasks
- [x] `tools/hf_model_manager.py`
  - [x] MiniLM-L6-v2 lazy load + 10 task template pre-encoding
  - [x] BGE-large lazy load + idle unload (600s)
  - [x] `classify_task(text)` via cosine similarity
  - [x] `encode_batch(texts)` → numpy array
  - [x] CUDA auto-detect
  - [x] TF-IDF fallback when model unavailable
  - [x] Singleton pattern for shared model state

**Deliverables:** hf_model_manager.py

**Success criteria:** Task classification works with model or TF-IDF fallback

**Estimated effort:** 3 person-days

**Status: DONE**

---

## Phase 5 — LLM API Integration & Prompt Engineering (Week 13–14)

**Goal:** Integrate Claude/OpenAI/Ollama judge calls with optimized prompts.

### Tasks
- [x] `tools/llm_client.py` — Claude/OpenAI/Ollama unified client
  - [x] Streaming support (Ollama SSE)
  - [x] Exponential backoff retry (1s, 2s, 4s)
  - [x] Cost tracking via Prometheus client
  - [x] Provider chain: Claude → OpenAI → Ollama
- [x] Quality judge prompt: 4-criteria JSON response (correctness, helpfulness, completeness, conciseness)
- [x] Hallucination check prompt: conservative fact-checking with suspect_claims list
- [x] Recommendation synthesis prompt: 4-sentence actionable prose
- [x] PRIVACY_MODE: force Ollama for all judge calls
- [x] Graceful degradation: heuristic fallback when all providers fail

**Deliverables:** llm_client.py + tuned prompts

**Success criteria:** Judge accuracy κ ≥ 0.6; Ollama fallback works offline

**Estimated effort:** 4 person-days

**Status: DONE**

---

## Phase 6 — SECOND-KNOWLEDGE-BRAIN Pipeline (Week 15–16)

**Goal:** Implement knowledge updater and run first crawl.

### Tasks
- [x] `tools/knowledge_updater.py`
  - [x] ArXiv cs.AI + cs.LG XML API crawl
  - [x] Semantic Scholar graph API (5 queries)
  - [x] GitHub Releases: lm-evaluation-harness, helm, FastChat, openai/evals
  - [x] Recency × relevance scoring (0.6 recency + 0.4 relevance)
  - [x] SHA-256 dedup via memory_manager
  - [x] Append to SECOND-KNOWLEDGE-BRAIN.md
  - [x] APScheduler weekly Sunday 02:00
- [x] Initial knowledge brain seeded with 15 core papers

**Deliverables:** knowledge_updater.py, initial knowledge brain population

**Success criteria:** Crawl pipeline functional; deduplication works

**Estimated effort:** 3 person-days

**Status: DONE**

---

## Phase 7 — Docker + Deployment + Testing (Week 17–18)

**Goal:** Containerize, configure, and validate for production.

### Tasks
- [x] `docker/docker-compose.yml` (3 services: benchmark-agent, benchmark-agent-gpu, ollama)
- [x] `docker/Dockerfile` (python:3.12-slim, non-root, EXPOSE 8022, HEALTHCHECK)
- [x] `tests/test_agent.py` — 42 automated tests
- [x] `tests/test-scenarios.md` — 8 end-to-end scenarios
- [x] `config/agent_config.yaml` — production-ready config
- [x] `config/.env.example` — all required env vars with documentation
- [x] `requirements.txt` — pinned dependencies
- [x] `pyproject.toml` — build config, optional deps, CLI entry point
- [x] `.gitignore` — data, models, env, IDE, cache exclusions
- [x] Prometheus metrics endpoint at `/metrics`
- [x] CSV export endpoint at `/api/v1/export/csv`
- [x] Cost report endpoint at `/api/v1/cost`

**Deliverables:** All config + deployment + test files

**Success criteria:** Docker Compose starts cleanly; all endpoints functional

**Estimated effort:** 5 person-days

**Status: DONE**

---

## Summary

| Phase | Status | Key Deliverable |
|-------|--------|----------------|
| 0 — Research | DONE | Architecture + schema design |
| 1 — Core Modules | DONE | MetricsCollector + BenchmarkEvaluator + response_preview |
| 2 — Orchestrator | DONE | FastAPI proxy + orchestration + Prometheus |
| 3 — Reports | DONE | ReportGenerator + ModelComparator |
| 4 — HuggingFace | DONE | hf_model_manager.py with TF-IDF fallback |
| 5 — LLM API | DONE | llm_client.py + judge prompts + cost logging |
| 6 — Knowledge | DONE | knowledge_updater.py + seeded knowledge brain |
| 7 — Docker/Tests | DONE | Deployment + test suite + pyproject.toml |
| **Total** | **100% COMPLETE** | **Production-grade benchmark agent** |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| LLM judge adds noticeable latency | Medium | High | All judge calls are async/background; never on critical path |
| Prompt hash collisions for ELO | Low | Medium | Use SHA-256 first 32 chars; collision probability negligible |
| Provider API format changes | Medium | High | Version-pin in `httpx` forwarding; add integration test per provider |
| SQLite contention at high throughput | Low | Medium | WAL mode + connection pool (10 connections) |
| MiniLM task classification errors | High | Low | Fallback to keyword heuristic; classification error degrades only task-type breakdown |
