# ai-benchmark-agent — AI Model Performance Evaluation Agent

**Tagline:** Transparent middleware that intercepts every LLM call, records 8 core metrics, and tells you which model actually works best for your work.

**Build Phase:** Phase 1 — Core Agent Modules

**Cluster B:** Agent Platform, Tooling & Meta-Agents

---

## Problem Statement

AI practitioners use multiple models (Claude, GPT-4o, Gemini, Llama) for different tasks but have no systematic way to measure real-world performance. Standard benchmarks (MMLU, HumanEval) test models in isolation — they don't reflect how a model performs on YOUR specific tasks, in YOUR workflow, with YOUR correction patterns. This agent solves that by acting as a transparent proxy layer that intercepts every LLM call, records 8 core operational metrics per call, and continuously builds a personalized model performance profile that improves with every interaction.

---

## Agent Architecture

```
User / Application
        ↓  HTTP request to localhost:8022 (drop-in replacement for provider API)
┌─────────────────────────────────────────────────────┐
│  BenchmarkOrchestrator (agent/orchestrator.py)      │
│  ┌──────────────────┐  ┌────────────────────────┐   │
│  │ MetricsCollector │→ │  BenchmarkEvaluator    │   │
│  │ (intercept+8     │  │  (HELM/MT-Bench/ELO    │   │
│  │  metrics/call)   │  │   scoring)             │   │
│  └──────────────────┘  └────────────────────────┘   │
│  ┌──────────────────┐  ┌────────────────────────┐   │
│  │ ReportGenerator  │  │  ModelComparator       │   │
│  │ (MD+JSON+CSV)    │  │  (task-fit + LLM recs) │   │
│  └──────────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────┘
        ↓              ↓              ↓
   SQLite DB      LLM API        HuggingFace
  (metrics)    (Claude judge)  (MiniLM quality)
        ↓
  SECOND-KNOWLEDGE-BRAIN.md (grows weekly)
```

**Step-by-step flow:**
1. Client sends LLM API request to proxy port 8022
2. MetricsCollector records request start time, prompt hash, input token count
3. Proxy forwards request to real provider (Anthropic/OpenAI/Ollama)
4. Response returns; MetricsCollector records latency, output tokens, cost
5. Correction detection: NLP heuristic scans next turn for correction intent
6. BenchmarkEvaluator scores quality (LLM judge) and classifies task type
7. SQLite persists the LLMCall record
8. APScheduler triggers daily report generation at 08:00 local time
9. ReportGenerator writes rolling `BENCHMARK-RESULTS.md` + `benchmark_log.json`
10. ModelComparator synthesizes recommendations via Claude API

---

## Module List (`agent/modules/`)

| File | Responsibility |
|------|---------------|
| `metrics_collector.py` | Intercepts LLM calls, extracts 8 metrics per call, detects correction intent |
| `benchmark_evaluator.py` | HELM/MT-Bench/LMSys ELO scoring; LLM-judged quality 1–5 |
| `report_generator.py` | Rolling Markdown report + JSON log + CSV export; LLM synthesis |
| `model_comparator.py` | Cross-model comparison; task-fit recommendations; efficiency scoring |

---

## Tools (`agent/tools/`)

| File | Responsibility |
|------|---------------|
| `prometheus_client.py` | Prometheus metrics exposition (Gauge/Counter/Histogram) |

---

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity for quality scoring & task type classification | Fast, 384-dim, 14K token/sec on CPU; MTEB score 56.26; ideal for real-time proxy |
| `BAAI/bge-large-en-v1.5` | High-quality embedding for task clustering & session similarity | MTEB #1 (Aug 2023) score 63.55; used for model comparison clustering |

---

## LLM API Integration

**Provider priority:** Claude (`claude-opus-4-8`) → OpenAI (`gpt-4o`) → Ollama (`llama3`)

**Use cases within this agent:**
- **Quality judging:** Score LLM output 1–5 on criteria: correctness, helpfulness, completeness, conciseness
- **HELM evaluation:** Generate scenario-specific evaluation rubrics
- **Recommendation synthesis:** Translate raw metric tables into natural language model recommendations
- **Hallucination check:** Cross-reference factual claims against knowledge base

**Privacy mode:** If `PRIVACY_MODE=true`, all judging is done via Ollama locally. No task content leaves the machine.

---

## Knowledge Crawl Sources

| Source | Config |
|--------|--------|
| ArXiv `cs.AI` | "LLM evaluation", "benchmark design", "language model assessment" |
| ArXiv `cs.LG` | "benchmark methodology", "model comparison", "evaluation harness" |
| Semantic Scholar | "HELM Stanford", "MT-Bench", "BIG-Bench", "LMSys chatbot arena" |
| GitHub Releases | `EleutherAI/lm-evaluation-harness`, `stanford-crfm/helm`, `lm-sys/FastChat` |
| **Update schedule** | **Weekly, Sunday 02:00 local time** |

---

## Supporting Tools (`tools/`)

| File | Purpose |
|------|---------|
| `knowledge_updater.py` | Crawls ArXiv + Semantic Scholar + GitHub; appends to SECOND-KNOWLEDGE-BRAIN.md |
| `llm_client.py` | Unified Claude/OpenAI/Ollama client with streaming, retry, cost tracking |
| `hf_model_manager.py` | Lazy-loading registry for MiniLM-L6-v2 and BGE-large-en-v1.5 |

---

## Active Development Tasks

- [x] CLAUDE.md
- [x] PROJECT-detail.md
- [x] PROJECT-DEVELOPMENT-PHASE-TRACKING.md
- [x] SECOND-KNOWLEDGE-BRAIN.md
- [x] agent/main.py
- [x] agent/orchestrator.py
- [x] agent/modules/metrics_collector.py
- [x] agent/modules/benchmark_evaluator.py
- [x] agent/modules/report_generator.py
- [x] agent/modules/model_comparator.py
- [x] agent/memory/memory_manager.py
- [x] tools/knowledge_updater.py
- [x] tools/llm_client.py
- [x] tools/hf_model_manager.py
- [x] config/agent_config.yaml
- [x] config/.env.example
- [x] docker/docker-compose.yml
- [x] tests/test-scenarios.md
- [x] tests/test_agent.py
