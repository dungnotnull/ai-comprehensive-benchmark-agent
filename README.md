<div align="center">

# AI Comprehensive Benchmark Agent

### Transparent middleware that intercepts every LLM call, records 8 core metrics, and tells you which model actually works best for **your** work.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-70%20passing-brightgreen)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](docker/)

[Installation](#-installation) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Docker](#-docker-deployment) · [Contributing](#-contributing)

</div>

---

## Why This Exists

Standard benchmarks (MMLU, HumanEval, BIG-Bench) test models in isolation. They don't tell you:

- How often **you** have to correct the model in real conversations
- What **your** actual p95 latency looks like under your workload
- Which model gives the best **quality/cost ratio** for your specific task mix
- Whether the model is **hallucinating** on your domain

This agent solves that by acting as a **drop-in proxy** between your application and any LLM provider. Change one environment variable, and every API call is automatically instrumented.

---

## Features

| Feature | Description |
|---------|-------------|
| **Zero-config proxy** | Change `base_url` and you're done. Works with Anthropic, OpenAI, and Ollama SDKs |
| **8 metrics per call** | Latency, cost, quality score, hallucination rate, correction rate, task success, tool success, context utilization |
| **HELM scoring** | Stanford HELM framework adapted for personal operational benchmarking |
| **MT-Bench categories** | 8 task categories auto-detected: coding, writing, reasoning, math, QA, extraction, translation, creative |
| **ELO ratings** | Bradley-Terry pairwise comparison on matched prompts - same methodology as LMSys Chatbot Arena |
| **LLM-as-judge** | Async quality scoring (1-5) and hallucination detection via Claude/GPT-4o/Ollama |
| **Daily reports** | 9-section Markdown reports with ASCII charts, cost analysis, and LLM-synthesized recommendations |
| **Privacy mode** | Route all judging to local Ollama. No data leaves your machine |
| **Prometheus metrics** | Full exposition of all 8 metrics with per-model/task-type labels |
| **Knowledge pipeline** | Weekly crawl of ArXiv + Semantic Scholar for latest LLM evaluation research |
| **HuggingFace integration** | MiniLM-L6-v2 for task classification, BGE-large for session clustering, with TF-IDF fallback |

---

## Architecture

```
  Your Application
        |
        | HTTP request (change base_url to localhost:8022)
        v
+-------------------------------------------------------+
|  FastAPI Proxy Server (port 8022)                      |
|                                                        |
|  +------------------+  +----------------------------+  |
|  | MetricsCollector |  | BenchmarkEvaluator         |  |
|  | (8 metrics/call) |->| (HELM / MT-Bench / ELO)    |  |
|  +------------------+  +----------------------------+  |
|                                                        |
|  +------------------+  +----------------------------+  |
|  | ReportGenerator  |  | ModelComparator            |  |
|  | (MD + JSON + CSV)|  | (task-fit + LLM recs)      |  |
|  +------------------+  +----------------------------+  |
+-------------------------------------------------------+
        |              |              |
        v              v              v
   SQLite DB      LLM Judge API   HuggingFace
   (WAL mode)     (async,         (MiniLM-L6-v2
                   non-blocking)    + BGE-large)
```

**Key design principle:** Judge calls are **never** on the critical path. The proxy returns the real provider response immediately. Quality evaluation happens in background async tasks.

---

## 8 Core Metrics

| # | Metric | How It's Measured | Range |
|---|--------|-------------------|-------|
| 1 | **Latency** | `time.perf_counter()` around proxy forward | ms |
| 2 | **Cost** | Token count x provider pricing table | USD |
| 3 | **Task Success** | LLM judge: "Did this complete the task?" | 0.0 - 1.0 |
| 4 | **Correction Rate** | 15-keyword NLP heuristic on next turn | count |
| 5 | **Hallucination Score** | LLM judge: "Any unsupported claims?" | 0.0 - 1.0 |
| 6 | **Tool Call Success** | Parse tool_use blocks for errors | 0.0 - 1.0 |
| 7 | **Context Utilization** | (input + output tokens) / context window | 0.0 - 1.0 |
| 8 | **Quality Score** | LLM judge on 4 criteria (correctness, helpfulness, completeness, conciseness) | 1.0 - 5.0 |

---

## Installation

### From Source

```bash
git clone https://github.com/dungnotnull/ai-comprehensive-benchmark-agent.git
cd ai-comprehensive-benchmark-agent

# Core install (proxy + metrics + reporting)
pip install -e .

# With LLM SDKs (for quality judging)
pip install -e ".[llm]"

# With HuggingFace models (for task classification)
pip install -e ".[ml]"

# Everything + dev tools
pip install -e ".[all]"
```

### Using pip

```bash
pip install ai-benchmark-agent
```

---

## Quick Start

### 1. Configure Environment

```bash
cp config/.env.example .env
```

Edit `.env` with your API keys:

```ini
ANTHROPIC_API_KEY=sk-ant-...      # Optional - for Anthropic proxy + judge
OPENAI_API_KEY=sk-...             # Optional - for OpenAI proxy + judge
OLLAMA_BASE_URL=http://localhost:11434  # For local/offline mode
PRIVACY_MODE=false                # Set true to judge locally only
```

### 2. Start the Proxy

```bash
benchmark-agent start-proxy --proxy-port 8022
```

### 3. Point Your SDK at the Proxy

**Anthropic Python SDK:**

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8022/anthropic",
    default_headers={"X-Session-ID": "my-project-session-001"},
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain quantum computing"}],
)
```

**OpenAI Python SDK:**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8022/openai/v1",
    api_key="any-value",
    default_headers={"X-Session-ID": "my-project-session-001"},
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a sorting function"}],
)
```

**Ollama (curl):**

```bash
curl http://localhost:8022/ollama/api/chat \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello"}]}'
```

### 4. View Results

```bash
# Generate Markdown report
benchmark-agent report --days 30

# Compare models
benchmark-agent compare --days 30

# Export as CSV
benchmark-agent export --output benchmark.csv

# Cost breakdown
benchmark-agent cost-report --days 30

# Update knowledge base (ArXiv + Semantic Scholar)
benchmark-agent update-knowledge
```

---

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with call/session counts |
| `GET` | `/api/v1/metrics` | Aggregated metrics (filter by `?model=&days=`) |
| `GET` | `/api/v1/report` | Full Markdown benchmark report |
| `GET` | `/api/v1/compare` | Cross-model comparison with recommendations |
| `GET` | `/api/v1/sessions` | Recent sessions (filter by `?model=&limit=`) |
| `GET` | `/api/v1/cost` | Cost breakdown by model |
| `GET` | `/api/v1/export/csv` | Download all call data as CSV |
| `POST` | `/api/v1/knowledge/update` | Trigger knowledge crawl |
| `GET` | `/metrics` | Prometheus-format metrics |

### Proxy Routes

| Route | Forwards To |
|-------|-------------|
| `/anthropic/{path}` | `https://api.anthropic.com/{path}` |
| `/openai/{path}` | `https://api.openai.com/{path}` |
| `/ollama/{path}` | `http://localhost:11434/{path}` |

---

## Docker Deployment

### Single Command

```bash
docker compose up -d
```

This starts the agent on port 8022 with an Ollama sidecar.

### With GPU Support

```bash
docker compose --profile gpu up -d
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `PRIVACY_MODE` | `false` | Route all judging to local Ollama |
| `PROXY_PORT` | `8022` | Proxy server port |
| `BENCHMARK_SCHEDULER` | `1` | Enable daily report + weekly knowledge |

---

## Benchmark Methodology

### HELM (Stanford)

Adapted from the Holistic Evaluation of Language Models framework. We compute 4 operational criteria:

- **Accuracy** - Mean quality score normalized to 0-1
- **Calibration** - Low variance in quality scores across calls
- **Robustness** - Consistency of scores on repeated/similar prompts
- **Efficiency** - Latency and cost normalized performance

### MT-Bench (LMSys)

Calls are auto-classified into 8 categories using MiniLM-L6-v2 semantic similarity. Quality scores are normalized to the 1-10 MT-Bench scale per category.

### ELO Rating (Bradley-Terry)

Models that answered the same prompt (matched by SHA-256 hash) are compared pairwise. Winner gains ELO points using K=32, starting from 1000. Same methodology as LMSys Chatbot Arena.

---

## Privacy Mode

Set `PRIVACY_MODE=true` to:

- Route **all** judge calls to local Ollama
- Never send prompt content to external APIs for evaluation
- Store only SHA-256 hash of prompts (first 16 chars), never full text
- Keep `response_preview` limited to 500 chars for local judging only

---

## Project Structure

```
ai-benchmark-agent/
├── agent/
│   ├── __init__.py
│   ├── __main__.py               # python -m agent entry point
│   ├── main.py                   # FastAPI proxy + Click CLI
│   ├── orchestrator.py           # Core orchestration loop
│   ├── modules/
│   │   ├── metrics_collector.py  # 8-metric extraction per call
│   │   ├── benchmark_evaluator.py # HELM / MT-Bench / ELO scoring
│   │   ├── report_generator.py   # Markdown + JSON + CSV reports
│   │   └── model_comparator.py   # Cross-model comparison
│   └── memory/
│       └── memory_manager.py     # SQLite 5-table schema (WAL mode)
├── tools/
│   ├── llm_client.py             # Unified Claude/OpenAI/Ollama client
│   ├── hf_model_manager.py       # HuggingFace model lazy-loader
│   ├── knowledge_updater.py      # ArXiv + Semantic Scholar crawler
│   └── prometheus_client.py      # Prometheus metrics exposition
├── config/
│   ├── agent_config.yaml         # Production configuration
│   └── .env.example              # Environment variable template
├── docker/
│   ├── Dockerfile                # python:3.12-slim, non-root
│   └── docker-compose.yml        # 3 service profiles
├── tests/
│   ├── test_agent.py             # 70 automated tests
│   └── test-scenarios.md         # 8 end-to-end test scenarios
├── SECOND-KNOWLEDGE-BRAIN.md     # Self-updating research knowledge base
├── pyproject.toml                # Build config + optional deps
├── requirements.txt              # Pinned dependencies
├── LICENSE                       # MIT
└── .gitignore
```

---

## Supported Models & Pricing

| Model | Context | Input $/1M | Output $/1M |
|-------|---------|-----------|------------|
| claude-opus-4-8 | 200K | $15.00 | $75.00 |
| claude-sonnet-4-6 | 200K | $3.00 | $15.00 |
| claude-haiku-4-5-20251001 | 200K | $0.80 | $4.00 |
| gpt-4o | 128K | $2.50 | $10.00 |
| gpt-4o-mini | 128K | $0.15 | $0.60 |
| o1 | 200K | $15.00 | $60.00 |
| gemini-1.5-pro | 1M | $3.50 | $10.50 |
| gemini-1.5-flash | 1M | $0.075 | $0.30 |
| llama3 (Ollama) | 128K | $0 (local) | $0 (local) |
| mistral-large-2 | 128K | $2.00 | $6.00 |
| deepseek-v3 | 64K | $0.27 | $1.10 |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=agent --cov=tools --cov-report=term-missing
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If this project helps you choose the right model, give it a star!**

[Report Bug](https://github.com/dungnotnull/ai-comprehensive-benchmark-agent/issues) · [Request Feature](https://github.com/dungnotnull/ai-comprehensive-benchmark-agent/issues) · [Ask Question](https://github.com/dungnotnull/ai-comprehensive-benchmark-agent/issues)

</div>
