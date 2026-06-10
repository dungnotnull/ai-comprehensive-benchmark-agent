# SECOND-KNOWLEDGE-BRAIN — ai-benchmark-agent

*Self-updating domain knowledge base for LLM evaluation, benchmark methodology, and model comparison.*

---

## Core Concepts & Frameworks

### HELM (Holistic Evaluation of Language Models)
Stanford CRFM framework evaluating models across 42 scenarios on 7 metrics: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency. Key insight: no single model dominates across all scenarios — task-specific evaluation is essential.

### MT-Bench
LMSys two-turn evaluation framework with 80 challenging questions across 8 categories: writing, roleplay, extraction, reasoning, math, coding, knowledge (STEM), knowledge (other). Uses GPT-4 as judge with scoring 1–10.

### Chatbot Arena / ELO Rating
LMSys crowdsourced pairwise comparison platform. Bradley-Terry model converts win/loss records to ELO scores. Key finding: ELO from real-user preferences correlates better with downstream task performance than static benchmarks.

### BIG-Bench
Google/DeepMind collaborative benchmark with 204 tasks beyond GPT-4 capabilities. Focus on tasks requiring reasoning, world knowledge, and novel problem-solving. BIG-Bench Hard (BBH) subset most predictive of model capability.

### LLM-as-Judge
Using a strong LLM (GPT-4, Claude) to evaluate responses of weaker models. Zheng et al. (2023) show 80%+ agreement between GPT-4 judgments and human preferences on MT-Bench. Requires position-debiasing (swap responses) and verbosity-debiasing.

### Operational Metrics (vs. Static Benchmarks)
Static benchmarks fail to capture: (1) correction rate in real usage, (2) latency under production load, (3) cost per task, (4) context window utilization efficiency. Operational metrics from production traffic are uniquely valuable for model selection.

---

## Key Research Papers

| Title | Authors | Year | Venue | Link | Key Finding | Relevance |
|-------|---------|------|-------|------|-------------|-----------|
| HELM: Holistic Evaluation of Language Models | Liang et al. | 2022 | NeurIPS | arxiv:2211.09110 | 7 evaluation criteria; no model dominates all | Core scoring framework |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Zheng et al. | 2023 | NeurIPS | arxiv:2306.05685 | GPT-4 judge achieves >80% human agreement | LLM judge methodology |
| Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference | Chiang et al. | 2024 | ICML | arxiv:2403.04132 | ELO from human votes predicts task performance | ELO rating system |
| Beyond the Imitation Game: BIG-Bench | Srivastava et al. | 2022 | TMLR | arxiv:2206.04615 | 204 tasks; BBH subset most predictive | Task category design |
| GPT-4 Technical Report | OpenAI | 2023 | OpenAI | arxiv:2303.08774 | GPT-4 evaluation methodology and capabilities | Baseline model evaluation |
| Claude 3 Model Card | Anthropic | 2024 | Anthropic | anthropic.com | Constitutional AI evaluation, safety-capability tradeoffs | Baseline model evaluation |
| Can Large Language Models Be an Alternative to Human Evaluations? | Chiang & Lee | 2023 | ACL | arxiv:2305.01937 | LLM evaluators reliable for many NLP tasks | Automated evaluation validity |
| LIMA: Less Is More for Alignment | Zhou et al. | 2023 | NeurIPS | arxiv:2305.11206 | 1,000 carefully curated examples match RLHF | Quality over quantity in evaluation |
| A Survey on Evaluation of Large Language Models | Chang et al. | 2023 | TIST | arxiv:2307.03109 | Taxonomy of 19 evaluation dimensions | Comprehensive framework |
| AlpacaEval: An Automatic Evaluator of Instruction-following Models | Li et al. | 2023 | GitHub | tatsu-lab/alpaca_eval | Win-rate vs. text-davinci-003 as quick proxy | Automated leaderboard design |
| Benchmarking Large Language Models in Complex Instructions Following with Multiple Constraints | He et al. | 2024 | ACL | arxiv:2401.11943 | Multi-constraint instruction following harder than single | Task complexity measurement |
| Measuring Massive Multitask Language Understanding | Hendrycks et al. | 2021 | ICLR | arxiv:2009.03300 | MMLU 57-subject benchmark; foundational capability test | Knowledge evaluation |
| HumanEval: Evaluating Large Language Models Trained on Code | Chen et al. | 2021 | arXiv | arxiv:2107.03374 | Pass@k functional correctness for code generation | Coding task evaluation |
| Prometheus: Inducing Fine-grained Evaluation Capability in Language Models | Kim et al. | 2023 | ICLR | arxiv:2310.08491 | Open-source LLM judge as alternative to GPT-4 judging | Privacy-preserving evaluation |
| FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning | Glazer et al. | 2024 | arXiv | arxiv:2411.04872 | State-of-art models solve <2% of problems | Ceiling effects in evaluation |

---

## State-of-the-Art Models (as of 2026-06)

| Model | Provider | Context | Input $/1M | Output $/1M | MT-Bench | MMLU | Notes |
|-------|---------|---------|-----------|------------|---------|------|-------|
| claude-opus-4-8 | Anthropic | 200K | $15.00 | $75.00 | ~9.4 | ~90.1 | Best for complex reasoning |
| claude-sonnet-4-6 | Anthropic | 200K | $3.00 | $15.00 | ~9.0 | ~88.7 | Best cost/quality ratio |
| gpt-4o | OpenAI | 128K | $2.50 | $10.00 | 9.0 | 87.2 | Strong multimodal |
| gpt-4o-mini | OpenAI | 128K | $0.15 | $0.60 | 8.0 | 82.0 | Best cheap model |
| gemini-1.5-pro | Google | 1M | $3.50 | $10.50 | ~8.8 | 86.5 | Longest context |
| llama-3.3-70b | Meta (Ollama) | 128K | $0 (local) | $0 (local) | ~8.3 | 82.6 | Best open-source |
| mistral-large-2 | Mistral | 128K | $2.00 | $6.00 | ~8.5 | 84.0 | Strong coding |
| deepseek-v3 | DeepSeek | 64K | $0.27 | $1.10 | ~8.6 | 87.1 | Best $/quality ratio |

---

## LLM Prompt Patterns

### Quality Judge (1–5 scale)
```
System: You are an expert AI output evaluator. Evaluate concisely and objectively.

User: Task: {task_description}
Response: {llm_response}

Score 1-5 on each (1=very poor, 5=excellent):
- correctness: factually accurate and logically sound
- helpfulness: directly addresses the request
- completeness: sufficiently covers the topic
- conciseness: no padding or unnecessary verbosity

Return ONLY JSON: {"correctness":N,"helpfulness":N,"completeness":N,"conciseness":N,"overall":N}
```

### Hallucination Detector
```
System: You are a fact-checking AI. Be conservative — only flag clear hallucinations.

User: Review for unsupported factual claims:
Response: {response_text}

Return ONLY JSON: {"has_hallucination":bool,"confidence":0.0-1.0,"suspect_claims":["claim1","claim2"]}
```

### Model Recommendation Synthesis
```
System: You are an AI performance analyst. Synthesize metrics into actionable recommendations.

User: Model performance data (last {days} days):
{metrics_table}

Write 3-4 sentences: (1) Which model performed best overall and why. (2) Which model is best for cost-sensitive tasks. (3) Where each model failed most. (4) Concrete recommendation for this user's workload.
```

### HELM Evaluation Rubric
```
System: You are evaluating an AI model response using the HELM framework criteria.

User: Scenario: {scenario_type}
Response to evaluate: {response}

Score these HELM-adapted criteria (0.0-1.0):
- accuracy: correct and factually sound
- calibration: expresses appropriate confidence (not overconfident)
- robustness: response would be similar to paraphrased version of same question
- efficiency: achieves the goal without excessive tokens

Return JSON: {"accuracy":N,"calibration":N,"robustness":N,"efficiency":N}
```

---

## Authoritative Data Sources

| Source | URL | Type | Update Freq |
|--------|-----|------|------------|
| HELM Leaderboard | crfm.stanford.edu/helm | Benchmark scores | Monthly |
| LMSYS Chatbot Arena | chat.lmsys.org | ELO leaderboard | Real-time |
| Open LLM Leaderboard | huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard | Open-source benchmarks | Weekly |
| Artificial Analysis | artificialanalysis.ai | Speed + quality + cost | Weekly |
| LLM Pricing | llm-price.com | Cost per 1M tokens | Continuously |
| arXiv cs.AI | arxiv.org/list/cs.AI | Research papers | Daily |
| arXiv cs.LG | arxiv.org/list/cs.LG | ML evaluation papers | Daily |
| Papers with Code | paperswithcode.com/sota | SOTA leaderboards | Weekly |
| EleutherAI Eval Harness | github.com/EleutherAI/lm-evaluation-harness | Evaluation framework | Monthly |
| FastChat / MT-Bench | github.com/lm-sys/FastChat | MT-Bench tools | Monthly |

---

## Self-Update Protocol

```yaml
schedule: "weekly — Sunday 02:00 local time"
sources:
  arxiv:
    categories: ["cs.AI", "cs.LG"]
    search_queries:
      - "LLM evaluation benchmark"
      - "language model assessment methodology"
      - "AI benchmark design"
      - "LLM judge evaluation"
      - "model comparison framework"
    max_results_per_query: 20
    recency_window_days: 90

  semantic_scholar:
    queries:
      - "HELM holistic evaluation language models"
      - "MT-Bench multi-turn evaluation"
      - "LMSys chatbot arena ELO"
      - "BIG-Bench language model benchmark"
      - "operational LLM metrics production"
    fields: ["title", "authors", "year", "venue", "abstract", "externalIds"]
    limit: 10

  github_releases:
    repos:
      - "EleutherAI/lm-evaluation-harness"
      - "stanford-crfm/helm"
      - "lm-sys/FastChat"
      - "openai/evals"
    check_interval_days: 7

scoring:
  recency_weight: 0.6      # papers from last 90 days rank higher
  relevance_weight: 0.4    # keyword match score
  keywords: ["benchmark", "evaluation", "LLM", "language model", "assessment", "metric", "judge", "ELO", "HELM", "MT-Bench"]
  top_n: 10                # add top 10 papers per run

deduplication:
  method: SHA-256 of (title + doi/url)
  storage: knowledge_hashes table in SQLite
```

---

## Knowledge Update Log

| Date | Source | Papers Added | Notes |
|------|--------|-------------|-------|
| 2026-06-09 | Manual seed | 15 | Initial knowledge base population |

### 2026-06-09 — Initial Seed (15 Papers)

1. **HELM: Holistic Evaluation of Language Models** — Liang et al., NeurIPS 2022 — 7 evaluation criteria; core scoring framework for this agent
2. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al., NeurIPS 2023 — GPT-4 judge achieves >80% human agreement; validates our judging approach
3. **Chatbot Arena: An Open Platform for Evaluating LLMs** — Chiang et al., ICML 2024 — ELO methodology; Bradley-Terry model implementation reference
4. **Beyond the Imitation Game: BIG-Bench** — Srivastava et al., TMLR 2022 — 204 tasks; BBH subset most predictive of capabilities
5. **GPT-4 Technical Report** — OpenAI, 2023 — Baseline evaluation methodology; capability benchmarks
6. **Claude 3 Model Card** — Anthropic, 2024 — Safety/capability evaluation; Constitutional AI scoring
7. **Can LLMs Be an Alternative to Human Evaluations?** — Chiang & Lee, ACL 2023 — Validates automated evaluation for NLP tasks
8. **LIMA: Less Is More for Alignment** — Zhou et al., NeurIPS 2023 — Quality over quantity; informs quality scoring design
9. **A Survey on Evaluation of LLMs** — Chang et al., TIST 2023 — 19-dimension taxonomy; comprehensive coverage
10. **AlpacaEval** — Li et al., 2023 — Win-rate vs. reference model as quick benchmark proxy
11. **Benchmarking LLMs in Complex Instruction Following** — He et al., ACL 2024 — Multi-constraint evaluation complexity
12. **MMLU: Measuring Massive Multitask Language Understanding** — Hendrycks et al., ICLR 2021 — 57-subject knowledge evaluation foundation
13. **HumanEval: Evaluating LLMs Trained on Code** — Chen et al., 2021 — Pass@k coding metric; informs coding task scoring
14. **Prometheus: Fine-grained Evaluation in LLMs** — Kim et al., ICLR 2023 — Open-source judge alternative to GPT-4
15. **FrontierMath: A Benchmark for Advanced Mathematical Reasoning** — Glazer et al., 2024 — Ceiling effects; importance of task difficulty calibration
