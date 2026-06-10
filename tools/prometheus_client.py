"""
Prometheus metrics exposition for ai-benchmark-agent.
Gauges, Counters, and Histograms for all 8 core metrics.
"""

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_metrics: Dict[str, float] = {
    "calls_total": 0.0,
    "cost_total_usd": 0.0,
    "latency_sum_ms": 0.0,
    "latency_count": 0.0,
    "quality_sum": 0.0,
    "quality_count": 0.0,
    "hallucination_sum": 0.0,
    "hallucination_count": 0.0,
    "corrections_total": 0.0,
    "tool_calls_total": 0.0,
    "context_utilization_sum": 0.0,
    "context_utilization_count": 0.0,
    "task_success_sum": 0.0,
    "task_success_count": 0.0,
}
_labels: Dict[str, Dict[str, float]] = {}


def record_call(
    model: str,
    provider: str,
    task_type: str,
    latency_ms: float,
    cost_usd: float,
    quality_score: Optional[float] = None,
    hallucination_score: Optional[float] = None,
    is_correction: bool = False,
    has_tool_call: bool = False,
    context_utilization: float = 0.0,
    task_success: Optional[float] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    with _lock:
        _metrics["calls_total"] += 1
        _metrics["cost_total_usd"] += cost_usd
        _metrics["latency_sum_ms"] += latency_ms
        _metrics["latency_count"] += 1
        _metrics["context_utilization_sum"] += context_utilization
        _metrics["context_utilization_count"] += 1

        if quality_score is not None:
            _metrics["quality_sum"] += quality_score
            _metrics["quality_count"] += 1

        if hallucination_score is not None:
            _metrics["hallucination_sum"] += hallucination_score
            _metrics["hallucination_count"] += 1

        if is_correction:
            _metrics["corrections_total"] += 1

        if has_tool_call:
            _metrics["tool_calls_total"] += 1

        if task_success is not None:
            _metrics["task_success_sum"] += task_success
            _metrics["task_success_count"] += 1

        key = f"{provider}:{model}:{task_type}"
        label_data = _labels.setdefault(key, {
            "calls": 0.0, "cost": 0.0, "latency_sum": 0.0,
            "tokens_in": 0.0, "tokens_out": 0.0,
        })
        label_data["calls"] += 1
        label_data["cost"] += cost_usd
        label_data["latency_sum"] += latency_ms
        label_data["tokens_in"] += input_tokens
        label_data["tokens_out"] += output_tokens


def generate() -> str:
    with _lock:
        m = dict(_metrics)
        labels = {k: dict(v) for k, v in _labels.items()}

    total = m["calls_total"] or 1
    avg_latency = m["latency_sum_ms"] / m["latency_count"] if m["latency_count"] else 0
    avg_quality = m["quality_sum"] / m["quality_count"] if m["quality_count"] else 0
    avg_hallucination = m["hallucination_sum"] / m["hallucination_count"] if m["hallucination_count"] else 0
    avg_context = m["context_utilization_sum"] / m["context_utilization_count"] if m["context_utilization_count"] else 0
    avg_task_success = m["task_success_sum"] / m["task_success_count"] if m["task_success_count"] else 0
    correction_rate = m["corrections_total"] / total if total else 0

    lines = [
        "# HELP benchmark_calls_total Total proxied LLM calls",
        "# TYPE benchmark_calls_total counter",
        f"benchmark_calls_total {int(m['calls_total'])}",
        "",
        "# HELP benchmark_cost_total_usd Total cost in USD across all calls",
        "# TYPE benchmark_cost_total_usd counter",
        f"benchmark_cost_total_usd {m['cost_total_usd']:.6f}",
        "",
        "# HELP benchmark_latency_avg_ms Average latency in ms",
        "# TYPE benchmark_latency_avg_ms gauge",
        f"benchmark_latency_avg_ms {avg_latency:.2f}",
        "",
        "# HELP benchmark_quality_avg Average quality score (1-5)",
        "# TYPE benchmark_quality_avg gauge",
        f"benchmark_quality_avg {avg_quality:.3f}",
        "",
        "# HELP benchmark_hallucination_avg Average hallucination score (0-1)",
        "# TYPE benchmark_hallucination_avg gauge",
        f"benchmark_hallucination_avg {avg_hallucination:.3f}",
        "",
        "# HELP benchmark_correction_rate Rate of correction turns",
        "# TYPE benchmark_correction_rate gauge",
        f"benchmark_correction_rate {correction_rate:.4f}",
        "",
        "# HELP benchmark_context_utilization_avg Average context utilization (0-1)",
        "# TYPE benchmark_context_utilization_avg gauge",
        f"benchmark_context_utilization_avg {avg_context:.4f}",
        "",
        "# HELP benchmark_task_success_avg Average task success rate (0-1)",
        "# TYPE benchmark_task_success_avg gauge",
        f"benchmark_task_success_avg {avg_task_success:.4f}",
        "",
        "# HELP benchmark_tool_calls_total Total calls with tool use",
        "# TYPE benchmark_tool_calls_total counter",
        f"benchmark_tool_calls_total {int(m['tool_calls_total'])}",
        "",
        "# HELP benchmark_corrections_total Total correction turns detected",
        "# TYPE benchmark_corrections_total counter",
        f"benchmark_corrections_total {int(m['corrections_total'])}",
        "",
        "# HELP benchmark_call_latency_ms Histogram of call latencies",
        "# TYPE benchmark_call_latency_ms summary",
        f"benchmark_call_latency_ms_sum {m['latency_sum_ms']:.2f}",
        f'benchmark_call_latency_ms_count {int(m["latency_count"])}',
        "",
    ]

    for key, data in sorted(labels.items()):
        parts = key.split(":", 2)
        if len(parts) == 3:
            provider, model, task_type = parts
            labels_str = f'provider="{provider}",model="{model}",task_type="{task_type}"'
            lines.extend([
                f'benchmark_calls_by_label{{{labels_str}}} {int(data["calls"])}',
                f'benchmark_cost_by_label{{{labels_str}}} {data["cost"]:.6f}',
                f'benchmark_tokens_input_by_label{{{labels_str}}} {int(data["tokens_in"])}',
                f'benchmark_tokens_output_by_label{{{labels_str}}} {int(data["tokens_out"])}',
            ])

    return "\n".join(lines) + "\n"
