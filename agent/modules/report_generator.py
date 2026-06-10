"""
ReportGenerator — rolling Markdown report, JSON Lines log, CSV export.
LLM synthesizes raw metrics into narrative recommendations.
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """\
You are an AI performance analyst. Based on the metrics below, write a concise 3-4 sentence executive summary.

Model: {model}
Period: last {days} days
Total calls: {total_calls}
Avg quality score (1-5): {avg_quality}
Avg latency ms: {avg_latency}
Total cost USD: {total_cost}
Correction rate: {correction_rate}
Hallucination rate: {hallucination_rate}
Best task type: {best_task}
Worst task type: {worst_task}

Write exactly 4 sentences:
1. Overall assessment of the model's performance.
2. Where the model excels (best task type, quality highlights).
3. Where the model struggled (correction rate, hallucination, cost).
4. Concrete recommendation: should the user keep using this model? For which tasks?
"""

COMPARISON_SYNTHESIS_PROMPT = """\
You are an AI performance analyst. Write a 3-sentence recommendation based on this model comparison.

{comparison_table}

Write:
1. Which model performed best overall and why.
2. Which model offers the best cost/quality ratio.
3. Specific recommendation: use model X for [task], model Y for [task].
"""


class ReportGenerator:
    def __init__(self, config: dict, memory, llm, output_dir: Path):
        self.config = config
        self.memory = memory
        self.llm = llm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)

    async def generate_rolling_report(
        self, model: Optional[str] = None, days: int = 30
    ) -> str:
        summary = await self.memory.get_metrics_summary(model=model, days=days)
        if summary.get("total_calls", 0) == 0:
            return f"# Benchmark Report\n\n_No data available yet. Start using the proxy to collect metrics._\n"

        models_data = await self.memory.get_per_model_summary(days=days)
        if model:
            models_data = {model: models_data.get(model, {})}

        task_breakdown = await self.memory.get_task_type_breakdown(model=model, days=days)
        elo_ratings = await self.memory.get_elo_ratings()

        sections = [
            self._section_header(model, days),
            await self._section_executive_summary(summary, model, days, task_breakdown),
            self._section_usage_overview(summary, days),
            self._section_metrics_table(summary),
            self._section_model_comparison(models_data, elo_ratings),
            self._section_task_breakdown(task_breakdown),
            self._section_cost_analysis(models_data),
            self._section_top_failures(await self.memory.get_top_correction_sessions(model=model, limit=5)),
            self._section_knowledge_note(),
        ]

        report = "\n\n".join(s for s in sections if s)
        report_path = self.output_dir / "BENCHMARK-RESULTS.md"
        report_path.write_text(report, encoding="utf-8")

        await self._append_json_log(summary, model, days)
        return report

    def _section_header(self, model: Optional[str], days: int) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        scope = f"model: `{model}`" if model else "all models"
        return f"# AI Benchmark Report\n\n*Generated: {now} | Scope: {scope} | Period: last {days} days*"

    async def _section_executive_summary(
        self, summary: dict, model: Optional[str], days: int, task_breakdown: dict
    ) -> str:
        def _v(key, default=0):
            val = summary.get(key, default)
            return val if val is not None else default

        try:
            task_scores = task_breakdown.get("by_task_type", {})
            best_task = max(task_scores, key=lambda t: task_scores[t].get("avg_quality") or 0) if task_scores else "N/A"
            worst_task = min(task_scores, key=lambda t: task_scores[t].get("avg_quality") or 5) if task_scores else "N/A"

            prompt = SYNTHESIS_PROMPT.format(
                model=model or "all",
                days=days,
                total_calls=_v("total_calls", 0),
                avg_quality=f"{_v('avg_quality_score'):.2f}",
                avg_latency=f"{_v('avg_latency_ms'):.0f}",
                total_cost=f"${_v('total_cost_usd'):.4f}",
                correction_rate=f"{_v('correction_rate'):.1%}",
                hallucination_rate=f"{_v('avg_hallucination_score'):.1%}",
                best_task=best_task,
                worst_task=worst_task,
            )
            synthesis = await self.llm.complete(
                prompt=prompt, max_tokens=200, temperature=0.3, task="report_synthesis"
            )
        except Exception as exc:
            logger.debug(f"LLM synthesis failed: {exc}")
            synthesis = (
                f"Analysis of {_v('total_calls')} calls over the last {days} days. "
                f"Average quality score: {_v('avg_quality_score'):.2f}/5.0. "
                f"Total cost: ${_v('total_cost_usd'):.4f}. "
                f"Correction rate: {_v('correction_rate'):.1%}."
            )

        return f"## Executive Summary\n\n{synthesis}"

    def _section_usage_overview(self, summary: dict, days: int) -> str:
        lines = [
            "## Usage Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Period | Last {days} days |",
            f"| Total Calls | {summary.get('total_calls', 0):,} |",
            f"| Total Sessions | {summary.get('total_sessions', 0):,} |",
            f"| Total Cost | ${summary.get('total_cost_usd', 0):.4f} |",
            f"| Avg Calls / Day | {summary.get('avg_calls_per_day', 0):.1f} |",
        ]
        return "\n".join(lines)

    def _section_metrics_table(self, summary: dict) -> str:
        def v(key, default=0):
            val = summary.get(key, default)
            return val if val is not None else default

        def fmt_opt(key, fmt=".3f"):
            val = summary.get(key)
            return f"{val:{fmt}}" if val is not None else "N/A"

        lines = [
            "## 8-Metric Summary",
            "",
            "| # | Metric | Mean | Notes |",
            "|---|--------|------|-------|",
            f"| 1 | Latency (ms) | {v('avg_latency_ms'):.0f} | p95: {v('p95_latency_ms'):.0f}ms |",
            f"| 2 | Cost / Call (USD) | ${v('avg_cost_per_call'):.6f} | Total: ${v('total_cost_usd'):.4f} |",
            f"| 3 | Task Success | {fmt_opt('avg_task_success', '.3f')} | 0.0-1.0 |",
            f"| 4 | Correction Count | {v('avg_correction_count'):.2f} | Per session |",
            f"| 5 | Hallucination Rate | {fmt_opt('avg_hallucination_score', '.3f')} | 0.0=none, 1.0=high |",
            f"| 6 | Tool Call Success | {fmt_opt('avg_tool_success', '.3f')} | null if no tools used |",
            f"| 7 | Context Utilization | {v('avg_context_utilization'):.3f} | 0.0-1.0 |",
            f"| 8 | Quality Score | {fmt_opt('avg_quality_score', '.3f')} | 1.0-5.0 scale |",
        ]
        return "\n".join(lines)

    def _section_model_comparison(
        self, models_data: Dict[str, dict], elo_ratings: Dict[str, float]
    ) -> str:
        if len(models_data) < 2:
            if models_data:
                model_name = next(iter(models_data))
                return f"## Model Comparison\n\n*Only one model (`{model_name}`) has data. Use multiple models to enable comparison.*"
            return ""

        lines = [
            "## Model Comparison",
            "",
            "| Model | ELO | Quality | Latency (ms) | Cost/Call | Correction Rate |",
            "|-------|-----|---------|-------------|-----------|----------------|",
        ]
        for model_name, data in sorted(models_data.items(), key=lambda x: elo_ratings.get(x[0], 1000), reverse=True):
            elo = elo_ratings.get(model_name, 1000)
            quality = data.get("avg_quality_score")
            quality_str = f"{quality:.2f}" if quality is not None else "N/A"
            lines.append(
                f"| {model_name} | {elo:.0f} | {quality_str} | "
                f"{data.get('avg_latency_ms', 0):.0f} | "
                f"${data.get('avg_cost_per_call', 0):.6f} | "
                f"{data.get('correction_rate', 0):.1%} |"
            )
        return "\n".join(lines)

    def _section_task_breakdown(self, task_breakdown: dict) -> str:
        by_type = task_breakdown.get("by_task_type", {})
        if not by_type:
            return ""

        max_count = max((v.get("count", 0) for v in by_type.values()), default=1)
        bar_width = 20

        lines = ["## Task Type Performance", "", "```"]
        for task_type, data in sorted(by_type.items(), key=lambda x: x[1].get("count", 0), reverse=True):
            count = data.get("count", 0)
            quality = data.get("avg_quality")
            quality_str = f"{quality:.1f}" if quality is not None else "---"
            bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            lines.append(f"{task_type:<15} [{bar}] {count:>4} calls  quality: {quality_str}/5")
        lines.append("```")
        return "\n".join(lines)

    def _section_cost_analysis(self, models_data: Dict[str, dict]) -> str:
        if not models_data:
            return ""

        lines = [
            "## Cost Analysis",
            "",
            "| Model | Total Cost | Calls | Cost/Call | Cost/Quality |",
            "|-------|-----------|-------|-----------|-------------|",
        ]
        for model_name, data in models_data.items():
            cost = data.get("total_cost_usd", 0)
            calls = data.get("total_calls", 0)
            cost_per_call = data.get("avg_cost_per_call", 0)
            quality = data.get("avg_quality_score") or 1.0
            cost_per_quality = cost_per_call / quality if quality > 0 else float("inf")
            lines.append(
                f"| {model_name} | ${cost:.4f} | {calls} | "
                f"${cost_per_call:.6f} | ${cost_per_quality:.6f} |"
            )
        return "\n".join(lines)

    def _section_top_failures(self, sessions: List[dict]) -> str:
        if not sessions:
            return ""
        lines = [
            "## Top Correction Sessions",
            "",
            "| Session | Model | Corrections | Turns | Task Type |",
            "|---------|-------|------------|-------|-----------|",
        ]
        for s in sessions:
            lines.append(
                f"| `{s.get('session_id', '')[:8]}…` | {s.get('model', '')} | "
                f"{s.get('correction_count', 0)} | {s.get('turn_count', 0)} | "
                f"{s.get('dominant_task_type', 'unknown')} |"
            )
        return "\n".join(lines)

    def _section_knowledge_note(self) -> str:
        return (
            "---\n\n*This report is generated by ai-benchmark-agent. "
            "Evaluation methodology: HELM (Stanford), MT-Bench (LMSys), Bradley-Terry ELO. "
            "Knowledge base updated weekly via ArXiv cs.AI + Semantic Scholar.*"
        )

    async def _append_json_log(self, summary: dict, model: Optional[str], days: int):
        log_path = self.output_dir / "benchmark_log.json"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model or "all",
            "days": days,
            "summary": summary,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def export_csv(self, model: Optional[str] = None, days: int = 30) -> str:
        calls = await self.memory.get_calls(model=model, days=days)
        if not calls:
            return "call_id,session_id,model,provider,timestamp,latency_ms,cost_usd,quality_score,task_type,is_correction\n"

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "call_id", "session_id", "model", "provider", "timestamp",
            "latency_ms", "cost_usd", "task_success", "correction_count",
            "hallucination_score", "tool_call_success", "context_utilization",
            "quality_score", "input_tokens", "output_tokens", "task_type",
            "is_correction", "has_tool_call",
        ])
        for call in calls:
            writer.writerow([
                call.call_id, call.session_id, call.model, call.provider,
                call.timestamp, call.latency_ms, call.cost_usd,
                call.task_success, call.correction_count,
                call.hallucination_score, call.tool_call_success,
                call.context_utilization, call.quality_score,
                call.input_tokens, call.output_tokens, call.task_type,
                call.is_correction, call.has_tool_call,
            ])
        return output.getvalue()

    async def generate_comparison_report(self, models_data: dict, elo_ratings: dict) -> str:
        if not models_data:
            return "No comparison data available."

        table_lines = ["| Model | Quality | Latency | Cost/Call | ELO |"]
        table_lines.append("|-------|---------|---------|-----------|-----|")
        for m, data in models_data.items():
            q = data.get("avg_quality_score")
            table_lines.append(
                f"| {m} | {f'{q:.2f}' if q else 'N/A'} | "
                f"{data.get('avg_latency_ms', 0):.0f}ms | "
                f"${data.get('avg_cost_per_call', 0):.5f} | "
                f"{elo_ratings.get(m, 1000):.0f} |"
            )
        comparison_table = "\n".join(table_lines)

        try:
            synthesis = await self.llm.complete(
                prompt=COMPARISON_SYNTHESIS_PROMPT.format(comparison_table=comparison_table),
                max_tokens=200,
                temperature=0.3,
                task="comparison_synthesis",
            )
        except Exception:
            synthesis = "Use the table above to compare model performance metrics."

        return f"## Model Comparison\n\n{comparison_table}\n\n### Recommendation\n\n{synthesis}"
