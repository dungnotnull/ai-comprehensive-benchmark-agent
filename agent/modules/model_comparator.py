"""
ModelComparator — cross-model comparison, task-fit scoring, LLM recommendations.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RECOMMENDATION_PROMPT = """\
You are an AI model selection advisor. Analyze this performance data and give concrete recommendations.

User's usage data over the last {days} days:
{models_table}

Task type breakdown:
{task_table}

Most common task: {top_task}
Total API cost so far: ${total_cost:.4f}

Give recommendations in 4 sentences:
1. Best model overall for this user's workload and why.
2. Best cost-efficient choice (if different from #1).
3. Which model to avoid and why.
4. Specific advice: "Use [model A] for [task types], [model B] for [task types]."
"""


@dataclass
class ModelScore:
    model: str
    total_calls: int = 0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_per_call: float = 0.0
    correction_rate: float = 0.0
    hallucination_rate: float = 0.0
    task_success_rate: float = 0.0
    context_utilization: float = 0.0
    efficiency_score: float = 0.0
    elo_rating: float = 1000.0
    task_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    models: List[ModelScore] = field(default_factory=list)
    best_overall: str = ""
    best_by_task: Dict[str, str] = field(default_factory=dict)
    best_value: str = ""
    recommendation: str = ""
    elo_rankings: Dict[str, float] = field(default_factory=dict)


class ModelComparator:
    def __init__(self, config: dict, memory, llm, hf):
        self.config = config
        self.memory = memory
        self.llm = llm
        self.hf = hf

    async def compare_models(
        self, model_ids: Optional[List[str]] = None, days: int = 30
    ) -> Dict[str, Any]:
        models_data = await self.memory.get_per_model_summary(days=days)
        if model_ids:
            models_data = {m: v for m, v in models_data.items() if m in model_ids}

        if not models_data:
            return {"error": "No model data found for the specified period."}

        elo_ratings = await self.memory.get_elo_ratings()
        task_data = await self.memory.get_task_type_breakdown(days=days)

        model_scores: List[ModelScore] = []
        for model_name, data in models_data.items():
            score = ModelScore(
                model=model_name,
                total_calls=data.get("total_calls", 0),
                avg_quality=data.get("avg_quality_score") or 0.0,
                avg_latency_ms=data.get("avg_latency_ms") or 0.0,
                avg_cost_per_call=data.get("avg_cost_per_call") or 0.0,
                correction_rate=data.get("correction_rate") or 0.0,
                hallucination_rate=data.get("avg_hallucination_score") or 0.0,
                task_success_rate=data.get("avg_task_success") or 0.0,
                context_utilization=data.get("avg_context_utilization") or 0.0,
                elo_rating=elo_ratings.get(model_name, 1000.0),
            )
            score.efficiency_score = self._compute_efficiency(score)
            score.task_scores = await self.memory.get_model_task_scores(model_name, days=days)
            model_scores.append(score)

        best_overall = self._find_best_overall(model_scores)
        best_value = self._find_best_value(model_scores)
        best_by_task = self._find_best_by_task(model_scores)

        recommendation = await self._generate_recommendation(
            model_scores=model_scores,
            task_data=task_data,
            days=days,
        )

        result = ComparisonResult(
            models=model_scores,
            best_overall=best_overall,
            best_by_task=best_by_task,
            best_value=best_value,
            recommendation=recommendation,
            elo_rankings=elo_ratings,
        )

        return _to_dict(result)

    def _compute_efficiency(self, score: ModelScore) -> float:
        """Higher is better: quality × success / (latency × cost)."""
        quality = score.avg_quality / 5.0
        success = score.task_success_rate if score.task_success_rate > 0 else 0.5
        latency_factor = score.avg_latency_ms / 10_000.0
        cost_factor = score.avg_cost_per_call * 1000.0

        denominator = (latency_factor + 0.1) * (cost_factor + 0.01)
        raw = (quality * success) / denominator
        return round(min(10.0, raw), 4)

    def _find_best_overall(self, scores: List[ModelScore]) -> str:
        if not scores:
            return ""

        def composite(s: ModelScore) -> float:
            return (
                s.avg_quality / 5.0 * 0.35 +
                (1.0 - s.correction_rate) * 0.25 +
                (1.0 - s.hallucination_rate) * 0.20 +
                s.efficiency_score / 10.0 * 0.20
            )

        best = max(scores, key=composite)
        return best.model

    def _find_best_value(self, scores: List[ModelScore]) -> str:
        if not scores:
            return ""

        def value_score(s: ModelScore) -> float:
            cost = s.avg_cost_per_call or 0.0001
            quality = s.avg_quality or 1.0
            return quality / (cost * 1000 + 0.001)

        scores_with_cost = [s for s in scores if s.avg_cost_per_call >= 0]
        if not scores_with_cost:
            return ""
        best = max(scores_with_cost, key=value_score)
        return best.model

    def _find_best_by_task(self, scores: List[ModelScore]) -> Dict[str, str]:
        task_types = [
            "coding", "writing", "reasoning", "extraction",
            "qa", "summarization", "translation", "creative", "math",
        ]
        result: Dict[str, str] = {}
        for task in task_types:
            best_model = ""
            best_score = -1.0
            for s in scores:
                ts = s.task_scores.get(task, 0.0)
                if ts > best_score:
                    best_score = ts
                    best_model = s.model
            if best_model and best_score > 0:
                result[task] = best_model
        return result

    async def _generate_recommendation(
        self,
        model_scores: List[ModelScore],
        task_data: dict,
        days: int,
    ) -> str:
        try:
            models_table_lines = [
                "| Model | Quality | Latency | Cost/Call | Correction | ELO |",
                "|-------|---------|---------|-----------|-----------|-----|",
            ]
            for s in sorted(model_scores, key=lambda x: x.elo_rating, reverse=True):
                models_table_lines.append(
                    f"| {s.model} | {s.avg_quality:.2f}/5 | "
                    f"{s.avg_latency_ms:.0f}ms | ${s.avg_cost_per_call:.5f} | "
                    f"{s.correction_rate:.1%} | {s.elo_rating:.0f} |"
                )
            models_table = "\n".join(models_table_lines)

            by_task = task_data.get("by_task_type", {})
            task_lines = ["| Task | Calls | Avg Quality |", "|------|-------|------------|"]
            for task, tdata in sorted(by_task.items(), key=lambda x: x[1].get("count", 0), reverse=True)[:5]:
                q = tdata.get("avg_quality")
                task_lines.append(
                    f"| {task} | {tdata.get('count', 0)} | "
                    f"{f'{q:.2f}' if q else 'N/A'} |"
                )
            task_table = "\n".join(task_lines)

            top_task = max(by_task, key=lambda t: by_task[t].get("count", 0)) if by_task else "unknown"
            total_cost = sum(s.avg_cost_per_call * s.total_calls for s in model_scores)

            prompt = RECOMMENDATION_PROMPT.format(
                days=days,
                models_table=models_table,
                task_table=task_table,
                top_task=top_task,
                total_cost=total_cost,
            )
            return await self.llm.complete(
                prompt=prompt, max_tokens=250, temperature=0.3, task="model_recommendation"
            )
        except Exception as exc:
            logger.debug(f"Recommendation generation error: {exc}")
            if model_scores:
                best = max(model_scores, key=lambda s: s.avg_quality)
                cheapest = min(model_scores, key=lambda s: s.avg_cost_per_call)
                return (
                    f"Based on {days} days of usage data, `{best.model}` achieved the highest "
                    f"quality score ({best.avg_quality:.2f}/5.0). "
                    f"`{cheapest.model}` offers the lowest cost per call "
                    f"(${cheapest.avg_cost_per_call:.5f}). "
                    f"Consider your task mix when choosing between quality and cost."
                )
            return "Insufficient data to generate recommendations."

    async def find_best_model_for_task(self, task_type: str) -> Dict[str, Any]:
        task_scores = {}
        models_data = await self.memory.get_per_model_summary(days=90)
        for model_name in models_data:
            scores = await self.memory.get_model_task_scores(model_name, days=90)
            task_score = scores.get(task_type, 0.0)
            if task_score > 0:
                task_scores[model_name] = task_score

        if not task_scores:
            return {"error": f"No data for task type '{task_type}'", "task_type": task_type}

        best_model = max(task_scores, key=lambda m: task_scores[m])
        return {
            "task_type": task_type,
            "best_model": best_model,
            "score": task_scores[best_model],
            "all_models": task_scores,
        }


# ── Serialization ─────────────────────────────────────────────────────────────

def _to_dict(result: ComparisonResult) -> Dict[str, Any]:
    return {
        "models": [
            {
                "model": s.model,
                "total_calls": s.total_calls,
                "avg_quality": s.avg_quality,
                "avg_latency_ms": s.avg_latency_ms,
                "avg_cost_per_call": s.avg_cost_per_call,
                "correction_rate": s.correction_rate,
                "hallucination_rate": s.hallucination_rate,
                "efficiency_score": s.efficiency_score,
                "elo_rating": s.elo_rating,
                "task_scores": s.task_scores,
            }
            for s in result.models
        ],
        "best_overall": result.best_overall,
        "best_by_task": result.best_by_task,
        "best_value": result.best_value,
        "elo_rankings": result.elo_rankings,
        "recommendation": result.recommendation,
    }
