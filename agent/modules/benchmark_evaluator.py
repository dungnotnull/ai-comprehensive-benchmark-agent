"""
BenchmarkEvaluator — HELM/MT-Bench/ELO scoring + LLM-judged quality.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUALITY_JUDGE_PROMPT = """\
You are an expert AI output evaluator. Evaluate the AI response below concisely and objectively.

Task description: {task_description}
AI response: {response_text}

Score each criterion 1-5 (1=very poor, 5=excellent):
- correctness: factually accurate and logically sound
- helpfulness: directly addresses the request
- completeness: sufficiently covers the topic
- conciseness: no padding or unnecessary verbosity

Return ONLY valid JSON, no explanation:
{{"correctness":N,"helpfulness":N,"completeness":N,"conciseness":N,"overall":N}}"""

HALLUCINATION_JUDGE_PROMPT = """\
You are a fact-checking AI. Be conservative — only flag clear, obvious hallucinations.

Review this AI response for unsupported factual claims:
{response_text}

Return ONLY valid JSON:
{{"has_hallucination":false,"confidence":0.0,"suspect_claims":[]}}"""

HELM_EVAL_PROMPT = """\
You are evaluating an AI model response using HELM framework criteria.

Scenario type: {scenario_type}
Response: {response}

Score these criteria 0.0-1.0:
- accuracy: correct and factually sound
- calibration: expresses appropriate confidence (not overconfident)
- robustness: answer appears consistent (would be similar for paraphrased question)
- efficiency: achieves goal without excessive tokens

Return ONLY valid JSON:
{{"accuracy":0.0,"calibration":0.0,"robustness":0.0,"efficiency":0.0}}"""


@dataclass
class HELMScore:
    accuracy: float = 0.0
    calibration: float = 0.0
    robustness: float = 0.0
    efficiency: float = 0.0
    composite: float = 0.0
    sample_count: int = 0


@dataclass
class MTBenchScore:
    writing: float = 0.0
    roleplay: float = 0.0
    extraction: float = 0.0
    reasoning: float = 0.0
    math: float = 0.0
    coding: float = 0.0
    knowledge: float = 0.0
    stem: float = 0.0
    overall: float = 0.0
    sample_count: int = 0


@dataclass
class BenchmarkResult:
    model: str = ""
    helm: HELMScore = field(default_factory=HELMScore)
    mt_bench: MTBenchScore = field(default_factory=MTBenchScore)
    elo_rating: float = 1000.0
    task_success_rate: float = 0.0
    correction_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_calls: int = 0


class BenchmarkEvaluator:
    def __init__(self, config: dict, memory, llm, hf):
        self.config = config
        self.memory = memory
        self.llm = llm
        self.hf = hf

    # ── LLM judge calls ────────────────────────────────────────────────────────

    async def judge_quality(self, call) -> Dict[str, float]:
        """Async judge — returns quality scores 1–5. Fallback to heuristic."""
        try:
            response_text = self._get_response_text(call)
            if not response_text:
                return self._heuristic_quality(call)

            prompt = QUALITY_JUDGE_PROMPT.format(
                task_description=f"[{call.task_type}] {call.prompt_preview}",
                response_text=response_text[:2000],
            )
            raw = await self.llm.complete(
                prompt=prompt,
                max_tokens=150,
                temperature=0.0,
                task="quality_judge",
            )
            cleaned = _strip_fences(raw)
            scores = json.loads(cleaned)
            for k in ("correctness", "helpfulness", "completeness", "conciseness", "overall"):
                scores[k] = max(1.0, min(5.0, float(scores.get(k, 3.0))))
            return scores
        except Exception as exc:
            logger.debug(f"judge_quality error: {exc}")
            return self._heuristic_quality(call)

    async def judge_hallucination(self, call) -> Dict[str, Any]:
        """Async judge — returns hallucination assessment. Fallback returns no hallucination."""
        try:
            response_text = self._get_response_text(call)
            if not response_text:
                return {"has_hallucination": False, "confidence": 0.0, "suspect_claims": []}

            prompt = HALLUCINATION_JUDGE_PROMPT.format(
                response_text=response_text[:2000],
            )
            raw = await self.llm.complete(
                prompt=prompt,
                max_tokens=200,
                temperature=0.0,
                task="hallucination_judge",
            )
            cleaned = _strip_fences(raw)
            result = json.loads(cleaned)
            result.setdefault("has_hallucination", False)
            result.setdefault("confidence", 0.0)
            result.setdefault("suspect_claims", [])
            return result
        except Exception as exc:
            logger.debug(f"judge_hallucination error: {exc}")
            return {"has_hallucination": False, "confidence": 0.0, "suspect_claims": []}

    async def evaluate_with_helm(self, calls: List) -> HELMScore:
        """Compute HELM-adapted score from a list of LLMCall records."""
        if len(calls) < 10:
            logger.warning("HELM evaluation needs ≥10 calls; got %d", len(calls))
            return HELMScore(sample_count=len(calls))

        quality_scores = [c.quality_score for c in calls if c.quality_score is not None]
        hallucination_scores = [c.hallucination_score for c in calls if c.hallucination_score is not None]
        latencies = [c.latency_ms for c in calls if c.latency_ms > 0]
        costs = [c.cost_usd for c in calls]

        accuracy = (sum(quality_scores) / (len(quality_scores) * 5.0)) if quality_scores else 0.0

        variance = _variance(quality_scores) if len(quality_scores) > 1 else 0.0
        calibration = max(0.0, 1.0 - (variance / 4.0))

        robustness = 1.0 - _coefficient_of_variation(quality_scores) if quality_scores else 0.5

        bias = (sum(hallucination_scores) / len(hallucination_scores)) if hallucination_scores else 0.0

        if latencies and costs:
            avg_lat = sum(latencies) / len(latencies)
            avg_cost = sum(costs) / len(costs)
            max_lat = max(latencies)
            lat_score = 1.0 - (avg_lat / (max_lat + 1e-9))
            cost_score = 1.0 if avg_cost < 0.001 else max(0.0, 1.0 - avg_cost * 100)
            efficiency = (lat_score + cost_score) / 2.0
        else:
            efficiency = 0.5

        composite = (accuracy * 0.35 + calibration * 0.20 + robustness * 0.20 +
                     efficiency * 0.15 + (1.0 - bias) * 0.10)

        return HELMScore(
            accuracy=round(accuracy, 4),
            calibration=round(calibration, 4),
            robustness=round(robustness, 4),
            efficiency=round(efficiency, 4),
            composite=round(composite, 4),
            sample_count=len(calls),
        )

    async def evaluate_with_mtbench(self, calls: List) -> MTBenchScore:
        """Map calls to MT-Bench categories and compute per-category quality score."""
        from agent.modules.metrics_collector import TaskType

        category_map = {
            TaskType.WRITING.value: "writing",
            TaskType.CREATIVE.value: "roleplay",
            TaskType.EXTRACTION.value: "extraction",
            TaskType.REASONING.value: "reasoning",
            TaskType.MATH.value: "math",
            TaskType.CODING.value: "coding",
            TaskType.QA.value: "knowledge",
            TaskType.SUMMARIZATION.value: "knowledge",
            TaskType.TRANSLATION.value: "knowledge",
            TaskType.UNKNOWN.value: "knowledge",
        }

        category_scores: Dict[str, List[float]] = {
            cat: [] for cat in ("writing", "roleplay", "extraction", "reasoning",
                                "math", "coding", "knowledge", "stem")
        }

        for call in calls:
            if call.quality_score is None:
                continue
            cat = category_map.get(call.task_type, "knowledge")
            normalized = (call.quality_score - 1.0) / 4.0 * 10.0
            category_scores[cat].append(normalized)

        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        scores = {cat: avg(vals) for cat, vals in category_scores.items()}
        all_scores = [v for v in scores.values() if v > 0]
        overall = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

        return MTBenchScore(
            writing=scores["writing"],
            roleplay=scores["roleplay"],
            extraction=scores["extraction"],
            reasoning=scores["reasoning"],
            math=scores["math"],
            coding=scores["coding"],
            knowledge=scores["knowledge"],
            stem=scores["stem"],
            overall=overall,
            sample_count=len(calls),
        )

    async def compute_elo_ratings(self, model_calls: Dict[str, List]) -> Dict[str, float]:
        """
        ELO rating via Bradley-Terry model.
        Compares models that answered the same prompt (matched by prompt_hash).
        """
        ratings: Dict[str, float] = {model: 1000.0 for model in model_calls}

        hash_to_calls: Dict[str, List] = {}
        for model, calls in model_calls.items():
            for call in calls:
                if call.quality_score is not None:
                    hash_to_calls.setdefault(call.prompt_hash, []).append(call)

        matched_pairs = [(calls[i], calls[j])
                         for calls in hash_to_calls.values()
                         for i in range(len(calls))
                         for j in range(i + 1, len(calls))]

        K = 32.0
        for call_a, call_b in matched_pairs:
            if call_a.model not in ratings or call_b.model not in ratings:
                continue
            ra = ratings[call_a.model]
            rb = ratings[call_b.model]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            eb = 1.0 - ea
            sa = 1.0 if (call_a.quality_score or 0) > (call_b.quality_score or 0) else (
                0.5 if (call_a.quality_score or 0) == (call_b.quality_score or 0) else 0.0
            )
            sb = 1.0 - sa
            ratings[call_a.model] = ra + K * (sa - ea)
            ratings[call_b.model] = rb + K * (sb - eb)

        return {model: round(r, 1) for model, r in ratings.items()}

    async def persist_elo_ratings(
        self, ratings: Dict[str, float], memory
    ) -> None:
        for model, rating in ratings.items():
            await memory.update_elo_rating(model, rating)

    async def aggregate_benchmark_report(
        self, model: str, calls: List
    ) -> Dict[str, Any]:
        if not calls:
            return {"error": f"No calls found for model {model}"}

        helm = await self.evaluate_with_helm(calls)
        mt = await self.evaluate_with_mtbench(calls)

        quality_scores = [c.quality_score for c in calls if c.quality_score is not None]
        hallucination_scores = [c.hallucination_score for c in calls if c.hallucination_score is not None]
        correction_calls = [c for c in calls if c.is_correction]
        task_success = [c.task_success for c in calls if c.task_success is not None]

        return {
            "model": model,
            "total_calls": len(calls),
            "helm": {
                "accuracy": helm.accuracy,
                "calibration": helm.calibration,
                "robustness": helm.robustness,
                "efficiency": helm.efficiency,
                "composite": helm.composite,
            },
            "mt_bench": {
                "overall": mt.overall,
                "categories": {
                    "writing": mt.writing,
                    "roleplay": mt.roleplay,
                    "extraction": mt.extraction,
                    "reasoning": mt.reasoning,
                    "math": mt.math,
                    "coding": mt.coding,
                    "knowledge": mt.knowledge,
                },
            },
            "operational": {
                "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None,
                "avg_hallucination_rate": round(sum(hallucination_scores) / len(hallucination_scores), 4) if hallucination_scores else None,
                "correction_rate": round(len(correction_calls) / len(calls), 4),
                "task_success_rate": round(sum(task_success) / len(task_success), 4) if task_success else None,
                "avg_latency_ms": round(sum(c.latency_ms for c in calls) / len(calls), 2),
                "total_cost_usd": round(sum(c.cost_usd for c in calls), 6),
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_response_text(self, call) -> str:
        return getattr(call, "response_preview", "")

    def _heuristic_quality(self, call) -> Dict[str, float]:
        latency_ms = call.latency_ms or 1000.0
        output_tokens = call.output_tokens or 100

        completeness = min(5.0, max(1.0, output_tokens / 100.0))
        latency_score = max(1.0, min(5.0, 5.0 - (latency_ms / 5000.0) * 4.0))
        base = (completeness + latency_score) / 2.0

        return {
            "correctness": round(base, 1),
            "helpfulness": round(base, 1),
            "completeness": round(completeness, 1),
            "conciseness": round(latency_score, 1),
            "overall": round(base, 1),
        }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    import re
    text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _coefficient_of_variation(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    std = math.sqrt(_variance(values))
    return min(1.0, std / (mean + 1e-9))
