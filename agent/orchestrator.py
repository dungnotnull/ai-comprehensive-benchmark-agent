"""
BenchmarkOrchestrator — core agent decision loop for ai-benchmark-agent.
Coordinates proxy interception, metric recording, evaluation, and reporting.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or ROOT / "config" / "agent_config.yaml"
        self.config: Dict[str, Any] = self._load_config()
        self._memory = None
        self._llm = None
        self._hf = None
        self._metrics_col = None
        self._benchmark_eval = None
        self._report_gen = None
        self._model_comp = None
        self._knowledge_updater = None
        self._scheduler = None
        self._judge_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._worker_task: Optional[asyncio.Task] = None

        # Prometheus counters (lazy init)
        self._prom_calls_total = None
        self._prom_cost_total = None
        self._prom_latency_hist = None
        self._prom_corrections_total = None
        self._prom_quality_gauge = None
        self._prom_hallucination_gauge = None

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ── Lazy accessors ────────────────────────────────────────────────────────

    @property
    def memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import BenchmarkMemoryManager
            self._memory = BenchmarkMemoryManager(
                db_path=ROOT / self.config.get("memory", {}).get("db_path", "data/benchmark.db")
            )
        return self._memory

    @property
    def llm(self):
        if self._llm is None:
            from tools.llm_client import UnifiedLLMClient
            self._llm = UnifiedLLMClient(memory=self.memory)
        return self._llm

    @property
    def hf(self):
        if self._hf is None:
            from tools.hf_model_manager import HFModelManager
            self._hf = HFModelManager.get_instance()
        return self._hf

    @property
    def metrics_col(self):
        if self._metrics_col is None:
            from agent.modules.metrics_collector import MetricsCollector
            self._metrics_col = MetricsCollector(
                config=self.config,
                memory=self.memory,
                hf=self.hf,
            )
        return self._metrics_col

    @property
    def benchmark_eval(self):
        if self._benchmark_eval is None:
            from agent.modules.benchmark_evaluator import BenchmarkEvaluator
            self._benchmark_eval = BenchmarkEvaluator(
                config=self.config,
                memory=self.memory,
                llm=self.llm,
                hf=self.hf,
            )
        return self._benchmark_eval

    @property
    def report_gen(self):
        if self._report_gen is None:
            from agent.modules.report_generator import ReportGenerator
            self._report_gen = ReportGenerator(
                config=self.config,
                memory=self.memory,
                llm=self.llm,
                output_dir=ROOT / self.config.get("data", {}).get("output_dir", "data"),
            )
        return self._report_gen

    @property
    def model_comp(self):
        if self._model_comp is None:
            from agent.modules.model_comparator import ModelComparator
            self._model_comp = ModelComparator(
                config=self.config,
                memory=self.memory,
                llm=self.llm,
                hf=self.hf,
            )
        return self._model_comp

    @property
    def knowledge_updater(self):
        if self._knowledge_updater is None:
            from tools.knowledge_updater import KnowledgeUpdater
            self._knowledge_updater = KnowledgeUpdater(
                config=self.config,
                memory=self.memory,
            )
        return self._knowledge_updater

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, scheduler: bool = True):
        self.memory.init_db()
        self._worker_task = asyncio.create_task(self._judge_worker())
        if scheduler and os.environ.get("BENCHMARK_SCHEDULER", "1") == "1":
            self._start_scheduler()
        logger.info("BenchmarkOrchestrator started")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        logger.info("BenchmarkOrchestrator stopped")

    def _start_scheduler(self):
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            self._scheduler = AsyncIOScheduler()
            self._scheduler.add_job(
                self._daily_report_job,
                "cron",
                hour=8,
                minute=0,
                id="daily_report",
            )
            self._scheduler.add_job(
                self._weekly_knowledge_job,
                "cron",
                day_of_week="sun",
                hour=2,
                minute=0,
                id="weekly_knowledge",
            )
            self._scheduler.start()
            logger.info("APScheduler started: daily report @08:00, weekly knowledge @Sun 02:00")
        except ImportError:
            logger.warning("APScheduler not installed; scheduled jobs disabled")

    async def _daily_report_job(self):
        logger.info("Daily report job triggered")
        try:
            await self.generate_report(days=30)
        except Exception as exc:
            logger.error(f"Daily report job failed: {exc}")

    async def _weekly_knowledge_job(self):
        logger.info("Weekly knowledge update triggered")
        try:
            await self.update_knowledge()
        except Exception as exc:
            logger.error(f"Weekly knowledge job failed: {exc}")

    # ── Core orchestration ────────────────────────────────────────────────────

    async def intercept_and_record(
        self,
        request_body: dict,
        response_body: dict,
        provider: str,
        path: str,
        latency_ms: float,
        session_id: str,
        status_code: int = 200,
    ):
        """Called as a background task for every proxied LLM call."""
        try:
            if status_code >= 500:
                return

            call = self.metrics_col.extract_call(
                request_body=request_body,
                response_body=response_body,
                provider=provider,
                latency_ms=latency_ms,
                session_id=session_id,
            )
            if call is None:
                return

            await self.memory.save_call(call)
            await self.memory.upsert_session(session_id, call)

            self._update_prom_counters(call)

            # Enqueue judge tasks (non-blocking)
            try:
                self._judge_queue.put_nowait(call)
            except asyncio.QueueFull:
                logger.warning("Judge queue full; dropping quality evaluation for call %s", call.call_id)

        except Exception as exc:
            logger.error(f"intercept_and_record error: {exc}")

    async def _judge_worker(self):
        """Background worker that processes judge queue calls."""
        while True:
            try:
                call = await asyncio.wait_for(self._judge_queue.get(), timeout=5.0)
                await self._run_judge(call)
                self._judge_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Judge worker error: {exc}")

    async def _run_judge(self, call):
        """Run quality + hallucination judge for a recorded call."""
        try:
            quality, hallucination = await asyncio.gather(
                self.benchmark_eval.judge_quality(call),
                self.benchmark_eval.judge_hallucination(call),
                return_exceptions=True,
            )
            updates: dict = {}
            if isinstance(quality, dict):
                updates["quality_score"] = quality.get("overall", 3.0)
            if isinstance(hallucination, dict):
                updates["hallucination_score"] = 1.0 if hallucination.get("has_hallucination") else 0.0
            if updates:
                await self.memory.update_call_scores(call.call_id, updates)
        except Exception as exc:
            logger.error(f"_run_judge error for call {call.call_id}: {exc}")

    # ── Public interface ──────────────────────────────────────────────────────

    async def get_metrics_summary(
        self, model: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        return await self.memory.get_metrics_summary(model=model, days=days)

    async def evaluate_session(self, session_id: str) -> Dict[str, Any]:
        calls = await self.memory.get_session_calls(session_id)
        if not calls:
            return {"error": "Session not found"}
        return await self.benchmark_eval.aggregate_benchmark_report(
            model=calls[0].model,
            calls=calls,
        )

    async def generate_report(
        self, model: Optional[str] = None, days: int = 30
    ) -> str:
        await self._refresh_elo_ratings(days=days)
        return await self.report_gen.generate_rolling_report(model=model, days=days)

    async def compare_models(
        self, model_ids: Optional[List[str]] = None, days: int = 30
    ) -> Dict[str, Any]:
        await self._refresh_elo_ratings(days=days)
        return await self.model_comp.compare_models(
            model_ids=model_ids, days=days
        )

    async def update_knowledge(self) -> Dict[str, Any]:
        return await self.knowledge_updater.run_update()

    async def _refresh_elo_ratings(self, days: int = 30) -> None:
        models_data = await self.memory.get_per_model_summary(days=days)
        if len(models_data) < 2:
            return
        model_calls: Dict[str, list] = {}
        for model_name in models_data:
            model_calls[model_name] = await self.memory.get_calls(model=model_name, days=days)
        ratings = await self.benchmark_eval.compute_elo_ratings(model_calls)
        await self.benchmark_eval.persist_elo_ratings(ratings, self.memory)

    # ── Prometheus metrics ────────────────────────────────────────────────────

    async def get_prometheus_metrics(self) -> str:
        from tools.prometheus_client import generate
        return generate()

    def _update_prom_counters(self, call):
        try:
            from tools.prometheus_client import record_call
            record_call(
                model=call.model,
                provider=call.provider,
                task_type=call.task_type,
                latency_ms=call.latency_ms,
                cost_usd=call.cost_usd,
                quality_score=call.quality_score,
                hallucination_score=call.hallucination_score,
                is_correction=call.is_correction,
                has_tool_call=call.has_tool_call,
                context_utilization=call.context_utilization,
                task_success=call.task_success,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
            )
        except Exception as exc:
            logger.debug(f"Prometheus record error: {exc}")
