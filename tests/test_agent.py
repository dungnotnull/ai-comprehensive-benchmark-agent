"""
Automated tests for ai-benchmark-agent.
42 tests: MetricsCollector, BenchmarkEvaluator, ReportGenerator, ModelComparator,
MemoryManager, LLMClient, HFModelManager, Integration, CLI smoke.
"""

import asyncio
import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    from agent.memory.memory_manager import BenchmarkMemoryManager
    db = BenchmarkMemoryManager(tmp_path / "test.db")
    db.init_db()
    return db


@pytest.fixture
def sample_call():
    from agent.modules.metrics_collector import LLMCall
    return LLMCall(
        call_id=str(uuid.uuid4()),
        session_id="test-session-001",
        provider="anthropic",
        model="claude-sonnet-4-6",
        timestamp="2026-06-09T10:00:00+00:00",
        prompt_hash="abc123def456",
        prompt_preview="Write a Python function to sort a list",
        latency_ms=342.5,
        cost_usd=0.000125,
        task_success=None,
        correction_count=0,
        hallucination_score=None,
        tool_call_success=None,
        context_utilization=0.012,
        quality_score=None,
        input_tokens=45,
        output_tokens=120,
        task_type="coding",
        is_correction=False,
        has_tool_call=False,
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value='{"correctness":4,"helpfulness":4,"completeness":4,"conciseness":3,"overall":4}'
    )
    return llm


@pytest.fixture
def mock_hf():
    import numpy as np
    hf = MagicMock()
    hf.encode = MagicMock(return_value=np.random.rand(384).astype("float32"))
    hf.encode_batch = MagicMock(return_value=np.random.rand(5, 384).astype("float32"))
    return hf


# ── MetricsCollector (8 tests) ────────────────────────────────────────────────

class TestMetricsCollector:
    def _make_collector(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        return MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)

    def test_extract_call_anthropic(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        request = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello, write a Python function"}],
        }
        response = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 15, "output_tokens": 80},
        }
        call = col.extract_call(request, response, "anthropic", 312.0, "sess-001")
        assert call is not None
        assert call.model == "claude-sonnet-4-6"
        assert call.provider == "anthropic"
        assert call.latency_ms == 312.0
        assert call.input_tokens == 15
        assert call.output_tokens == 80

    def test_extract_call_openai(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        request = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Summarize this document"}],
        }
        response = {
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 20, "completion_tokens": 100},
        }
        call = col.extract_call(request, response, "openai", 550.0, "sess-002")
        assert call is not None
        assert call.model == "gpt-4o-mini"
        assert call.input_tokens == 20
        assert call.output_tokens == 100

    def test_cost_computation_anthropic(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        cost = col._compute_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_cost_computation_zero_for_ollama(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        cost = col._compute_cost("llama3", 1000, 500)
        assert cost == 0.0

    def test_context_utilization(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        util = col._compute_context_utilization("claude-sonnet-4-6", 1000, 500)
        assert 0.0 <= util <= 1.0
        assert abs(util - 1500 / 200_000) < 1e-6

    def test_correction_detection_positive(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        assert col.detect_correction_intent("That's wrong, please fix it") is True
        assert col.detect_correction_intent("Not right, redo this") is True
        assert col.detect_correction_intent("Incorrect answer, try again") is True

    def test_correction_detection_negative(self, tmp_db, mock_hf):
        col = self._make_collector(tmp_db, mock_hf)
        assert col.detect_correction_intent("Great, thank you!") is False
        assert col.detect_correction_intent("Can you add more details?") is False

    def test_task_classification_coding(self, tmp_db, mock_hf):
        import numpy as np
        mock_hf.encode.return_value = np.zeros(384, dtype="float32")
        template_embs = np.zeros((9, 384), dtype="float32")
        template_embs[0, 0] = 1.0  # coding gets highest score
        mock_hf.encode_batch.return_value = template_embs
        col = self._make_collector(tmp_db, mock_hf)
        result = col._classify_task("Write a Python function to implement quicksort algorithm")
        assert result in ["coding", "unknown"]

    def test_latency_percentiles(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
        result = MetricsCollector.compute_latency_percentiles(latencies)
        assert result["p50"] == pytest.approx(550.0, rel=0.1)
        assert result["p95"] >= result["p50"]
        assert result["p99"] >= result["p95"]


# ── BenchmarkEvaluator (7 tests) ─────────────────────────────────────────────

class TestBenchmarkEvaluator:
    def _make_evaluator(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        return BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)

    @pytest.mark.asyncio
    async def test_judge_quality_returns_scores(self, tmp_db, mock_llm, mock_hf, sample_call):
        eval = self._make_evaluator(tmp_db, mock_llm, mock_hf)
        result = await eval.judge_quality(sample_call)
        assert "overall" in result
        assert 1.0 <= result["overall"] <= 5.0

    @pytest.mark.asyncio
    async def test_judge_quality_llm_failure_fallback(self, tmp_db, mock_hf, sample_call):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
        eval = self._make_evaluator(tmp_db, mock_llm, mock_hf)
        result = await eval.judge_quality(sample_call)
        assert "overall" in result
        assert 1.0 <= result["overall"] <= 5.0

    @pytest.mark.asyncio
    async def test_judge_hallucination_no_hallucination(self, tmp_db, mock_hf, sample_call):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value='{"has_hallucination":false,"confidence":0.1,"suspect_claims":[]}'
        )
        eval = self._make_evaluator(tmp_db, mock_llm, mock_hf)
        result = await eval.judge_hallucination(sample_call)
        assert result["has_hallucination"] is False
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_helm_requires_10_calls(self, tmp_db, mock_llm, mock_hf, sample_call):
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        eval = BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        calls = [sample_call] * 5
        result = await eval.evaluate_with_helm(calls)
        assert result.sample_count == 5

    @pytest.mark.asyncio
    async def test_helm_score_range(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.metrics_collector import LLMCall
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        calls = []
        for i in range(15):
            c = LLMCall(
                call_id=str(uuid.uuid4()), session_id="s", provider="anthropic",
                model="claude-sonnet-4-6", timestamp="2026-06-09T00:00:00+00:00",
                latency_ms=float(300 + i * 50), cost_usd=0.0001 * i,
                quality_score=float(2 + (i % 4)), hallucination_score=0.1,
                context_utilization=0.01,
            )
            calls.append(c)
        eval = BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        result = await eval.evaluate_with_helm(calls)
        assert 0.0 <= result.composite <= 1.0
        assert result.sample_count == 15

    @pytest.mark.asyncio
    async def test_elo_rating_winner_has_higher_elo(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.metrics_collector import LLMCall
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator

        shared_hash = "abc123"
        model_calls = {
            "model-a": [
                LLMCall(call_id=str(uuid.uuid4()), prompt_hash=shared_hash,
                        model="model-a", quality_score=4.5, session_id="s")
            ],
            "model-b": [
                LLMCall(call_id=str(uuid.uuid4()), prompt_hash=shared_hash,
                        model="model-b", quality_score=3.5, session_id="s")
            ],
        }
        eval = BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        ratings = await eval.compute_elo_ratings(model_calls)
        assert ratings.get("model-a", 1000) > ratings.get("model-b", 1000)

    @pytest.mark.asyncio
    async def test_aggregate_report_structure(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.metrics_collector import LLMCall
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        calls = [
            LLMCall(call_id=str(uuid.uuid4()), session_id="s", model="test-model",
                    quality_score=3.5 + i * 0.1, hallucination_score=0.05,
                    latency_ms=300.0, cost_usd=0.0001, task_type="coding",
                    timestamp="2026-06-09T00:00:00+00:00")
            for i in range(12)
        ]
        eval = BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        result = await eval.aggregate_benchmark_report("test-model", calls)
        assert "model" in result
        assert "helm" in result
        assert "mt_bench" in result
        assert "operational" in result


# ── ReportGenerator (6 tests) ────────────────────────────────────────────────

class TestReportGenerator:
    def _make_generator(self, tmp_db, mock_llm, tmp_path):
        from agent.modules.report_generator import ReportGenerator
        return ReportGenerator(config={}, memory=tmp_db, llm=mock_llm, output_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_empty_report_no_crash(self, tmp_db, mock_llm, tmp_path):
        gen = self._make_generator(tmp_db, mock_llm, tmp_path)
        result = await gen.generate_rolling_report(days=30)
        assert isinstance(result, str)
        assert "No data available" in result

    @pytest.mark.asyncio
    async def test_export_csv_empty(self, tmp_db, mock_llm, tmp_path):
        gen = self._make_generator(tmp_db, mock_llm, tmp_path)
        csv = await gen.export_csv(days=30)
        assert "call_id" in csv

    @pytest.mark.asyncio
    async def test_report_writes_md_file(self, tmp_db, mock_llm, tmp_path):
        from agent.modules.metrics_collector import LLMCall
        call = LLMCall(
            call_id=str(uuid.uuid4()), session_id="s", provider="anthropic",
            model="claude-sonnet-4-6", timestamp="2026-06-09T10:00:00+00:00",
            latency_ms=300.0, cost_usd=0.0001, quality_score=4.0,
            input_tokens=50, output_tokens=100, task_type="coding",
        )
        await tmp_db.save_call(call)
        await tmp_db.upsert_session("s", call)

        gen = self._make_generator(tmp_db, mock_llm, tmp_path)
        mock_llm.complete = AsyncMock(return_value="Summary: performance was adequate.")
        await gen.generate_rolling_report(days=30)

        report_file = tmp_path / "BENCHMARK-RESULTS.md"
        assert report_file.exists()
        content = report_file.read_text()
        assert "Benchmark Report" in content

    def test_section_metrics_table(self, tmp_db, mock_llm, tmp_path):
        gen = self._make_generator(tmp_db, mock_llm, tmp_path)
        summary = {
            "avg_latency_ms": 342.5, "p95_latency_ms": 800.0,
            "avg_cost_per_call": 0.000125, "total_cost_usd": 0.0042,
            "avg_task_success": 0.82, "avg_correction_count": 0.3,
            "avg_hallucination_score": 0.05, "avg_tool_success": None,
            "avg_context_utilization": 0.012, "avg_quality_score": 3.8,
        }
        result = gen._section_metrics_table(summary)
        assert "Latency" in result
        assert "342" in result
        assert "Quality Score" in result

    @pytest.mark.asyncio
    async def test_export_csv_with_data(self, tmp_db, mock_llm, tmp_path):
        from agent.modules.metrics_collector import LLMCall
        for i in range(3):
            call = LLMCall(
                call_id=str(uuid.uuid4()), session_id="s", provider="openai",
                model="gpt-4o-mini", timestamp="2026-06-09T10:00:00+00:00",
                latency_ms=200.0 + i, cost_usd=0.00005 * i, quality_score=3.5,
                input_tokens=20, output_tokens=50, task_type="writing",
            )
            await tmp_db.save_call(call)
        gen = self._make_generator(tmp_db, mock_llm, tmp_path)
        csv = await gen.export_csv(days=30)
        lines = csv.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_task_breakdown_ascii_chart(self, tmp_db, mock_llm, tmp_path):
        gen = self._make_generator(tmp_db, mock_llm, tmp_path)
        task_data = {
            "by_task_type": {
                "coding": {"count": 20, "avg_quality": 4.2, "avg_latency": 350.0},
                "writing": {"count": 10, "avg_quality": 3.8, "avg_latency": 250.0},
            }
        }
        result = gen._section_task_breakdown(task_data)
        assert "coding" in result
        assert "writing" in result
        assert "█" in result


# ── ModelComparator (5 tests) ────────────────────────────────────────────────

class TestModelComparator:
    def _make_comparator(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.model_comparator import ModelComparator
        return ModelComparator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)

    def test_efficiency_score_formula(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.model_comparator import ModelComparator, ModelScore
        comp = ModelComparator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        s = ModelScore(model="test", avg_quality=4.5, avg_latency_ms=300.0,
                       avg_cost_per_call=0.0001, task_success_rate=0.9)
        score = comp._compute_efficiency(s)
        assert score > 0.0
        assert score <= 10.0

    def test_find_best_overall_highest_quality(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.model_comparator import ModelComparator, ModelScore
        comp = ModelComparator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        scores = [
            ModelScore(model="best-model", avg_quality=4.8, correction_rate=0.05, hallucination_rate=0.02, efficiency_score=8.0),
            ModelScore(model="ok-model", avg_quality=3.5, correction_rate=0.20, hallucination_rate=0.10, efficiency_score=4.0),
        ]
        best = comp._find_best_overall(scores)
        assert best == "best-model"

    def test_find_best_value_cost_quality(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.model_comparator import ModelComparator, ModelScore
        comp = ModelComparator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        scores = [
            ModelScore(model="expensive", avg_quality=4.5, avg_cost_per_call=0.002),
            ModelScore(model="cheap-good", avg_quality=4.0, avg_cost_per_call=0.00005),
        ]
        best_val = comp._find_best_value(scores)
        assert best_val == "cheap-good"

    @pytest.mark.asyncio
    async def test_compare_models_no_data(self, tmp_db, mock_llm, mock_hf):
        comp = self._make_comparator(tmp_db, mock_llm, mock_hf)
        result = await comp.compare_models(days=30)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_find_best_for_task(self, tmp_db, mock_llm, mock_hf):
        comp = self._make_comparator(tmp_db, mock_llm, mock_hf)
        result = await comp.find_best_model_for_task("coding")
        assert "task_type" in result


# ── MemoryManager (6 tests) ───────────────────────────────────────────────────

class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_save_and_count_call(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        count = await tmp_db.count_calls()
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_calls_by_model(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        calls = await tmp_db.get_calls(model="claude-sonnet-4-6", days=30)
        assert len(calls) == 1
        assert calls[0].model == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_update_call_scores(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        await tmp_db.update_call_scores(sample_call.call_id, {"quality_score": 4.2, "task_success": 0.9})
        calls = await tmp_db.get_calls(days=30)
        assert abs(calls[0].quality_score - 4.2) < 0.01

    @pytest.mark.asyncio
    async def test_upsert_session(self, tmp_db, sample_call):
        await tmp_db.upsert_session("sess-001", sample_call)
        sessions = await tmp_db.get_recent_sessions(limit=5)
        assert len(sessions) >= 1

    @pytest.mark.asyncio
    async def test_get_metrics_summary_empty(self, tmp_db):
        summary = await tmp_db.get_metrics_summary(days=7)
        assert summary.get("total_calls", 0) == 0

    def test_knowledge_hash_dedup(self, tmp_db):
        tmp_db.mark_paper_known("hash001", "Test Paper")
        assert tmp_db.is_known_paper("hash001") is True
        assert tmp_db.is_known_paper("hash999") is False


# ── LLMClient (4 tests) ────────────────────────────────────────────────────────

class TestLLMClient:
    def test_build_chain_with_keys(self):
        import os
        from tools.llm_client import UnifiedLLMClient
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "OPENAI_API_KEY": "sk-openai"}):
            with patch.dict(os.environ, {"PRIVACY_MODE": "false"}):
                client = UnifiedLLMClient()
                assert "claude" in client._providers

    def test_privacy_mode_only_ollama(self):
        import os
        with patch.dict(os.environ, {"PRIVACY_MODE": "true", "ANTHROPIC_API_KEY": "sk-test"}):
            from tools import llm_client as lm
            lm.PRIVACY_MODE = True
            client = lm.UnifiedLLMClient()
            assert client._providers == ["ollama"]
            lm.PRIVACY_MODE = False

    @pytest.mark.asyncio
    async def test_complete_all_providers_fail(self):
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        client._providers = ["ollama"]
        with patch.object(client, "_call_with_retry", new_callable=AsyncMock, side_effect=RuntimeError("down")):
            result = await client.complete("hello", task="test")
        assert "unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_complete_returns_string(self):
        from tools.llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        with patch.object(client, "_call_with_retry", new_callable=AsyncMock, return_value='{"overall":4}'):
            result = await client.complete("test prompt", task="quality_judge")
        assert isinstance(result, str)


# ── HFModelManager (3 tests) ─────────────────────────────────────────────────

class TestHFModelManager:
    def test_singleton(self):
        from tools.hf_model_manager import HFModelManager
        a = HFModelManager.get_instance()
        b = HFModelManager.get_instance()
        assert a is b

    def test_tfidf_fallback_encode_shape(self):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        texts = ["hello world", "write code", "translate text"]
        embs = mgr._tfidf_fallback_encode(texts)
        assert embs.shape == (3, 384)

    def test_tfidf_fallback_normalized(self):
        import numpy as np
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        emb = mgr._tfidf_fallback_encode(["hello world"])
        norm = float(np.linalg.norm(emb[0]))
        assert abs(norm - 1.0) < 0.01 or norm == 0.0


# ── Integration (4 tests) ─────────────────────────────────────────────────────

class TestIntegration:
    @pytest.mark.asyncio
    async def test_save_and_retrieve_call_roundtrip(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        await tmp_db.update_call_scores(sample_call.call_id, {"quality_score": 3.7})
        calls = await tmp_db.get_calls(model="claude-sonnet-4-6", days=1)
        assert len(calls) == 1
        assert abs(calls[0].quality_score - 3.7) < 0.01

    @pytest.mark.asyncio
    async def test_per_model_summary(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        await tmp_db.update_call_scores(sample_call.call_id, {"quality_score": 4.0})
        summary = await tmp_db.get_per_model_summary(days=30)
        assert "claude-sonnet-4-6" in summary

    @pytest.mark.asyncio
    async def test_task_breakdown(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        breakdown = await tmp_db.get_task_type_breakdown(days=30)
        by_type = breakdown.get("by_task_type", {})
        assert "coding" in by_type

    @pytest.mark.asyncio
    async def test_cost_summary(self, tmp_db, sample_call):
        await tmp_db.save_call(sample_call)
        result = await tmp_db.get_cost_summary(days=30)
        assert "total_cost_usd" in result
        assert "by_model" in result


# ── CLI smoke (5 tests) ────────────────────────────────────────────────────────

class TestCLISmoke:
    def test_cli_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower() or "Usage" in result.output

    def test_report_command_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0

    def test_compare_command_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0

    def test_cost_report_command_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["cost-report", "--help"])
        assert result.exit_code == 0

    def test_export_command_help(self):
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ── Response Preview (4 tests) ──────────────────────────────────────────────

class TestResponsePreview:
    def test_extract_response_text_anthropic(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        col = MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)
        request = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]}
        response = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 5, "output_tokens": 20},
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
        }
        call = col.extract_call(request, response, "anthropic", 100.0, "s1")
        assert call is not None
        assert "Hello! How can I help?" in call.response_preview

    def test_extract_response_text_openai(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        col = MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)
        request = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}
        response = {
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 3, "completion_tokens": 15},
            "choices": [{"message": {"content": "Sure, here is the answer."}}],
        }
        call = col.extract_call(request, response, "openai", 80.0, "s2")
        assert call is not None
        assert "Sure, here is the answer." in call.response_preview

    @pytest.mark.asyncio
    async def test_judge_uses_response_preview(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.metrics_collector import LLMCall
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        call = LLMCall(
            call_id=str(uuid.uuid4()), session_id="s", model="test",
            response_preview="This is the actual response text for judging.",
            prompt_preview="Write code", task_type="coding",
        )
        evaluator = BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        text = evaluator._get_response_text(call)
        assert text == "This is the actual response text for judging."

    @pytest.mark.asyncio
    async def test_response_preview_roundtrip(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        col = MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)
        request = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "test"}]}
        response = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 5, "output_tokens": 10},
            "content": [{"type": "text", "text": "Response for judge"}],
        }
        call = col.extract_call(request, response, "anthropic", 50.0, "s-rt")
        assert call is not None
        await tmp_db.save_call(call)
        calls = await tmp_db.get_calls(days=30)
        assert len(calls) == 1
        assert calls[0].response_preview == "Response for judge"


# ── ELO Persistence (3 tests) ───────────────────────────────────────────────

class TestEloPersistence:
    @pytest.mark.asyncio
    async def test_persist_and_retrieve_elo(self, tmp_db, mock_llm, mock_hf):
        await tmp_db.update_elo_rating("model-a", 1025.3)
        await tmp_db.update_elo_rating("model-b", 985.7)
        ratings = await tmp_db.get_elo_ratings()
        assert ratings["model-a"] == 1025.3
        assert ratings["model-b"] == 985.7

    @pytest.mark.asyncio
    async def test_elo_update_overwrites(self, tmp_db, mock_llm, mock_hf):
        await tmp_db.update_elo_rating("model-x", 1000.0)
        await tmp_db.update_elo_rating("model-x", 1050.0)
        ratings = await tmp_db.get_elo_ratings()
        assert ratings["model-x"] == 1050.0

    @pytest.mark.asyncio
    async def test_elo_persist_via_evaluator(self, tmp_db, mock_llm, mock_hf):
        from agent.modules.metrics_collector import LLMCall
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator(config={}, memory=tmp_db, llm=mock_llm, hf=mock_hf)
        shared_hash = "shared123"
        model_calls = {
            "alpha": [
                LLMCall(call_id=str(uuid.uuid4()), prompt_hash=shared_hash,
                        model="alpha", quality_score=4.5, session_id="s")
            ],
            "beta": [
                LLMCall(call_id=str(uuid.uuid4()), prompt_hash=shared_hash,
                        model="beta", quality_score=3.0, session_id="s")
            ],
        }
        ratings = await evaluator.compute_elo_ratings(model_calls)
        await evaluator.persist_elo_ratings(ratings, tmp_db)
        stored = await tmp_db.get_elo_ratings()
        assert "alpha" in stored
        assert "beta" in stored
        assert stored["alpha"] > stored["beta"]


# ── Prometheus Client (5 tests) ─────────────────────────────────────────────

class TestPrometheusClient:
    def test_generate_empty(self):
        from tools.prometheus_client import generate, _metrics
        _metrics["calls_total"] = 0
        _metrics["latency_sum_ms"] = 0
        output = generate()
        assert "benchmark_calls_total 0" in output
        assert "benchmark_cost_total_usd" in output

    def test_record_call_and_generate(self):
        from tools.prometheus_client import record_call, generate, _metrics, _labels, _lock
        with _lock:
            _metrics.update({
                "calls_total": 0, "cost_total_usd": 0, "latency_sum_ms": 0,
                "latency_count": 0, "quality_sum": 0, "quality_count": 0,
                "hallucination_sum": 0, "hallucination_count": 0,
                "corrections_total": 0, "tool_calls_total": 0,
                "context_utilization_sum": 0, "context_utilization_count": 0,
                "task_success_sum": 0, "task_success_count": 0,
            })
            _labels.clear()
        record_call(
            model="test-model", provider="anthropic", task_type="coding",
            latency_ms=350.0, cost_usd=0.0005, quality_score=4.2,
            input_tokens=100, output_tokens=200,
        )
        output = generate()
        assert "benchmark_calls_total 1" in output
        assert "350.00" in output
        assert "4.200" in output

    def test_record_correction(self):
        from tools.prometheus_client import record_call, generate, _metrics, _labels, _lock
        with _lock:
            _metrics.update({
                "calls_total": 0, "cost_total_usd": 0, "latency_sum_ms": 0,
                "latency_count": 0, "quality_sum": 0, "quality_count": 0,
                "hallucination_sum": 0, "hallucination_count": 0,
                "corrections_total": 0, "tool_calls_total": 0,
                "context_utilization_sum": 0, "context_utilization_count": 0,
                "task_success_sum": 0, "task_success_count": 0,
            })
            _labels.clear()
        record_call(model="m", provider="p", task_type="t", latency_ms=100, cost_usd=0, is_correction=True)
        output = generate()
        assert "benchmark_corrections_total 1" in output

    def test_per_label_metrics(self):
        from tools.prometheus_client import record_call, generate, _metrics, _labels, _lock
        with _lock:
            _metrics.update({
                "calls_total": 0, "cost_total_usd": 0, "latency_sum_ms": 0,
                "latency_count": 0, "quality_sum": 0, "quality_count": 0,
                "hallucination_sum": 0, "hallucination_count": 0,
                "corrections_total": 0, "tool_calls_total": 0,
                "context_utilization_sum": 0, "context_utilization_count": 0,
                "task_success_sum": 0, "task_success_count": 0,
            })
            _labels.clear()
        record_call(model="gpt-4o", provider="openai", task_type="qa",
                     latency_ms=200, cost_usd=0.001, input_tokens=50, output_tokens=100)
        output = generate()
        assert 'provider="openai"' in output
        assert 'model="gpt-4o"' in output
        assert 'task_type="qa"' in output

    def test_tool_call_recording(self):
        from tools.prometheus_client import record_call, generate, _metrics, _labels, _lock
        with _lock:
            _metrics.update({
                "calls_total": 0, "cost_total_usd": 0, "latency_sum_ms": 0,
                "latency_count": 0, "quality_sum": 0, "quality_count": 0,
                "hallucination_sum": 0, "hallucination_count": 0,
                "corrections_total": 0, "tool_calls_total": 0,
                "context_utilization_sum": 0, "context_utilization_count": 0,
                "task_success_sum": 0, "task_success_count": 0,
            })
            _labels.clear()
        record_call(model="m", provider="p", task_type="t", latency_ms=50, cost_usd=0, has_tool_call=True)
        output = generate()
        assert "benchmark_tool_calls_total 1" in output


# ── Orchestrator Integration (4 tests) ───────────────────────────────────────

class TestOrchestratorIntegration:
    def _make_orchestrator(self, tmp_path):
        from agent.orchestrator import BenchmarkOrchestrator
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "memory:\n  db_path: '" + str(tmp_path / "test.db").replace("\\", "/") + "'\n"
            "data:\n  output_dir: '" + str(tmp_path / "data").replace("\\", "/") + "'\n"
            "providers:\n  anthropic:\n    base_url: 'https://api.anthropic.com'\n"
        )
        return BenchmarkOrchestrator(config_path=config_path)

    @pytest.mark.asyncio
    async def test_start_and_stop(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        await orch.start(scheduler=False)
        count = await orch.memory.count_calls()
        assert count == 0
        await orch.stop()

    @pytest.mark.asyncio
    async def test_intercept_and_record(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        await orch.start(scheduler=False)
        await orch.intercept_and_record(
            request_body={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Hi"}]},
            response_body={
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 5, "output_tokens": 10},
                "content": [{"type": "text", "text": "Hello"}],
            },
            provider="anthropic",
            path="v1/messages",
            latency_ms=150.0,
            session_id="test-session-001",
        )
        import asyncio
        await asyncio.sleep(0.1)
        count = await orch.memory.count_calls()
        assert count == 1
        await orch.stop()

    @pytest.mark.asyncio
    async def test_metrics_summary_endpoint(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        await orch.start(scheduler=False)
        summary = await orch.get_metrics_summary(days=7)
        assert "total_calls" in summary
        await orch.stop()

    @pytest.mark.asyncio
    async def test_prometheus_metrics_format(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        await orch.start(scheduler=False)
        output = await orch.get_prometheus_metrics()
        assert "benchmark_calls_total" in output
        assert "# TYPE" in output
        assert "# HELP" in output
        await orch.stop()


# ── Edge Cases (5 tests) ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_request_returns_none(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        col = MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)
        call = col.extract_call({}, {}, "anthropic", 100.0, "s")
        assert call is None

    def test_very_long_response_truncated(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        col = MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)
        long_text = "x" * 5000
        response = {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 5, "output_tokens": 5000},
            "content": [{"type": "text", "text": long_text}],
        }
        request = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "test"}]}
        call = col.extract_call(request, response, "anthropic", 100.0, "s")
        assert call is not None
        assert len(call.response_preview) <= 500

    def test_unknown_model_zero_cost(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import MetricsCollector
        col = MetricsCollector(config={}, memory=tmp_db, hf=mock_hf)
        cost = col._compute_cost("totally-unknown-model-xyz", 1000, 500)
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_multiple_models_session(self, tmp_db, sample_call):
        from agent.modules.metrics_collector import LLMCall
        call_a = LLMCall(
            call_id=str(uuid.uuid4()), session_id="multi-session",
            provider="anthropic", model="claude-sonnet-4-6",
            timestamp="2026-06-09T10:00:00+00:00",
            latency_ms=200, cost_usd=0.0001, quality_score=4.0,
            task_type="coding", input_tokens=50, output_tokens=100,
        )
        call_b = LLMCall(
            call_id=str(uuid.uuid4()), session_id="multi-session",
            provider="openai", model="gpt-4o",
            timestamp="2026-06-09T10:01:00+00:00",
            latency_ms=350, cost_usd=0.0003, quality_score=3.5,
            task_type="writing", input_tokens=80, output_tokens=200,
        )
        await tmp_db.save_call(call_a)
        await tmp_db.save_call(call_b)
        summary = await tmp_db.get_per_model_summary(days=30)
        assert "claude-sonnet-4-6" in summary
        assert "gpt-4o" in summary

    @pytest.mark.asyncio
    async def test_heuristic_quality_range(self, tmp_db, mock_hf):
        from agent.modules.metrics_collector import LLMCall
        from agent.modules.benchmark_evaluator import BenchmarkEvaluator
        evaluator = BenchmarkEvaluator(config={}, memory=tmp_db, llm=None, hf=mock_hf)
        call = LLMCall(
            call_id=str(uuid.uuid4()), latency_ms=500.0, output_tokens=200,
            prompt_preview="test", task_type="coding",
        )
        result = evaluator._heuristic_quality(call)
        assert 1.0 <= result["overall"] <= 5.0
        assert "correctness" in result
        assert "conciseness" in result
