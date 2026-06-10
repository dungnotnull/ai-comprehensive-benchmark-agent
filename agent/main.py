"""
ai-benchmark-agent — transparent LLM proxy + evaluation layer
Entry point: FastAPI proxy server + Click CLI
Port: 8022 (proxy + REST API + metrics)
"""

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import click
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / "config" / ".env")
load_dotenv(ROOT / ".env")

from agent.orchestrator import BenchmarkOrchestrator

_orchestrator: Optional[BenchmarkOrchestrator] = None


def get_orchestrator() -> BenchmarkOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BenchmarkOrchestrator()
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    orch = get_orchestrator()
    await orch.start()
    yield
    await orch.stop()


app = FastAPI(
    title="ai-benchmark-agent",
    description="Transparent LLM proxy with 8-metric evaluation layer",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Proxy routes ──────────────────────────────────────────────────────────────

@app.api_route(
    "/anthropic/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_anthropic(path: str, request: Request):
    return await _proxy_request(request, "anthropic", path)


@app.api_route(
    "/openai/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_openai(path: str, request: Request):
    return await _proxy_request(request, "openai", path)


@app.api_route(
    "/ollama/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_ollama(path: str, request: Request):
    return await _proxy_request(request, "ollama", path)


async def _proxy_request(request: Request, provider: str, path: str) -> Response:
    orch = get_orchestrator()

    cfg = orch.config.get("providers", {}).get(provider, {})
    base_url = cfg.get("base_url", "")
    if not base_url:
        raise HTTPException(status_code=502, detail=f"Provider '{provider}' not configured")

    session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
    body_bytes = await request.body()
    body_dict: dict = {}
    if body_bytes:
        try:
            body_dict = json.loads(body_bytes)
        except json.JSONDecodeError:
            pass

    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "x-session-id", "content-length")
    }

    target_url = f"{base_url.rstrip('/')}/{path}"
    params = dict(request.query_params)

    start_ts = time.perf_counter()
    response_body = b""
    status_code = 200
    response_headers: dict = {}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=forward_headers,
                content=body_bytes,
                params=params,
            )
            latency_ms = (time.perf_counter() - start_ts) * 1000
            response_body = resp.content
            status_code = resp.status_code
            response_headers = dict(resp.headers)
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000
        status_code = 502
        response_body = json.dumps({"error": str(exc)}).encode()

    resp_dict: dict = {}
    if response_body:
        try:
            resp_dict = json.loads(response_body)
        except json.JSONDecodeError:
            pass

    asyncio.create_task(
        orch.intercept_and_record(
            request_body=body_dict,
            response_body=resp_dict,
            provider=provider,
            path=path,
            latency_ms=latency_ms,
            session_id=session_id,
            status_code=status_code,
        )
    )

    passthrough_headers = {
        k: v
        for k, v in response_headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
    }

    return Response(
        content=response_body,
        status_code=status_code,
        headers=passthrough_headers,
        media_type=response_headers.get("content-type", "application/json"),
    )


# ── REST API endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    orch = get_orchestrator()
    return {
        "status": "ok",
        "total_calls": await orch.memory.count_calls(),
        "total_sessions": await orch.memory.count_sessions(),
    }


@app.get("/api/v1/metrics")
async def get_metrics(model: Optional[str] = None, days: int = 7):
    orch = get_orchestrator()
    return await orch.get_metrics_summary(model=model, days=days)


@app.get("/api/v1/report")
async def get_report(model: Optional[str] = None, days: int = 30):
    orch = get_orchestrator()
    report_md = await orch.generate_report(model=model, days=days)
    return PlainTextResponse(report_md, media_type="text/markdown")


@app.get("/api/v1/compare")
async def compare_models(models: Optional[str] = None, days: int = 30):
    """Pass ?models=claude-opus-4-8,gpt-4o to compare specific models."""
    orch = get_orchestrator()
    model_list = models.split(",") if models else None
    return await orch.compare_models(model_ids=model_list, days=days)


@app.get("/api/v1/sessions")
async def list_sessions(model: Optional[str] = None, limit: int = 20):
    orch = get_orchestrator()
    return await orch.memory.get_recent_sessions(model=model, limit=limit)


@app.post("/api/v1/knowledge/update")
async def update_knowledge():
    orch = get_orchestrator()
    result = await orch.update_knowledge()
    return result


@app.get("/api/v1/cost")
async def get_cost_report(days: int = 30):
    orch = get_orchestrator()
    return await orch.memory.get_cost_summary(days=days)


@app.get("/api/v1/export/csv")
async def export_csv(model: Optional[str] = None, days: int = 30):
    orch = get_orchestrator()
    csv_data = await orch.report_gen.export_csv(model=model, days=days)
    return PlainTextResponse(
        csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark_export.csv"},
    )


@app.get("/metrics")
async def prometheus_metrics():
    orch = get_orchestrator()
    return PlainTextResponse(
        await orch.get_prometheus_metrics(),
        media_type="text/plain; version=0.0.4",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """ai-benchmark-agent — LLM performance evaluation proxy."""


@cli.command()
@click.option("--proxy-port", default=8022, show_default=True)
@click.option("--start-scheduler/--no-scheduler", default=True)
def start_proxy(proxy_port: int, start_scheduler: bool):
    """Start the benchmark proxy + REST API server."""
    click.echo(f"Starting ai-benchmark-agent proxy on port {proxy_port}...")
    click.echo(f"Anthropic endpoint: http://localhost:{proxy_port}/anthropic")
    click.echo(f"OpenAI endpoint:    http://localhost:{proxy_port}/openai")
    click.echo(f"Ollama endpoint:    http://localhost:{proxy_port}/ollama")
    click.echo(f"REST API:           http://localhost:{proxy_port}/api/v1/")
    os.environ["BENCHMARK_SCHEDULER"] = "1" if start_scheduler else "0"
    uvicorn.run(
        "agent.main:app",
        host="0.0.0.0",
        port=proxy_port,
        reload=False,
        log_level="info",
    )


@cli.command()
@click.option("--model", default=None, help="Filter by model name")
@click.option("--days", default=30, show_default=True)
@click.option("--output", default=None, help="Save report to file")
def report(model: Optional[str], days: int, output: Optional[str]):
    """Generate a Markdown benchmark report."""
    async def _run():
        orch = BenchmarkOrchestrator()
        await orch.start(scheduler=False)
        md = await orch.generate_report(model=model, days=days)
        if output:
            Path(output).write_text(md, encoding="utf-8")
            click.echo(f"Report saved to {output}")
        else:
            click.echo(md)
        await orch.stop()
    asyncio.run(_run())


@cli.command()
@click.option("--models", default=None, help="Comma-separated model IDs")
@click.option("--days", default=30, show_default=True)
def compare(models: Optional[str], days: int):
    """Compare models and show recommendations."""
    async def _run():
        orch = BenchmarkOrchestrator()
        await orch.start(scheduler=False)
        model_list = models.split(",") if models else None
        result = await orch.compare_models(model_ids=model_list, days=days)
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        await orch.stop()
    asyncio.run(_run())


@cli.command()
def update_knowledge():
    """Crawl latest LLM evaluation papers and update the knowledge brain."""
    async def _run():
        orch = BenchmarkOrchestrator()
        await orch.start(scheduler=False)
        result = await orch.update_knowledge()
        click.echo(json.dumps(result, indent=2))
        await orch.stop()
    asyncio.run(_run())


@cli.command()
@click.option("--days", default=30, show_default=True)
def cost_report(days: int):
    """Show token cost breakdown by model."""
    async def _run():
        orch = BenchmarkOrchestrator()
        await orch.start(scheduler=False)
        result = await orch.memory.get_cost_summary(days=days)
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        await orch.stop()
    asyncio.run(_run())


@cli.command()
@click.option("--model", default=None)
@click.option("--days", default=30, show_default=True)
@click.option("--output", default="benchmark_export.csv")
def export(model: Optional[str], days: int, output: str):
    """Export benchmark data as CSV."""
    async def _run():
        orch = BenchmarkOrchestrator()
        await orch.start(scheduler=False)
        csv_data = await orch.report_gen.export_csv(model=model, days=days)
        Path(output).write_text(csv_data, encoding="utf-8")
        click.echo(f"Exported to {output}")
        await orch.stop()
    asyncio.run(_run())


if __name__ == "__main__":
    cli()
