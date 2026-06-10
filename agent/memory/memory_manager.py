"""
BenchmarkMemoryManager — SQLite persistence for all benchmark data.
5 tables: llm_calls, sessions, benchmark_results, elo_ratings, knowledge_hashes.
"""

import asyncio
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id              TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL,
    provider             TEXT NOT NULL DEFAULT '',
    model                TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    prompt_hash          TEXT NOT NULL DEFAULT '',
    prompt_preview       TEXT DEFAULT '',
    response_preview     TEXT DEFAULT '',
    latency_ms           REAL DEFAULT 0.0,
    cost_usd             REAL DEFAULT 0.0,
    task_success         REAL,
    correction_count     INTEGER DEFAULT 0,
    hallucination_score  REAL,
    tool_call_success    REAL,
    context_utilization  REAL DEFAULT 0.0,
    quality_score        REAL,
    input_tokens         INTEGER DEFAULT 0,
    output_tokens        INTEGER DEFAULT 0,
    task_type            TEXT DEFAULT 'unknown',
    is_correction        INTEGER DEFAULT 0,
    has_tool_call        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_calls_model     ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_calls_session   ON llm_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_calls_task_type ON llm_calls(task_type);
CREATE INDEX IF NOT EXISTS idx_calls_hash      ON llm_calls(prompt_hash);

CREATE TABLE IF NOT EXISTS sessions (
    session_id           TEXT PRIMARY KEY,
    model                TEXT NOT NULL DEFAULT '',
    provider             TEXT NOT NULL DEFAULT '',
    started_at           TEXT NOT NULL,
    last_seen_at         TEXT NOT NULL,
    turn_count           INTEGER DEFAULT 0,
    correction_count     INTEGER DEFAULT 0,
    total_cost_usd       REAL DEFAULT 0.0,
    avg_quality          REAL,
    dominant_task_type   TEXT DEFAULT 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_sessions_model ON sessions(model);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model            TEXT NOT NULL,
    benchmark_type   TEXT NOT NULL,
    score            REAL DEFAULT 0.0,
    details_json     TEXT DEFAULT '{}',
    evaluated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bench_model ON benchmark_results(model);

CREATE TABLE IF NOT EXISTS elo_ratings (
    model       TEXT PRIMARY KEY,
    rating      REAL DEFAULT 1000.0,
    games       INTEGER DEFAULT 0,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_hashes (
    url_hash    TEXT PRIMARY KEY,
    title       TEXT DEFAULT '',
    added_at    TEXT NOT NULL
);
"""


class BenchmarkMemoryManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def init_db(self):
        with self._conn() as conn:
            conn.executescript(DDL)
            existing_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(llm_calls)").fetchall()
            }
            if "response_preview" not in existing_cols:
                conn.execute(
                    "ALTER TABLE llm_calls ADD COLUMN response_preview TEXT DEFAULT ''"
                )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── LLM Calls ─────────────────────────────────────────────────────────────

    async def save_call(self, call) -> None:
        await asyncio.get_event_loop().run_in_executor(None, self._save_call_sync, call)

    def _save_call_sync(self, call):
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_calls VALUES (
                    :call_id,:session_id,:provider,:model,:timestamp,:prompt_hash,
                    :prompt_preview,:response_preview,:latency_ms,:cost_usd,:task_success,:correction_count,
                    :hallucination_score,:tool_call_success,:context_utilization,:quality_score,
                    :input_tokens,:output_tokens,:task_type,:is_correction,:has_tool_call
                )""",
                {
                    "call_id": call.call_id,
                    "session_id": call.session_id,
                    "provider": call.provider,
                    "model": call.model,
                    "timestamp": call.timestamp,
                    "prompt_hash": call.prompt_hash,
                    "prompt_preview": call.prompt_preview,
                    "response_preview": getattr(call, "response_preview", ""),
                    "latency_ms": call.latency_ms,
                    "cost_usd": call.cost_usd,
                    "task_success": call.task_success,
                    "correction_count": call.correction_count,
                    "hallucination_score": call.hallucination_score,
                    "tool_call_success": call.tool_call_success,
                    "context_utilization": call.context_utilization,
                    "quality_score": call.quality_score,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "task_type": call.task_type,
                    "is_correction": 1 if call.is_correction else 0,
                    "has_tool_call": 1 if call.has_tool_call else 0,
                },
            )

    async def update_call_scores(self, call_id: str, updates: dict) -> None:
        def _run():
            fields = ", ".join(f"{k}=:{k}" for k in updates)
            updates["call_id"] = call_id
            with self._lock, self._conn() as conn:
                conn.execute(f"UPDATE llm_calls SET {fields} WHERE call_id=:call_id", updates)
        await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_calls(self, model: Optional[str] = None, days: int = 30) -> List:
        from agent.modules.metrics_collector import LLMCall
        def _run():
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with self._conn() as conn:
                if model:
                    rows = conn.execute(
                        "SELECT * FROM llm_calls WHERE model=? AND timestamp>=? ORDER BY timestamp DESC",
                        (model, cutoff),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM llm_calls WHERE timestamp>=? ORDER BY timestamp DESC",
                        (cutoff,),
                    ).fetchall()
            return [_row_to_call(r) for r in rows]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_session_calls(self, session_id: str) -> List:
        from agent.modules.metrics_collector import LLMCall
        def _run():
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM llm_calls WHERE session_id=? ORDER BY timestamp ASC",
                    (session_id,),
                ).fetchall()
            return [_row_to_call(r) for r in rows]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def count_calls(self) -> int:
        def _run():
            with self._conn() as conn:
                return conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    # ── Sessions ──────────────────────────────────────────────────────────────

    async def upsert_session(self, session_id: str, call) -> None:
        def _run():
            now = call.timestamp
            with self._lock, self._conn() as conn:
                existing = conn.execute(
                    "SELECT * FROM sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """INSERT INTO sessions VALUES (
                            :sid,:model,:provider,:started,:last,:turns,:corrections,:cost,:quality,:task
                        )""",
                        {
                            "sid": session_id,
                            "model": call.model,
                            "provider": call.provider,
                            "started": now,
                            "last": now,
                            "turns": 1,
                            "corrections": 1 if call.is_correction else 0,
                            "cost": call.cost_usd,
                            "quality": call.quality_score,
                            "task": call.task_type,
                        },
                    )
                else:
                    corrections = existing["correction_count"] + (1 if call.is_correction else 0)
                    turns = existing["turn_count"] + 1
                    total_cost = existing["total_cost_usd"] + call.cost_usd
                    conn.execute(
                        """UPDATE sessions SET last_seen_at=?, turn_count=?,
                           correction_count=?, total_cost_usd=?, dominant_task_type=?
                           WHERE session_id=?""",
                        (now, turns, corrections, total_cost, call.task_type, session_id),
                    )
        await asyncio.get_event_loop().run_in_executor(None, _run)

    async def count_sessions(self) -> int:
        def _run():
            with self._conn() as conn:
                return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_recent_sessions(self, model: Optional[str] = None, limit: int = 20) -> List[dict]:
        def _run():
            with self._conn() as conn:
                if model:
                    rows = conn.execute(
                        "SELECT * FROM sessions WHERE model=? ORDER BY last_seen_at DESC LIMIT ?",
                        (model, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM sessions ORDER BY last_seen_at DESC LIMIT ?", (limit,)
                    ).fetchall()
            return [dict(r) for r in rows]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_top_correction_sessions(
        self, model: Optional[str] = None, limit: int = 5
    ) -> List[dict]:
        def _run():
            with self._conn() as conn:
                if model:
                    rows = conn.execute(
                        "SELECT * FROM sessions WHERE model=? ORDER BY correction_count DESC LIMIT ?",
                        (model, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM sessions ORDER BY correction_count DESC LIMIT ?", (limit,)
                    ).fetchall()
            return [dict(r) for r in rows]
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    # ── Aggregates ────────────────────────────────────────────────────────────

    async def get_metrics_summary(
        self, model: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        def _run():
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            params = [cutoff]
            where = "WHERE timestamp>=?"
            if model:
                where += " AND model=?"
                params.append(model)
            with self._conn() as conn:
                row = conn.execute(
                    f"""SELECT
                        COUNT(*) AS total_calls,
                        AVG(latency_ms) AS avg_latency_ms,
                        MAX(latency_ms) AS max_latency_ms,
                        SUM(cost_usd) AS total_cost_usd,
                        AVG(cost_usd) AS avg_cost_per_call,
                        AVG(task_success) AS avg_task_success,
                        AVG(hallucination_score) AS avg_hallucination_score,
                        AVG(tool_call_success) AS avg_tool_success,
                        AVG(context_utilization) AS avg_context_utilization,
                        AVG(quality_score) AS avg_quality_score,
                        SUM(is_correction) AS correction_calls
                    FROM llm_calls {where}""",
                    params,
                ).fetchone()

                latencies = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT latency_ms FROM llm_calls {where} AND latency_ms>0",
                        params,
                    ).fetchall()
                ]

                session_count = conn.execute(
                    "SELECT COUNT(*) FROM sessions",
                ).fetchone()[0]

            result = dict(row)
            result["total_sessions"] = session_count
            total_calls = result.get("total_calls") or 1
            result["correction_rate"] = (result.get("correction_calls") or 0) / total_calls
            result["avg_calls_per_day"] = total_calls / max(1, days)

            if latencies:
                import numpy as np
                result["p95_latency_ms"] = float(np.percentile(latencies, 95))
                result["p99_latency_ms"] = float(np.percentile(latencies, 99))
            else:
                result["p95_latency_ms"] = 0.0
                result["p99_latency_ms"] = 0.0

            return {k: round(v, 6) if isinstance(v, float) else v for k, v in result.items()}
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_per_model_summary(self, days: int = 30) -> Dict[str, dict]:
        def _run():
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT model,
                        COUNT(*) AS total_calls,
                        AVG(latency_ms) AS avg_latency_ms,
                        SUM(cost_usd) AS total_cost_usd,
                        AVG(cost_usd) AS avg_cost_per_call,
                        AVG(quality_score) AS avg_quality_score,
                        AVG(task_success) AS avg_task_success,
                        AVG(hallucination_score) AS avg_hallucination_score,
                        AVG(context_utilization) AS avg_context_utilization,
                        SUM(is_correction)*1.0/COUNT(*) AS correction_rate
                    FROM llm_calls WHERE timestamp>=? GROUP BY model""",
                    (cutoff,),
                ).fetchall()
            return {row["model"]: {k: round(v, 6) if isinstance(v, float) else v
                                   for k, v in dict(row).items()} for row in rows}
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_task_type_breakdown(
        self, model: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        def _run():
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            params = [cutoff]
            where = "WHERE timestamp>=?"
            if model:
                where += " AND model=?"
                params.append(model)
            with self._conn() as conn:
                rows = conn.execute(
                    f"""SELECT task_type, COUNT(*) AS count,
                        AVG(quality_score) AS avg_quality,
                        AVG(latency_ms) AS avg_latency
                    FROM llm_calls {where} GROUP BY task_type""",
                    params,
                ).fetchall()
            return {
                "by_task_type": {
                    row["task_type"]: {
                        "count": row["count"],
                        "avg_quality": round(row["avg_quality"], 3) if row["avg_quality"] else None,
                        "avg_latency": round(row["avg_latency"], 1) if row["avg_latency"] else None,
                    }
                    for row in rows
                }
            }
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_model_task_scores(self, model: str, days: int = 30) -> Dict[str, float]:
        def _run():
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT task_type, AVG(quality_score) AS avg_quality
                    FROM llm_calls WHERE model=? AND timestamp>=? AND quality_score IS NOT NULL
                    GROUP BY task_type""",
                    (model, cutoff),
                ).fetchall()
            return {row["task_type"]: round(row["avg_quality"] or 0.0, 3) for row in rows}
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        def _run():
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT model, SUM(cost_usd) AS total_cost, COUNT(*) AS calls,
                        AVG(cost_usd) AS avg_cost
                    FROM llm_calls WHERE timestamp>=? GROUP BY model ORDER BY total_cost DESC""",
                    (cutoff,),
                ).fetchall()
                total = conn.execute(
                    "SELECT SUM(cost_usd) FROM llm_calls WHERE timestamp>=?", (cutoff,)
                ).fetchone()[0] or 0.0
            return {
                "total_cost_usd": round(total, 6),
                "by_model": [
                    {
                        "model": r["model"],
                        "total_cost_usd": round(r["total_cost"] or 0, 6),
                        "calls": r["calls"],
                        "avg_cost_per_call": round(r["avg_cost"] or 0, 8),
                    }
                    for r in rows
                ],
            }
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    # ── ELO ratings ───────────────────────────────────────────────────────────

    async def get_elo_ratings(self) -> Dict[str, float]:
        def _run():
            with self._conn() as conn:
                rows = conn.execute("SELECT model, rating FROM elo_ratings").fetchall()
            return {r["model"]: r["rating"] for r in rows}
        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def update_elo_rating(self, model: str, rating: float) -> None:
        def _run():
            now = datetime.now(timezone.utc).isoformat()
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT INTO elo_ratings(model, rating, games, updated_at) VALUES (?,?,1,?)
                    ON CONFLICT(model) DO UPDATE SET
                        rating=excluded.rating,
                        games=games+1,
                        updated_at=excluded.updated_at""",
                    (model, rating, now),
                )
        await asyncio.get_event_loop().run_in_executor(None, _run)

    # ── Knowledge hashes ──────────────────────────────────────────────────────

    def is_known_paper(self, url_hash: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM knowledge_hashes WHERE url_hash=?", (url_hash,)
            ).fetchone() is not None

    def mark_paper_known(self, url_hash: str, title: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_hashes(url_hash,title,added_at) VALUES(?,?,?)",
                (url_hash, title, now),
            )


def _row_to_call(row) -> Any:
    from agent.modules.metrics_collector import LLMCall
    return LLMCall(
        call_id=row["call_id"],
        session_id=row["session_id"],
        provider=row["provider"],
        model=row["model"],
        timestamp=row["timestamp"],
        prompt_hash=row["prompt_hash"],
        prompt_preview=row["prompt_preview"] or "",
        response_preview=row["response_preview"] if "response_preview" in row.keys() else "",
        latency_ms=row["latency_ms"] or 0.0,
        cost_usd=row["cost_usd"] or 0.0,
        task_success=row["task_success"],
        correction_count=row["correction_count"] or 0,
        hallucination_score=row["hallucination_score"],
        tool_call_success=row["tool_call_success"],
        context_utilization=row["context_utilization"] or 0.0,
        quality_score=row["quality_score"],
        input_tokens=row["input_tokens"] or 0,
        output_tokens=row["output_tokens"] or 0,
        task_type=row["task_type"] or "unknown",
        is_correction=bool(row["is_correction"]),
        has_tool_call=bool(row["has_tool_call"]),
    )
