"""
MetricsCollector — extracts 8 operational metrics from every proxied LLM call.
"""

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Task type taxonomy ────────────────────────────────────────────────────────

class TaskType(str, Enum):
    CODING = "coding"
    WRITING = "writing"
    REASONING = "reasoning"
    EXTRACTION = "extraction"
    QA = "qa"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE = "creative"
    MATH = "math"
    UNKNOWN = "unknown"


TASK_TEMPLATES = {
    TaskType.CODING: [
        "write code", "implement function", "debug", "fix bug", "refactor",
        "python script", "javascript", "sql query", "algorithm",
    ],
    TaskType.WRITING: [
        "write an essay", "draft email", "write a blog", "compose",
        "write a report", "newsletter", "cover letter",
    ],
    TaskType.REASONING: [
        "analyze", "explain why", "what are the reasons", "compare and contrast",
        "evaluate", "pros and cons", "logical",
    ],
    TaskType.EXTRACTION: [
        "extract", "list all", "find all", "identify", "parse",
        "what are the", "enumerate",
    ],
    TaskType.QA: [
        "what is", "how does", "explain", "define", "tell me about",
        "what does", "describe",
    ],
    TaskType.SUMMARIZATION: [
        "summarize", "summary", "tldr", "brief overview", "key points",
        "main takeaways",
    ],
    TaskType.TRANSLATION: [
        "translate", "in french", "in spanish", "in vietnamese",
        "en français", "auf deutsch",
    ],
    TaskType.CREATIVE: [
        "write a story", "poem", "creative", "imagine", "fiction",
        "roleplay", "character",
    ],
    TaskType.MATH: [
        "calculate", "solve", "equation", "integral", "derivative",
        "probability", "statistics",
    ],
}

CORRECTION_KEYWORDS = [
    "wrong", "incorrect", "that's not", "not right", "fix this",
    "redo", "try again", "not what i", "you missed", "please redo",
    "that doesn't", "revise", "rethink", "change that", "no, actually",
    "not what i asked", "you're wrong", "that is wrong", "please fix",
    "not correct", "mistake", "error in your", "wrong answer",
]

# ── Model context windows ─────────────────────────────────────────────────────

MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-4-7": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "llama3": 128_000,
    "mistral-large-2": 128_000,
    "deepseek-v3": 64_000,
}

# Input $/1M tokens, Output $/1M tokens
MODEL_COST_TABLE: Dict[str, Dict[str, float]] = {
    "claude-opus-4-8":           {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":         {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.8,   "output": 4.0},
    "gpt-4o":                    {"input": 2.5,   "output": 10.0},
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":               {"input": 10.0,  "output": 30.0},
    "gpt-3.5-turbo":             {"input": 0.5,   "output": 1.5},
    "o1":                        {"input": 15.0,  "output": 60.0},
    "gemini-1.5-pro":            {"input": 3.5,   "output": 10.5},
    "gemini-1.5-flash":          {"input": 0.075, "output": 0.3},
    "llama3":                    {"input": 0.0,   "output": 0.0},
    "mistral-large-2":           {"input": 2.0,   "output": 6.0},
    "deepseek-v3":               {"input": 0.27,  "output": 1.10},
}


# ── LLMCall dataclass ─────────────────────────────────────────────────────────

@dataclass
class LLMCall:
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    provider: str = ""
    model: str = ""
    timestamp: str = ""
    prompt_hash: str = ""
    prompt_preview: str = ""  # first 200 chars only
    response_preview: str = ""  # first 500 chars for judge

    # 8 core metrics
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    task_success: Optional[float] = None   # 0.0–1.0, filled async by judge
    correction_count: int = 0              # updated when next turn detected
    hallucination_score: Optional[float] = None  # 0.0–1.0, filled async
    tool_call_success: Optional[float] = None    # null if no tools used
    context_utilization: float = 0.0
    quality_score: Optional[float] = None  # 1.0–5.0, filled async

    input_tokens: int = 0
    output_tokens: int = 0
    task_type: str = TaskType.UNKNOWN.value
    is_correction: bool = False
    has_tool_call: bool = False
    path: str = ""


class MetricsCollector:
    def __init__(self, config: dict, memory, hf):
        self.config = config
        self.memory = memory
        self.hf = hf

    def extract_call(
        self,
        request_body: dict,
        response_body: dict,
        provider: str,
        latency_ms: float,
        session_id: str,
    ) -> Optional[LLMCall]:
        try:
            model = self._extract_model(request_body, response_body, provider)
            if not model:
                return None

            prompt_text = self._extract_prompt(request_body, provider)
            if not prompt_text:
                return None

            input_tokens = self._extract_input_tokens(request_body, response_body, provider, prompt_text)
            output_tokens = self._extract_output_tokens(response_body, provider)

            prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
            prompt_preview = prompt_text[:200]
            response_text = self._extract_response_text(response_body, provider)
            response_preview = response_text[:500]

            cost = self._compute_cost(model, input_tokens, output_tokens)
            context_util = self._compute_context_utilization(model, input_tokens, output_tokens)
            task_type = self._classify_task(prompt_text)
            has_tool_call = self._has_tool_call(request_body, response_body)
            tool_success = self._compute_tool_success(response_body, provider) if has_tool_call else None
            is_correction = self._detect_correction_intent(prompt_text)

            return LLMCall(
                session_id=session_id,
                provider=provider,
                model=model,
                timestamp=_utc_now(),
                prompt_hash=prompt_hash,
                prompt_preview=prompt_preview,
                response_preview=response_preview,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost, 8),
                task_success=None,
                correction_count=0,
                hallucination_score=None,
                tool_call_success=tool_success,
                context_utilization=round(context_util, 4),
                quality_score=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                task_type=task_type,
                is_correction=is_correction,
                has_tool_call=has_tool_call,
            )
        except Exception as exc:
            logger.error(f"extract_call error: {exc}")
            return None

    def _extract_model(self, req: dict, resp: dict, provider: str) -> str:
        if provider == "anthropic":
            return req.get("model", resp.get("model", ""))
        if provider == "openai":
            return req.get("model", resp.get("model", ""))
        if provider == "ollama":
            return req.get("model", "")
        return req.get("model", "")

    def _extract_prompt(self, req: dict, provider: str) -> str:
        if provider == "anthropic":
            messages = req.get("messages", [])
            texts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
            return " ".join(texts)
        if provider == "openai":
            messages = req.get("messages", [])
            texts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
            return " ".join(texts)
        if provider == "ollama":
            messages = req.get("messages", [])
            return " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return str(req)

    def _extract_input_tokens(self, req: dict, resp: dict, provider: str, prompt_text: str) -> int:
        if provider == "anthropic":
            usage = resp.get("usage", {})
            return usage.get("input_tokens", 0) or self._count_tokens_approx(prompt_text)
        if provider == "openai":
            usage = resp.get("usage", {})
            return usage.get("prompt_tokens", 0) or self._count_tokens_approx(prompt_text)
        if provider == "ollama":
            return resp.get("prompt_eval_count", 0) or self._count_tokens_approx(prompt_text)
        return self._count_tokens_approx(prompt_text)

    def _extract_output_tokens(self, resp: dict, provider: str) -> int:
        if provider == "anthropic":
            usage = resp.get("usage", {})
            return usage.get("output_tokens", 0)
        if provider == "openai":
            usage = resp.get("usage", {})
            return usage.get("completion_tokens", 0)
        if provider == "ollama":
            return resp.get("eval_count", 0)
        return 0

    def _extract_response_text(self, resp: dict, provider: str) -> str:
        if provider == "anthropic":
            content = resp.get("content", [])
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return " ".join(texts)
        if provider == "openai":
            choices = resp.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
            return ""
        if provider == "ollama":
            return resp.get("message", {}).get("content", "")
        return ""

    def _count_tokens_approx(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_lower = model.lower()
        pricing = None
        for key, price in MODEL_COST_TABLE.items():
            if key in model_lower or model_lower in key:
                pricing = price
                break
        if pricing is None:
            return 0.0
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def _compute_context_utilization(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_lower = model.lower()
        ctx_window = 128_000
        for key, window in MODEL_CONTEXT_WINDOWS.items():
            if key in model_lower or model_lower in key:
                ctx_window = window
                break
        total = input_tokens + output_tokens
        return min(1.0, total / ctx_window)

    def _classify_task(self, prompt_text: str) -> str:
        prompt_lower = prompt_text.lower()
        scores: Dict[TaskType, int] = {}
        for task_type, keywords in TASK_TEMPLATES.items():
            scores[task_type] = sum(1 for kw in keywords if kw in prompt_lower)

        if not self.hf:
            # keyword fallback
            best = max(scores, key=lambda t: scores[t])
            return best.value if scores[best] > 0 else TaskType.UNKNOWN.value

        try:
            templates = [" ".join(TASK_TEMPLATES[t]) for t in TaskType if t != TaskType.UNKNOWN]
            task_keys = [t for t in TaskType if t != TaskType.UNKNOWN]
            prompt_emb = self.hf.encode(prompt_text[:512])
            template_embs = self.hf.encode_batch(templates)
            import numpy as np
            sims = template_embs @ prompt_emb / (
                np.linalg.norm(template_embs, axis=1) * np.linalg.norm(prompt_emb) + 1e-9
            )
            best_idx = int(np.argmax(sims))
            return task_keys[best_idx].value
        except Exception:
            best = max(scores, key=lambda t: scores[t])
            return best.value if scores[best] > 0 else TaskType.UNKNOWN.value

    def _has_tool_call(self, req: dict, resp: dict) -> bool:
        if req.get("tools") or req.get("tool_choice"):
            return True
        choices = resp.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            if msg.get("tool_calls"):
                return True
        if resp.get("stop_reason") == "tool_use":
            return True
        content = resp.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return True
        return False

    def _compute_tool_success(self, resp: dict, provider: str) -> Optional[float]:
        choices = resp.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # If any tool call has an error id — mark as failure
                return 1.0  # optimistic; will be updated if error observed
        if resp.get("stop_reason") == "tool_use":
            return 1.0
        return None

    def detect_correction_intent(self, message: str) -> bool:
        return self._detect_correction_intent(message)

    def _detect_correction_intent(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in CORRECTION_KEYWORDS)

    @staticmethod
    def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        import numpy as np
        arr = sorted(latencies)
        n = len(arr)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
