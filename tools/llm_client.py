"""
UnifiedLLMClient — Claude (primary) → OpenAI (fallback) → Ollama (offline).
Streaming + retry with exponential backoff. Cost tracking per call.
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

COST_PER_1K: dict = {
    "claude-opus-4-8":           {"input": 15.0,   "output": 75.0},
    "claude-sonnet-4-6":         {"input": 3.0,    "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.8,    "output": 4.0},
    "gpt-4o":                    {"input": 2.5,    "output": 10.0},
    "gpt-4o-mini":               {"input": 0.15,   "output": 0.60},
    "llama3":                    {"input": 0.0,    "output": 0.0},
    "mistral-large-2":           {"input": 2.0,    "output": 6.0},
}

PRIVACY_MODE = os.getenv("PRIVACY_MODE", "false").lower() == "true"


class UnifiedLLMClient:
    def __init__(self, memory=None):
        self.memory = memory
        self._providers = self._build_chain()
        self._claude_client = None
        self._openai_client = None

    def _get_claude_client(self):
        if self._claude_client is None:
            import anthropic
            self._claude_client = anthropic.AsyncAnthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        return self._claude_client

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )
        return self._openai_client

    def _build_chain(self) -> list:
        if PRIVACY_MODE:
            return ["ollama"]
        chain = []
        if os.getenv("ANTHROPIC_API_KEY"):
            chain.append("claude")
        if os.getenv("OPENAI_API_KEY"):
            chain.append("openai")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        chain.append("ollama")
        return chain or ["ollama"]

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        system: Optional[str] = None,
        task: str = "general",
    ) -> str:
        for provider in self._providers:
            try:
                result = await self._call_with_retry(provider, prompt, max_tokens, temperature, system)
                await self._log_cost(provider, prompt, result, task)
                return result
            except Exception as exc:
                logger.debug(f"Provider {provider} failed ({task}): {exc}")
                continue
        logger.error("All LLM providers failed")
        return f"[LLM unavailable — all providers exhausted for task: {task}]"

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        for provider in self._providers:
            try:
                async for chunk in self._stream_provider(provider, prompt, max_tokens, temperature, system):
                    yield chunk
                return
            except Exception as exc:
                logger.debug(f"Streaming provider {provider} failed: {exc}")
                continue

    async def complete_sync(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        return await self.complete(prompt, max_tokens=max_tokens, **kwargs)

    async def _call_with_retry(
        self, provider: str, prompt: str, max_tokens: int, temperature: float, system: Optional[str]
    ) -> str:
        delays = [1.0, 2.0, 4.0]
        last_exc = None
        for delay in [0] + delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                if provider == "claude":
                    return await self._call_claude(prompt, max_tokens, temperature, system)
                if provider == "openai":
                    return await self._call_openai(prompt, max_tokens, temperature, system)
                if provider == "ollama":
                    return await self._call_ollama(prompt, max_tokens, temperature, system)
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                continue
            except Exception as exc:
                raise exc
        raise last_exc or RuntimeError(f"Provider {provider} failed after retries")

    async def _call_claude(self, prompt: str, max_tokens: int, temperature: float, system: Optional[str]) -> str:
        try:
            client = self._get_claude_client()
        except (ImportError, KeyError):
            raise RuntimeError("anthropic package not installed or ANTHROPIC_API_KEY not set")
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        msg = await client.messages.create(**kwargs)
        return msg.content[0].text

    async def _call_openai(self, prompt: str, max_tokens: int, temperature: float, system: Optional[str]) -> str:
        try:
            client = self._get_openai_client()
        except (ImportError, KeyError):
            raise RuntimeError("openai package not installed or OPENAI_API_KEY not set")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    async def _call_ollama(self, prompt: str, max_tokens: int, temperature: float, system: Optional[str]) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(f"{base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned {resp.status}")
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    async def _stream_provider(
        self, provider: str, prompt: str, max_tokens: int, temperature: float, system: Optional[str]
    ) -> AsyncGenerator[str, None]:
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"num_predict": max_tokens},
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(f"{base_url}/api/chat", json=payload) as resp:
                    async for line in resp.content:
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        else:
            text = await self._call_with_retry(provider, prompt, max_tokens, temperature, system)
            yield text

    async def _log_cost(self, provider: str, prompt: str, response: str, task: str):
        if not self.memory:
            return
        model_key = {
            "claude": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "ollama": os.getenv("OLLAMA_MODEL", "llama3"),
        }.get(provider, provider)
        pricing = COST_PER_1K.get(model_key, {"input": 0.0, "output": 0.0})
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(response) // 4)
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        try:
            from tools.prometheus_client import record_call
            record_call(
                model=model_key,
                provider=provider,
                task_type="internal_judge",
                latency_ms=0.0,
                cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            logger.debug(f"Cost logging error: {exc}")


_RETRYABLE_ERRORS = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    ConnectionResetError,
)
