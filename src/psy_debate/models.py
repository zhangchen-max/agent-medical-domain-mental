from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except ValueError:
        return default


class DeepSeekHub:
    """Cloud DeepSeek API client."""

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = _float_env("MODEL_TIMEOUT_SECONDS", 30.0)
        self.risk_max_tokens = _int_env("RISK_MAX_TOKENS", 300)
        self.brain_max_tokens = _int_env("BRAIN_MAX_TOKENS", 1800)
        self.report_max_tokens = _int_env("REPORT_MAX_TOKENS", 1500)

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def call_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        completion = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = completion.choices[0].message.content or ""
        return _extract_json(content)

    async def risk_guard(self, system: str, user_input: str) -> dict[str, Any]:
        return await self.call_json(
            system=system,
            user=user_input,
            max_tokens=self.risk_max_tokens,
            temperature=0.1,
        )

    async def clinical_brain(self, system: str, payload: str) -> dict[str, Any]:
        return await self.call_json(
            system=system,
            user=payload,
            max_tokens=self.brain_max_tokens,
            temperature=0.3,
        )

    async def generate_report(self, system: str, payload: str) -> dict[str, Any]:
        return await self.call_json(
            system=system,
            user=payload,
            max_tokens=self.report_max_tokens,
            temperature=0.2,
        )


def _extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start: end + 1])
            except json.JSONDecodeError:
                pass
    return {}
