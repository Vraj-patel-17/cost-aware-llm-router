import asyncio
import time

import litellm

from app.core.config import settings
from app.schemas.llm import LLMResponse


class LiteLLMProvider:
    def __init__(self, provider: str, api_key: str | None = None) -> None:
        self.provider = provider
        self.api_key = api_key

    async def generate(self, prompt: str, model: str) -> LLMResponse:
        start_time = time.perf_counter()
        request_options = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.api_key:
            request_options["api_key"] = self.api_key

        response = await asyncio.wait_for(
            litellm.acompletion(**request_options),
            timeout=settings.request_timeout_seconds,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        text = response.choices[0].message.content if response.choices else ""
        usage = response.usage
        return LLMResponse(
            text=text or "",
            model=model,
            provider=self.provider,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
        )