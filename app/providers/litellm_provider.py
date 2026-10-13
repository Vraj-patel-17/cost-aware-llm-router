import time
import litellm
from app.schemas.llm import LLMResponse

class LiteLLMProvider:
    def __init__(self,provider:str)-> None:
        self.provider=provider

    async def generate(self, prompt: str, model: str) -> LLMResponse:
        start_time = time.perf_counter()
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],)
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