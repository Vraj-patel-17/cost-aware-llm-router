from app.providers.litellm_provider import LiteLLMProvider
from app.schemas.llm import LLMResponse
from app.services.cost_service import calculate_cost


class ModelService:
    def __init__(self, provider: LiteLLMProvider):
        self.provider = provider

    async def generate(self, prompt: str, model: str) -> LLMResponse:
        response = await self.provider.generate(
            prompt=prompt,
            model=model,
        )

        if (
            response.input_tokens is None
            or response.output_tokens is None
        ):
            return response

        estimated_cost = calculate_cost(
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return response.model_copy(
            update={"estimated_cost_usd": estimated_cost}
        )