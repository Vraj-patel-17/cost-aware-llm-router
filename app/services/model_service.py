from app.providers.litellm_provider import LiteLLMProvider
from app.schemas.llm import LLMResponse
class ModelService:
    def __init__(self,provider: LiteLLMProvider):
        self.provider = provider
    async def generate(self,prompt:str,model:str)->LLMResponse:
        return await self.provider.generate(prompt=prompt,model=model)
    