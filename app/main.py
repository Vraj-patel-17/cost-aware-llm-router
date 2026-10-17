from fastapi import FastAPI , HTTPException
from app.core.config import settings
from app.providers.litellm_provider import LiteLLMProvider
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.model_service import ModelService

def get_model_service(provider_name: str) -> ModelService:
    if provider_name == "gemini":
        provider = LiteLLMProvider(
            provider="gemini",
            api_key=settings.gemini_api_key,
        )
        return ModelService(provider=provider)

    if provider_name == "groq":
        provider = LiteLLMProvider(
            provider="groq",
            api_key=settings.groq_api_key,
        )
        return ModelService(provider=provider)

    raise HTTPException(
        status_code=400,
        detail="Unsupported provider. Choose 'gemini' or 'groq'.",
    )


app = FastAPI(
    title="Cost-Aware LLM Router",
    description="A system for routing queries to cost-effective language models.",
    version="0.1.0",
)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest) -> LLMResponse:
    service = get_model_service(request.provider)

    if request.provider == "gemini":
        model = settings.gemini_model
    else:
        model = settings.groq_model

    try:
        return await service.generate(
            prompt=request.prompt,
            model=model,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The model provider request failed.",
        ) from exc