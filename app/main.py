from fastapi import FastAPI , HTTPException
from app.core.config import settings
from app.core.model_registry import MODEL_REGISTRY
from app.providers.litellm_provider import LiteLLMProvider
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.model_service import ModelService

app = FastAPI(
    title="Cost-Aware LLM Router",
    description="A system for routing queries to cost-effective language models.",
    version="0.1.0",
)
def get_model_service(model_name: str) -> ModelService:
    model_config=MODEL_REGISTRY.get(model_name)
    if model_config is None:
        raise HTTPException(status_code=400,detail="Unsupported model.")
    provider=LiteLLMProvider(provider=model_config["provider"],api_key=model_config["api_key"])
    return ModelService(provider=provider)

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest) -> LLMResponse:
    service = get_model_service(request.model)

    try:
        return await service.generate(
            prompt=request.prompt,
            model=request.model,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The model provider request failed.",
        ) from exc