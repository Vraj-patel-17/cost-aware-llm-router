from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException , Depends
from sqlalchemy.orm import Session
from app.core.dependencies import get_db
from app.core.database import Base, engine
from app.core.model_registry import MODEL_REGISTRY
from app.models.usage_log import UsageLog
from app.schemas.analytics import AnalyticsSummary
from app.services.analytics_service import AnalyticsService
from app.providers.litellm_provider import LiteLLMProvider
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.model_service import ModelService
from app.services.usage_service import UsageService
from app.schemas.model_analytics import ModelAnalytics

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Cost-Aware LLM Router",
    description="A system for routing queries to cost-effective language models.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model_service(model_name: str) -> ModelService:
    model_config = MODEL_REGISTRY.get(model_name)

    if model_config is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported model.",
        )

    provider = LiteLLMProvider(
        provider=model_config["provider"],
        api_key=model_config["api_key"],
    )

    return ModelService(provider=provider)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=LLMResponse)
async def generate(
    request: LLMRequest,
    db: Session = Depends(get_db),
) -> LLMResponse:
    service = get_model_service(request.model)

    try:
        response = await service.generate(
            prompt=request.prompt,
            model=request.model,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The model provider request failed.",
        ) from exc

    try:
        usage_service = UsageService(db)
        usage_service.log_usage(response)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to record model usage.",
        ) from exc

    return response

@app.get("/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary(
    db: Session = Depends(get_db),
) -> AnalyticsSummary:
    analytics_service = AnalyticsService(db)
    summary = analytics_service.get_summary()
    return AnalyticsSummary(**summary)

@app.get("/analytics/models", response_model=list[ModelAnalytics])
def get_model_analytics(
    db: Session = Depends(get_db),
) -> list[ModelAnalytics]:
    analytics_service = AnalyticsService(db)
    results = analytics_service.get_model_summary()

    return [
        ModelAnalytics(**result)
        for result in results
    ]