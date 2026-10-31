from collections.abc import Generator

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.model_registry import MODEL_REGISTRY
from app.providers.litellm_provider import LiteLLMProvider
from app.services.model_service import ModelService


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


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


def validate_benchmark_models(model_names: list[str]) -> None:
    unsupported_models = [
        model for model in model_names
        if model not in MODEL_REGISTRY
    ]

    if unsupported_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported models: {unsupported_models}",
        )

    if not model_names:
        raise HTTPException(
            status_code=400,
            detail="At least one model must be provided.",
        )

    if len(model_names) != len(set(model_names)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate models are not allowed.",
        )