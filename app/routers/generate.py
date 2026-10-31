import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.dependencies import get_db, get_model_service
from app.schemas.llm import LLMRequest, LLMResponse
from app.services.usage_service import UsageService
logger = logging.getLogger(__name__)
router = APIRouter(tags=["generate"])

@router.post("/generate", response_model=LLMResponse)
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
        logger.exception(
            "Model provider request failed for model '%s'.",
            request.model,
        )
        raise HTTPException(
            status_code=502,
            detail="The model provider request failed.",
        ) from exc

    try:
        usage_service = UsageService(db)
        usage_service.log_usage(response)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to record model usage.")
        raise HTTPException(
            status_code=500,
            detail="Failed to record model usage.",
        ) from exc

    return response