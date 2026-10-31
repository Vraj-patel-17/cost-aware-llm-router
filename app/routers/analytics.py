from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.schemas.analytics import AnalyticsSummary
from app.schemas.model_analytics import ModelAnalytics
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
def get_analytics_summary(
    db: Session = Depends(get_db),
) -> AnalyticsSummary:
    analytics_service = AnalyticsService(db)
    summary = analytics_service.get_summary()
    return AnalyticsSummary(**summary)


@router.get("/models", response_model=list[ModelAnalytics])
def get_model_analytics(
    db: Session = Depends(get_db),
) -> list[ModelAnalytics]:
    analytics_service = AnalyticsService(db)
    results = analytics_service.get_model_summary()

    return [
        ModelAnalytics(**result)
        for result in results
    ]