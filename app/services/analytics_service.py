from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.usage_log import UsageLog

class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db

    def get_summary(self) -> dict:
        statement = select(
            func.count(UsageLog.id).label("total_requests"),
            func.coalesce(
                func.sum(UsageLog.input_tokens), 0
            ).label("total_input_tokens"),
            func.coalesce(
                func.sum(UsageLog.output_tokens), 0
            ).label("total_output_tokens"),
            func.sum(
                UsageLog.estimated_cost_usd
            ).label("total_estimated_cost_usd"),
            func.avg(UsageLog.latency_ms).label("average_latency_ms"),
        )

        result = self.db.execute(statement).one()

        return {
            "total_requests": result.total_requests,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_estimated_cost_usd": result.total_estimated_cost_usd,
            "average_latency_ms": result.average_latency_ms,
        }