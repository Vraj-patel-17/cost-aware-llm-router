from sqlalchemy.orm import Session

from app.models.usage_log import UsageLog
from app.schemas.llm import LLMResponse


class UsageService:
    def __init__(self, db: Session):
        self.db = db

    def log_usage(self, response: LLMResponse) -> UsageLog:
        usage = UsageLog(
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            estimated_cost_usd=response.estimated_cost_usd,
        )

        self.db.add(usage)
        self.db.commit()
        self.db.refresh(usage)

        return usage