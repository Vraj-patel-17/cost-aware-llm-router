from sqlalchemy.orm import Session

from app.models.usage_log import UsageLog
from app.schemas.llm import LLMResponse


class UsageService:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _to_record(response: LLMResponse) -> UsageLog:
        return UsageLog(
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            estimated_cost_usd=response.estimated_cost_usd,
        )

    def log_usage(self, response: LLMResponse) -> UsageLog:
        usage = self._to_record(response)

        self.db.add(usage)
        self.db.commit()
        self.db.refresh(usage)

        return usage

    def log_many(self, responses: list[LLMResponse]) -> None:
        """Record several usage entries in a single transaction."""
        if not responses:
            return

        self.db.add_all(
            [self._to_record(response) for response in responses]
        )
        self.db.commit()