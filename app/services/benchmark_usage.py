import logging

from sqlalchemy.orm import Session

from app.core.model_registry import MODEL_REGISTRY
from app.schemas.benchmark_evaluation import BenchmarkEvaluation
from app.schemas.benchmark_result import BenchmarkResult
from app.schemas.llm import LLMResponse
from app.services.usage_service import UsageService

logger = logging.getLogger(__name__)


def _provider_for(model: str) -> str:
    config = MODEL_REGISTRY.get(model)
    provider = config.get("provider") if config else None

    return provider or model.split("/", 1)[0]


def build_generation_usage(
    results: list[BenchmarkResult],
) -> list[LLMResponse]:
    return [
        LLMResponse(
            text="",
            model=result.model,
            provider=_provider_for(result.model),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
        )
        for result in results
        if result.success
    ]


def build_judge_usage(
    evaluations: list[BenchmarkEvaluation],
) -> list[LLMResponse]:
    return [
        LLMResponse(
            text="",
            model=evaluation.judge_model,
            provider=_provider_for(evaluation.judge_model),
            input_tokens=evaluation.judge_input_tokens,
            output_tokens=evaluation.judge_output_tokens,
            estimated_cost_usd=evaluation.judge_cost_usd,
        )
        for evaluation in evaluations
        if evaluation.judge_model
    ]


def record_usage_safely(db: Session, responses: list[LLMResponse]) -> None:
    try:
        UsageService(db).log_many(responses)
    except Exception:
        db.rollback()
        logger.exception("Failed to record benchmark usage.")