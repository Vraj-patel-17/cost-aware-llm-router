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
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_service import BenchmarkService
from app.services.benchmark_runner import BenchmarkRunner
from app.schemas.benchmark_analysis import BenchmarkAnalysis
from app.services.benchmark_analysis_service import BenchmarkAnalysisService
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_evaluation import BenchmarkEvaluation
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)
from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRequest,
    BenchmarkEvaluationRun,
)
from app.services.benchmark_evaluation_run_service import (
    BenchmarkEvaluationRunService,
)
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

@app.post("/benchmark/run", response_model=list[BenchmarkResult])
async def run_benchmark(
    model_names: list[str],
) -> list[BenchmarkResult]:
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

    queries = BenchmarkService().load_queries()
    runner = BenchmarkRunner(model_service_factory=get_model_service)

    return await runner.run_benchmark(
        queries=queries,
        model_names=model_names,
    )
@app.post("/benchmark/analyze", response_model=BenchmarkAnalysis)
async def analyze_benchmark(
    results: list[BenchmarkResult],
) -> BenchmarkAnalysis:
    service = BenchmarkAnalysisService()
    return service.analyze(results)
@app.post(
    "/benchmark/evaluate",
    response_model=BenchmarkEvaluation,
)
async def evaluate_benchmark(
    query: BenchmarkQuery,
    result: BenchmarkResult,
) -> BenchmarkEvaluation:
    service = BenchmarkEvaluationService()

    try:
        return await service.evaluate(
            query=query,
            result=result,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The evaluation provider request failed.",
        ) from exc
@app.post(
    "/benchmark/evaluate-all",
    response_model=BenchmarkEvaluationRun,
)
async def evaluate_all_benchmarks(
    request: BenchmarkEvaluationRequest,
) -> BenchmarkEvaluationRun:
    service = BenchmarkEvaluationRunService()

    try:
        return await service.evaluate_all(request)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The benchmark evaluation run failed.",
        ) from exc