import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException , Depends
from sqlalchemy.orm import Session
from app.core.config import settings
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
from app.schemas.benchmark_run import (
    BenchmarkRunComparison,
    BenchmarkRunCreate,
    BenchmarkRunDetail,
    BenchmarkRunSummary,
)
from app.services.benchmark_comparison_service import (
    BenchmarkComparisonService,
)
from app.services.benchmark_run_service import BenchmarkRunService

logger = logging.getLogger(__name__)


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


def _provider_for(model: str) -> str:
    config = MODEL_REGISTRY.get(model)
    provider = config.get("provider") if config else None

    return provider or model.split("/", 1)[0]


def _generation_usage(results: list[BenchmarkResult]) -> list[LLMResponse]:
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


def _judge_usage(
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


def _safe_log_usage(db: Session, responses: list[LLMResponse]) -> None:
    try:
        UsageService(db).log_many(responses)
    except Exception:
        db.rollback()
        logger.exception("Failed to record benchmark usage.")


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
    validate_benchmark_models(model_names)

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
        logger.exception("Benchmark evaluation request failed.")
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
        logger.exception("Benchmark evaluation run failed.")
        raise HTTPException(
            status_code=502,
            detail="The benchmark evaluation run failed.",
        ) from exc


@app.post(
    "/benchmark/runs",
    response_model=BenchmarkRunSummary,
    status_code=201,
)
async def create_benchmark_run(
    request: BenchmarkRunCreate,
    db: Session = Depends(get_db),
) -> BenchmarkRunSummary:
    validate_benchmark_models(request.model_names)

    queries = BenchmarkService().load_queries()
    runner = BenchmarkRunner(model_service_factory=get_model_service)

    results = await runner.run_benchmark(
        queries=queries,
        model_names=request.model_names,
    )

    run_service = BenchmarkRunService(db)

    try:
        run = run_service.create_run(
            model_names=request.model_names,
            dataset_hash=BenchmarkService.dataset_hash(queries),
            results=results,
        )
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to save benchmark run.")
        raise HTTPException(
            status_code=500,
            detail="Failed to save the benchmark run.",
        ) from exc

    _safe_log_usage(db, _generation_usage(results))

    return run_service.build_summary(run)


@app.get(
    "/benchmark/runs",
    response_model=list[BenchmarkRunSummary],
)
def list_benchmark_runs(
    db: Session = Depends(get_db),
) -> list[BenchmarkRunSummary]:
    run_service = BenchmarkRunService(db)
    return run_service.build_summaries(run_service.list_runs())


@app.get(
    "/benchmark/runs/{run_id}",
    response_model=BenchmarkRunDetail,
)
def get_benchmark_run(
    run_id: str,
    db: Session = Depends(get_db),
) -> BenchmarkRunDetail:
    run_service = BenchmarkRunService(db)
    run = run_service.get_run(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark run not found.",
        )

    summary = run_service.build_summary(run)

    return BenchmarkRunDetail(
        **summary.model_dump(),
        results=run_service.get_results(run_id),
    )


@app.post(
    "/benchmark/runs/{run_id}/evaluate",
    response_model=BenchmarkEvaluationRun,
)
async def evaluate_benchmark_run(
    run_id: str,
    force: bool = False,
    db: Session = Depends(get_db),
) -> BenchmarkEvaluationRun:
    run_service = BenchmarkRunService(db)
    run = run_service.get_run(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark run not found.",
        )

    queries = BenchmarkService().load_queries()

    if BenchmarkService.dataset_hash(queries) != run.dataset_hash:
        raise HTTPException(
            status_code=409,
            detail=(
                "The benchmark dataset has changed since this run "
                "was created."
            ),
        )

    try:
        evaluation_service = BenchmarkEvaluationService()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    results = run_service.get_results(run_id)

    if not force:
        already_evaluated = run_service.get_evaluated_keys(
            run_id,
            evaluation_service.judge_model,
        )
        results = [
            result for result in results
            if (result.query_id, result.model) not in already_evaluated
        ]

    if not results:
        return BenchmarkEvaluationRun(
            total_results=0,
            evaluated_results=0,
            failed_generations=0,
            evaluation_errors=0,
            records=[],
        )

    try:
        evaluation_run = await BenchmarkEvaluationRunService(
            evaluation_service=evaluation_service,
        ).evaluate_all(
            BenchmarkEvaluationRequest(
                queries=queries,
                results=results,
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Benchmark run evaluation failed.")
        raise HTTPException(
            status_code=502,
            detail="The benchmark evaluation run failed.",
        ) from exc

    evaluations = [
        record.evaluation
        for record in evaluation_run.records
        if record.evaluation is not None
    ]

    _safe_log_usage(db, _judge_usage(evaluations))

    try:
        run_service.save_evaluations(run_id, evaluations)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to save benchmark evaluations.")
        raise HTTPException(
            status_code=500,
            detail="Failed to save benchmark evaluations.",
        ) from exc

    return evaluation_run


@app.get(
    "/benchmark/runs/{run_id}/comparison",
    response_model=BenchmarkRunComparison,
)
def compare_benchmark_run(
    run_id: str,
    judge_model: str | None = None,
    db: Session = Depends(get_db),
) -> BenchmarkRunComparison:
    run_service = BenchmarkRunService(db)

    if run_service.get_run(run_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Benchmark run not found.",
        )

    judge = judge_model or settings.judge_model or settings.gemini_model

    return BenchmarkComparisonService().compare(
        run_id=run_id,
        judge_model=judge,
        results=run_service.get_results(run_id),
        evaluations=run_service.get_evaluations(
            run_id,
            judge_model=judge,
        ),
    )