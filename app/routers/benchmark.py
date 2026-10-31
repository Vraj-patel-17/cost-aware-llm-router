import logging

from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_model_service, validate_benchmark_models
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_analysis import BenchmarkAnalysis
from app.schemas.benchmark_evaluation import BenchmarkEvaluation
from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRequest,
    BenchmarkEvaluationRun,
)
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_analysis_service import BenchmarkAnalysisService
from app.services.benchmark_evaluation_run_service import (
    BenchmarkEvaluationRunService,
)
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)
from app.services.benchmark_runner import BenchmarkRunner
from app.services.benchmark_service import BenchmarkService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.post("/run", response_model=list[BenchmarkResult])
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


@router.post("/analyze", response_model=BenchmarkAnalysis)
async def analyze_benchmark(
    results: list[BenchmarkResult],
) -> BenchmarkAnalysis:
    service = BenchmarkAnalysisService()
    return service.analyze(results)


@router.post("/evaluate", response_model=BenchmarkEvaluation)
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


@router.post("/evaluate-all", response_model=BenchmarkEvaluationRun)
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