import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import (
    get_db,
    get_model_service,
    validate_benchmark_models,
)
from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRequest,
    BenchmarkEvaluationRun,
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
from app.services.benchmark_evaluation_run_service import (
    BenchmarkEvaluationRunService,
)
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)
from app.services.benchmark_run_service import BenchmarkRunService
from app.services.benchmark_runner import BenchmarkRunner
from app.services.benchmark_service import BenchmarkService
from app.services.benchmark_usage import (
    build_generation_usage,
    build_judge_usage,
    record_usage_safely,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark/runs", tags=["benchmark runs"])


@router.post("", response_model=BenchmarkRunSummary, status_code=201)
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

    record_usage_safely(db, build_generation_usage(results))

    return run_service.build_summary(run)


@router.get("", response_model=list[BenchmarkRunSummary])
def list_benchmark_runs(
    db: Session = Depends(get_db),
) -> list[BenchmarkRunSummary]:
    run_service = BenchmarkRunService(db)
    return run_service.build_summaries(run_service.list_runs())


@router.get("/{run_id}", response_model=BenchmarkRunDetail)
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


@router.post("/{run_id}/evaluate", response_model=BenchmarkEvaluationRun)
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

    record_usage_safely(db, build_judge_usage(evaluations))

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


@router.get("/{run_id}/comparison", response_model=BenchmarkRunComparison)
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