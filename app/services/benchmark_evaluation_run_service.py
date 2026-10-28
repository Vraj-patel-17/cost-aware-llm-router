import asyncio
import logging

from app.core.config import settings
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRecord,
    BenchmarkEvaluationRequest,
    BenchmarkEvaluationRun,
)
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)

logger = logging.getLogger(__name__)


class BenchmarkEvaluationRunService:
    def __init__(
        self,
        evaluation_service: BenchmarkEvaluationService | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.evaluation_service = (
            evaluation_service or BenchmarkEvaluationService()
        )
        self.max_concurrency = (
            max_concurrency or settings.benchmark_concurrency
        )

    async def _evaluate_one(
        self,
        query: BenchmarkQuery,
        result: BenchmarkResult,
        semaphore: asyncio.Semaphore,
    ) -> BenchmarkEvaluationRecord:
        async with semaphore:
            try:
                evaluation = await self.evaluation_service.evaluate(
                    query=query,
                    result=result,
                )
            except Exception:
                logger.exception(
                    "Evaluation failed for query '%s' and model '%s'.",
                    result.query_id,
                    result.model,
                )
                return BenchmarkEvaluationRecord(
                    generation_result=result,
                    evaluation_error=(
                        "Evaluation failed. Check the evaluator "
                        "configuration and logs for details."
                    ),
                )

        return BenchmarkEvaluationRecord(
            generation_result=result,
            evaluation=evaluation,
        )

    async def evaluate_all(
        self,
        request: BenchmarkEvaluationRequest,
    ) -> BenchmarkEvaluationRun:
        queries_by_id = {}

        for query in request.queries:
            if query.id in queries_by_id:
                raise ValueError(
                    f"Duplicate benchmark query ID: '{query.id}'."
                )

            queries_by_id[query.id] = query

        seen_results = set()

        for result in request.results:
            result_key = (result.query_id, result.model)

            if result_key in seen_results:
                raise ValueError(
                    "Duplicate benchmark result for "
                    f"query '{result.query_id}' and model '{result.model}'."
                )

            seen_results.add(result_key)

            if result.query_id not in queries_by_id:
                raise ValueError(
                    f"No query found for result query ID '{result.query_id}'."
                )

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def build_record(
            result: BenchmarkResult,
        ) -> BenchmarkEvaluationRecord:
            if not result.success:
                return BenchmarkEvaluationRecord(generation_result=result)

            return await self._evaluate_one(
                query=queries_by_id[result.query_id],
                result=result,
                semaphore=semaphore,
            )

        # gather() preserves input order, so records line up with results.
        records = list(
            await asyncio.gather(
                *(build_record(result) for result in request.results)
            )
        )

        return BenchmarkEvaluationRun(
            total_results=len(request.results),
            evaluated_results=sum(
                record.evaluation is not None for record in records
            ),
            failed_generations=sum(
                not result.success for result in request.results
            ),
            evaluation_errors=sum(
                record.evaluation_error is not None for record in records
            ),
            records=records,
        )