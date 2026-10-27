from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRecord,
    BenchmarkEvaluationRequest,
    BenchmarkEvaluationRun,
)
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)


class BenchmarkEvaluationRunService:
    def __init__(
        self,
        evaluation_service: BenchmarkEvaluationService | None = None,
    ) -> None:
        self.evaluation_service = (
            evaluation_service or BenchmarkEvaluationService()
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

        records = []
        evaluated_results = 0
        failed_generations = 0
        evaluation_errors = 0

        for result in request.results:
            if not result.success:
                failed_generations += 1

                records.append(
                    BenchmarkEvaluationRecord(
                        generation_result=result,
                    )
                )
                continue

            query = queries_by_id[result.query_id]

            try:
                evaluation = await self.evaluation_service.evaluate(
                    query=query,
                    result=result,
                )

                evaluated_results += 1

                records.append(
                    BenchmarkEvaluationRecord(
                        generation_result=result,
                        evaluation=evaluation,
                    )
                )

            except Exception:
                evaluation_errors += 1

                records.append(
                    BenchmarkEvaluationRecord(
                        generation_result=result,
                        evaluation_error=(
                            "Evaluation failed. Check the evaluator "
                            "configuration and logs for details."
                        ),
                    )
                )

        return BenchmarkEvaluationRun(
            total_results=len(request.results),
            evaluated_results=evaluated_results,
            failed_generations=failed_generations,
            evaluation_errors=evaluation_errors,
            records=records,
        )