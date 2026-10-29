from collections import defaultdict

from app.schemas.benchmark_evaluation import BenchmarkEvaluation
from app.schemas.benchmark_result import BenchmarkResult
from app.schemas.benchmark_run import BenchmarkRunComparison,ModelRunSummary

from app.services.benchmark_analysis_service import BenchmarkAnalysisService


class BenchmarkComparisonService:
    def compare(
        self,
        run_id: str,
        judge_model: str,
        results: list[BenchmarkResult],
        evaluations: list[BenchmarkEvaluation],
    ) -> BenchmarkRunComparison:
        evaluations_by_key = {
            (evaluation.query_id, evaluation.model): evaluation
            for evaluation in evaluations
        }

        results_by_model: dict[str, list[BenchmarkResult]] = defaultdict(
            list
        )
        for result in results:
            results_by_model[result.model].append(result)

        analysis_service = BenchmarkAnalysisService()
        summaries = []

        for model, model_results in results_by_model.items():
            analysis = analysis_service.analyze(model_results)

            model_evaluations = [
                evaluations_by_key[(result.query_id, model)]
                for result in model_results
                if (result.query_id, model) in evaluations_by_key
            ]

            passed = sum(
                1 for evaluation in model_evaluations if evaluation.passed
            )
            scores = [evaluation.score for evaluation in model_evaluations]
            total_cost = analysis.total_estimated_cost_usd

            summaries.append(
                ModelRunSummary(
                    model=model,
                    total_queries=analysis.total_queries,
                    successful_queries=analysis.successful_queries,
                    success_rate=analysis.success_rate,
                    average_latency_ms=analysis.average_latency_ms,
                    total_generation_cost_usd=total_cost,
                    evaluated_queries=len(model_evaluations),
                    passed_queries=passed,
                    pass_rate=(
                        passed / len(model_evaluations) * 100
                        if model_evaluations
                        else None
                    ),
                    average_score=(
                        sum(scores) / len(scores) if scores else None
                    ),
                    cost_per_passed_query_usd=(
                        total_cost / passed
                        if total_cost is not None and passed > 0
                        else None
                    ),
                    self_judged=(model == judge_model),
                )
            )

        judge_costs = [
            evaluation.judge_cost_usd for evaluation in evaluations
        ]

        return BenchmarkRunComparison(
            run_id=run_id,
            judge_model=judge_model,
            judge_cost_usd=(
                sum(judge_costs)
                if judge_costs
                and all(cost is not None for cost in judge_costs)
                else None
            ),
            models=summaries,
        )