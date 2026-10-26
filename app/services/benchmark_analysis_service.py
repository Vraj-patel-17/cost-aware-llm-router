from collections import defaultdict

from app.schemas.benchmark_analysis import (
    BenchmarkAnalysis,
    CategoryAnalysis,
)
from app.schemas.benchmark_result import BenchmarkResult

class BenchmarkAnalysisService:
    def analyze(
        self,
        results: list[BenchmarkResult],
    ) -> BenchmarkAnalysis:
        if not results:
            return BenchmarkAnalysis(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                success_rate=0.0,
                average_latency_ms=None,
                total_estimated_cost_usd=None,
                average_cost_per_successful_query_usd=None,
                categories={},
            )

        successful_results = [
            result for result in results if result.success
        ]

        measured_latencies = [
            result.latency_ms
            for result in successful_results
            if result.latency_ms is not None
        ]

        available_costs = [
            result.estimated_cost_usd
            for result in successful_results
            if result.estimated_cost_usd is not None
        ]

        category_results = defaultdict(list)

        for result in results:
            category_results[result.category].append(result)

        categories: dict[str, CategoryAnalysis] = {}

        for category, category_items in category_results.items():
            successful_items = [
                item for item in category_items if item.success
            ]

            latencies = [
                item.latency_ms
                for item in successful_items
                if item.latency_ms is not None
            ]

            costs = [
                item.estimated_cost_usd
                for item in successful_items
                if item.estimated_cost_usd is not None
            ]

            categories[category] = CategoryAnalysis(
                total_queries=len(category_items),
                successful_queries=len(successful_items),
                failed_queries=len(category_items) - len(successful_items),
                success_rate=(
                    len(successful_items) / len(category_items) * 100
                ),
                average_latency_ms=(
                    sum(latencies) / len(latencies)
                    if latencies
                    else None
                ),
                total_estimated_cost_usd=(
                    sum(costs)
                    if len(costs) == len(successful_items)
                    else None
                ),
            )

        return BenchmarkAnalysis(
            total_queries=len(results),
            successful_queries=len(successful_results),
            failed_queries=len(results) - len(successful_results),
            success_rate=(
                len(successful_results) / len(results) * 100
            ),
            average_latency_ms=(
                sum(measured_latencies) / len(measured_latencies)
                if measured_latencies
                else None
            ),
            total_estimated_cost_usd=(
                sum(available_costs)
                if len(available_costs) == len(successful_results)
                else None
            ),
            average_cost_per_successful_query_usd=(
                sum(available_costs) / len(successful_results)
                if successful_results
                and len(available_costs) == len(successful_results)
                else None
            ),
            categories=categories,
        )