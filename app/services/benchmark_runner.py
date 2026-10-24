from collections.abc import Callable

from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_result import BenchmarkResult
from app.services.model_service import ModelService


class BenchmarkRunner:
    def __init__(
        self,
        model_service_factory: Callable[[str], ModelService],
    ):
        self.model_service_factory = model_service_factory

    async def run_query(
        self,
        query: BenchmarkQuery,
        model_name: str,
    ) -> BenchmarkResult:
        try:
            model_service = self.model_service_factory(model_name)

            response = await model_service.generate(
                prompt=query.prompt,
                model=model_name,
            )

            return BenchmarkResult(
                query_id=query.id,
                category=query.category,
                model=model_name,
                response_text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                estimated_cost_usd=response.estimated_cost_usd,
                success=True,
            )

        except Exception as exc:
            return BenchmarkResult(
                query_id=query.id,
                category=query.category,
                model=model_name,
                response_text="",
                success=False,
                error=str(exc),
            )
    async def run_benchmark(
        self,
        queries: list[BenchmarkQuery],
        model_names: list[str],
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []

        for model_name in model_names:
            for query in queries:
                result = await self.run_query(
                    query=query,
                    model_name=model_name,
                )
                results.append(result)

        return results