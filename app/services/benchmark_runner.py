import asyncio
from collections.abc import Callable

from app.core.config import settings
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_result import BenchmarkResult
from app.services.model_service import ModelService


class BenchmarkRunner:
    def __init__(
        self,
        model_service_factory: Callable[[str], ModelService],
        max_concurrency: int | None = None,
    ):
        self.model_service_factory = model_service_factory
        self.max_concurrency = (
            max_concurrency or settings.benchmark_concurrency
        )

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
                # Some exceptions (e.g. TimeoutError) have an empty message.
                error=str(exc) or type(exc).__name__,
            )

    async def run_benchmark(
        self,
        queries: list[BenchmarkQuery],
        model_names: list[str],
    ) -> list[BenchmarkResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_limited(
            query: BenchmarkQuery,
            model_name: str,
        ) -> BenchmarkResult:
            async with semaphore:
                return await self.run_query(
                    query=query,
                    model_name=model_name,
                )

        return list(
            await asyncio.gather(
                *(
                    run_limited(query, model_name)
                    for model_name in model_names
                    for query in queries
                )
            )
        )