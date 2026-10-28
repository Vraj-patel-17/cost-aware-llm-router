import asyncio
from types import SimpleNamespace

import pytest

from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_evaluation import (
    BenchmarkEvaluation,
    CriterionEvaluation,
)
from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRequest,
)
from app.schemas.benchmark_result import BenchmarkResult
from app.schemas.llm import LLMResponse
from app.services.benchmark_evaluation_run_service import (
    BenchmarkEvaluationRunService,
)
from app.services.benchmark_runner import BenchmarkRunner


def make_queries(count: int) -> list[BenchmarkQuery]:
    return [
        BenchmarkQuery(
            id=f"q{i}",
            category="simple",
            prompt="Prompt",
            evaluation_criteria=["Criterion"],
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_run_benchmark_respects_concurrency_limit():
    active = 0
    peak = 0

    class FakeService:
        async def generate(self, prompt: str, model: str) -> LLMResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return LLMResponse(text="ok", model=model, provider="test")

    runner = BenchmarkRunner(
        model_service_factory=lambda name: FakeService(),
        max_concurrency=2,
    )
    queries = make_queries(6)

    results = await runner.run_benchmark(
        queries=queries,
        model_names=["test/model"],
    )

    assert [r.query_id for r in results] == [q.id for q in queries]
    assert all(r.success for r in results)
    assert 1 < peak <= 2


@pytest.mark.asyncio
async def test_failed_query_without_message_reports_exception_type():
    class FailingService:
        async def generate(self, prompt: str, model: str) -> LLMResponse:
            raise TimeoutError()

    runner = BenchmarkRunner(
        model_service_factory=lambda name: FailingService(),
    )

    result = await runner.run_query(
        query=make_queries(1)[0],
        model_name="test/model",
    )

    assert result.success is False
    assert result.error == "TimeoutError"


@pytest.mark.asyncio
async def test_evaluation_records_preserve_input_order():
    queries = make_queries(5)
    results = [
        BenchmarkResult(
            query_id=query.id,
            category="simple",
            model="test/model",
            response_text="answer",
            success=True,
        )
        for query in queries
    ]

    async def fake_evaluate(query, result):
        # Later queries finish first, to prove ordering is preserved.
        await asyncio.sleep((5 - int(query.id[1:])) * 0.005)
        return BenchmarkEvaluation(
            query_id=query.id,
            model=result.model,
            score=5,
            passed=True,
            reasoning="Fine.",
            criteria_evaluations=[
                CriterionEvaluation(
                    criterion="Criterion",
                    passed=True,
                    explanation="Met.",
                )
            ],
        )

    service = BenchmarkEvaluationRunService(
        evaluation_service=SimpleNamespace(evaluate=fake_evaluate),
        max_concurrency=5,
    )

    run = await service.evaluate_all(
        BenchmarkEvaluationRequest(queries=queries, results=results)
    )

    assert run.evaluated_results == 5
    assert [r.generation_result.query_id for r in run.records] == [
        q.id for q in queries
    ]