import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)


@pytest.fixture
def query() -> BenchmarkQuery:
    return BenchmarkQuery(
        id="simple-001",
        category="simple",
        prompt="What is 2 + 2?",
        evaluation_criteria=[
            "States that the answer is 4.",
            "Provides a clear answer.",
        ],
    )


@pytest.fixture
def result() -> BenchmarkResult:
    return BenchmarkResult(
        query_id="simple-001",
        category="simple",
        model="groq/openai/gpt-oss-120b",
        response_text="2 + 2 = 4.",
        input_tokens=10,
        output_tokens=8,
        latency_ms=150.0,
        estimated_cost_usd=0.00001,
        success=True,
    )


@pytest.fixture
def valid_evaluation() -> dict:
    return {
        "query_id": "simple-001",
        "model": "groq/openai/gpt-oss-120b",
        "score": 5,
        "passed": True,
        "reasoning": "The answer is correct and satisfies both criteria.",
        "criteria_evaluations": [
            {
                "criterion": "States that the answer is 4.",
                "passed": True,
                "explanation": "The response explicitly gives 4.",
            },
            {
                "criterion": "Provides a clear answer.",
                "passed": True,
                "explanation": "The answer is concise and clear.",
            },
        ],
    }


@pytest.fixture
def evaluator_provider() -> SimpleNamespace:
    return SimpleNamespace(generate=AsyncMock())


@pytest.mark.asyncio
async def test_evaluate_valid_response(
    query,
    result,
    valid_evaluation,
    evaluator_provider,
):
    evaluator_provider.generate.return_value = SimpleNamespace(
        text=json.dumps(valid_evaluation)
    )

    service = BenchmarkEvaluationService(
        evaluator_provider=evaluator_provider
    )

    evaluation = await service.evaluate(query, result)

    assert evaluation.query_id == query.id
    assert evaluation.model == result.model
    assert evaluation.score == 5
    assert evaluation.passed is True
    assert len(evaluation.criteria_evaluations) == 2

    evaluator_provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_rejects_invalid_json(
    query,
    result,
    evaluator_provider,
):
    evaluator_provider.generate.return_value = SimpleNamespace(
        text="This is not JSON"
    )

    service = BenchmarkEvaluationService(
        evaluator_provider=evaluator_provider
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        await service.evaluate(query, result)


@pytest.mark.asyncio
async def test_evaluate_rejects_mismatched_query_id(
    query,
    result,
    evaluator_provider,
):
    result.query_id = "different-query"

    service = BenchmarkEvaluationService(
        evaluator_provider=evaluator_provider
    )

    with pytest.raises(ValueError, match="Query ID"):
        await service.evaluate(query, result)

    evaluator_provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_rejects_failed_generation(
    query,
    result,
    evaluator_provider,
):
    result.success = False

    service = BenchmarkEvaluationService(
        evaluator_provider=evaluator_provider
    )

    with pytest.raises(ValueError, match="failed benchmark generation"):
        await service.evaluate(query, result)

    evaluator_provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_rejects_missing_criterion(
    query,
    result,
    valid_evaluation,
    evaluator_provider,
):
    valid_evaluation["criteria_evaluations"].pop()

    evaluator_provider.generate.return_value = SimpleNamespace(
        text=json.dumps(valid_evaluation)
    )

    service = BenchmarkEvaluationService(
        evaluator_provider=evaluator_provider
    )

    with pytest.raises(ValueError, match="every supplied criterion"):
        await service.evaluate(query, result)


@pytest.mark.asyncio
async def test_evaluate_rejects_inconsistent_pass_result(
    query,
    result,
    valid_evaluation,
    evaluator_provider,
):
    valid_evaluation["score"] = 2
    valid_evaluation["passed"] = True

    evaluator_provider.generate.return_value = SimpleNamespace(
        text=json.dumps(valid_evaluation)
    )

    service = BenchmarkEvaluationService(
        evaluator_provider=evaluator_provider
    )

    with pytest.raises(ValueError, match="inconsistent pass/fail"):
        await service.evaluate(query, result)