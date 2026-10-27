import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_evaluation import (
    BenchmarkEvaluation,
    CriterionEvaluation,
)
from app.schemas.benchmark_evaluation_run import (
    BenchmarkEvaluationRequest,
)
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_evaluation_run_service import (
    BenchmarkEvaluationRunService,
)


@pytest.fixture
def query():
    return BenchmarkQuery(
        id="q1",
        category="simple",
        prompt="What is 2 + 2?",
        evaluation_criteria=["The answer is 4."],
    )


@pytest.fixture
def successful_result():
    return BenchmarkResult(
        query_id="q1",
        category="simple",
        model="gemini/gemini-3.5-flash-lite",
        response_text="4",
        input_tokens=10,
        output_tokens=2,
        latency_ms=100.0,
        estimated_cost_usd=0.00001,
        success=True,
    )


@pytest.fixture
def evaluation():
    return BenchmarkEvaluation(
        query_id="q1",
        model="gemini/gemini-3.5-flash-lite",
        score=5,
        passed=True,
        reasoning="The response correctly answers the question.",
        criteria_evaluations=[
            CriterionEvaluation(
                criterion="The answer is 4.",
                passed=True,
                explanation="The response gives the correct answer.",
            )
        ],
    )


@pytest.fixture
def evaluator():
    return SimpleNamespace(evaluate=AsyncMock())


@pytest.fixture
def service(evaluator):
    return BenchmarkEvaluationRunService(
        evaluation_service=evaluator,
    )


@pytest.mark.asyncio
async def test_evaluate_all_successful_result(
    service,
    evaluator,
    query,
    successful_result,
    evaluation,
):
    evaluator.evaluate.return_value = evaluation

    request = BenchmarkEvaluationRequest(
        queries=[query],
        results=[successful_result],
    )

    response = await service.evaluate_all(request)

    assert response.total_results == 1
    assert response.evaluated_results == 1
    assert response.failed_generations == 0
    assert response.evaluation_errors == 0
    assert response.records[0].evaluation == evaluation
    assert response.records[0].evaluation_error is None

    evaluator.evaluate.assert_awaited_once_with(
        query=query,
        result=successful_result,
    )


@pytest.mark.asyncio
async def test_failed_generation_is_not_evaluated(
    service,
    evaluator,
    query,
):
    failed_result = BenchmarkResult(
        query_id="q1",
        category="simple",
        model="groq/openai/gpt-oss-120b",
        response_text="",
        success=False,
        error="Provider request failed.",
    )

    request = BenchmarkEvaluationRequest(
        queries=[query],
        results=[failed_result],
    )

    response = await service.evaluate_all(request)

    assert response.total_results == 1
    assert response.evaluated_results == 0
    assert response.failed_generations == 1
    assert response.evaluation_errors == 0
    assert response.records[0].generation_result == failed_result
    assert response.records[0].evaluation is None

    evaluator.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluation_error_does_not_stop_run(
    service,
    evaluator,
    query,
    successful_result,
):
    evaluator.evaluate.side_effect = RuntimeError(
        "Evaluator provider unavailable."
    )

    request = BenchmarkEvaluationRequest(
        queries=[query],
        results=[successful_result],
    )

    response = await service.evaluate_all(request)

    assert response.total_results == 1
    assert response.evaluated_results == 0
    assert response.evaluation_errors == 1
    assert response.records[0].evaluation is None
    assert response.records[0].evaluation_error is not None


@pytest.mark.asyncio
async def test_duplicate_query_ids_are_rejected(
    service,
    query,
    successful_result,
):
    duplicate_query = query.model_copy()

    request = BenchmarkEvaluationRequest(
        queries=[query, duplicate_query],
        results=[successful_result],
    )

    with pytest.raises(ValueError, match="Duplicate benchmark query ID"):
        await service.evaluate_all(request)


@pytest.mark.asyncio
async def test_duplicate_result_pairs_are_rejected(
    service,
    query,
    successful_result,
):
    request = BenchmarkEvaluationRequest(
        queries=[query],
        results=[successful_result, successful_result],
    )

    with pytest.raises(ValueError, match="Duplicate benchmark result"):
        await service.evaluate_all(request)


@pytest.mark.asyncio
async def test_missing_query_reference_is_rejected(
    service,
    successful_result,
):
    request = BenchmarkEvaluationRequest(
        queries=[
            BenchmarkQuery(
                id="q2",
                category="simple",
                prompt="What is 2 + 2?",
                evaluation_criteria=["The answer is 4."],
            )
        ],
        results=[successful_result],
    )

    with pytest.raises(ValueError, match="No query found"):
        await service.evaluate_all(request)


@pytest.mark.asyncio
async def test_missing_query_id_is_rejected_by_service(
    service,
    successful_result,
):
    another_result = successful_result.model_copy(
        update={"query_id": "unknown"}
    )

    request = BenchmarkEvaluationRequest(
        queries=[
            BenchmarkQuery(
                id="q1",
                category="simple",
                prompt="What is 2 + 2?",
                evaluation_criteria=["The answer is 4."],
            )
        ],
        results=[another_result],
    )

    with pytest.raises(ValueError, match="No query found"):
        await service.evaluate_all(request)