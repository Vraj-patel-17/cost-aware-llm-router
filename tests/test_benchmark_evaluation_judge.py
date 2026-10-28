import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.core.model_pricing import MODEL_PRICING
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_evaluation_service import (
    BenchmarkEvaluationService,
)
CANDIDATE_MODEL = "groq/openai/gpt-oss-120b"


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
        model=CANDIDATE_MODEL,
        response_text="2 + 2 = 4.",
        success=True,
    )


@pytest.fixture
def judge_payload() -> dict:
    return {
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


def make_provider(**response_fields) -> SimpleNamespace:
    return SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(**response_fields))
    )


def test_prompt_does_not_reveal_candidate_model(query, result):
    prompt = BenchmarkEvaluationService._build_evaluation_prompt(
        query, result
    )

    assert CANDIDATE_MODEL not in prompt
    assert "gpt-oss" not in prompt


@pytest.mark.asyncio
async def test_evaluate_accepts_markdown_fenced_json(
    query, result, judge_payload
):
    provider = make_provider(
        text="```json\n" + json.dumps(judge_payload) + "\n```"
    )
    service = BenchmarkEvaluationService(evaluator_provider=provider)

    evaluation = await service.evaluate(query, result)

    assert evaluation.score == 5
    assert evaluation.passed is True


@pytest.mark.asyncio
async def test_evaluate_overrides_ids_echoed_by_judge(
    query, result, judge_payload
):
    judge_payload["query_id"] = "wrong-query"
    judge_payload["model"] = "wrong/model"
    provider = make_provider(text=json.dumps(judge_payload))
    service = BenchmarkEvaluationService(evaluator_provider=provider)

    evaluation = await service.evaluate(query, result)

    assert evaluation.query_id == query.id
    assert evaluation.model == result.model


@pytest.mark.asyncio
async def test_evaluate_records_judge_model_and_cost(
    monkeypatch, query, result, judge_payload
):
    monkeypatch.setattr(settings, "judge_model", "test/judge")
    monkeypatch.setitem(
        MODEL_PRICING, "test/judge", {"input": 2.0, "output": 8.0}
    )
    provider = make_provider(
        text=json.dumps(judge_payload),
        input_tokens=1000,
        output_tokens=500,
    )
    service = BenchmarkEvaluationService(evaluator_provider=provider)

    evaluation = await service.evaluate(query, result)

    assert provider.generate.await_args.kwargs["model"] == "test/judge"
    assert evaluation.judge_model == "test/judge"
    assert evaluation.judge_input_tokens == 1000
    assert evaluation.judge_output_tokens == 500
    assert evaluation.judge_cost_usd == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_judge_cost_is_none_without_usage(
    query, result, judge_payload
):
    provider = make_provider(text=json.dumps(judge_payload))
    service = BenchmarkEvaluationService(evaluator_provider=provider)

    evaluation = await service.evaluate(query, result)

    assert evaluation.judge_cost_usd is None
    assert evaluation.judge_input_tokens is None