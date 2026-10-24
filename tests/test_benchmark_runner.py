from unittest.mock import AsyncMock

import pytest

from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_result import BenchmarkResult
from app.schemas.llm import LLMResponse
from app.services.benchmark_runner import BenchmarkRunner

@pytest.fixture
def benchmark_query():
    return BenchmarkQuery(
        id="simple_001",
        category="simple",
        prompt="Explain what an API is.",
        evaluation_criteria=["Explains the concept correctly"],
    )


@pytest.fixture
def mock_model_service():
    return AsyncMock()

@pytest.fixture
def model_service_factory(mock_model_service):
    from unittest.mock import Mock

    return Mock(return_value=mock_model_service)

@pytest.mark.asyncio
async def test_run_query_success(benchmark_query, mock_model_service, model_service_factory,):
    mock_model_service.generate.return_value = LLMResponse(
        text="An API allows software systems to communicate.",
        model="gemini/gemini-3.5-flash-lite",
        provider="gemini",
        input_tokens=10,
        output_tokens=8,
        latency_ms=250.0,
        estimated_cost_usd=0.000023,
    )

    runner = BenchmarkRunner(model_service_factory)

    result = await runner.run_query(
        query=benchmark_query,
        model_name="gemini/gemini-3.5-flash-lite",
    )

    assert isinstance(result, BenchmarkResult)
    assert result.success is True
    assert result.query_id == "simple_001"
    assert result.category == "simple"
    assert result.model == "gemini/gemini-3.5-flash-lite"
    assert result.response_text == "An API allows software systems to communicate."
    assert result.input_tokens == 10
    assert result.output_tokens == 8
    assert result.latency_ms == 250.0
    assert result.estimated_cost_usd == 0.000023

    mock_model_service.generate.assert_awaited_once_with(
    prompt=benchmark_query.prompt,
    model="gemini/gemini-3.5-flash-lite",
)


@pytest.mark.asyncio
async def test_run_query_failure(benchmark_query, mock_model_service,model_service_factory):
    mock_model_service.generate.side_effect = RuntimeError("Provider unavailable")

    runner = BenchmarkRunner(model_service_factory)

    result = await runner.run_query(
        query=benchmark_query,
        model_name="gemini/gemini-3.5-flash-lite",
    )

    assert result.success is False
    assert result.query_id == "simple_001"
    assert result.response_text == ""
    assert result.error == "Provider unavailable"
    assert result.input_tokens is None
    assert result.estimated_cost_usd is None

@pytest.mark.asyncio
async def test_run_benchmark_multiple_models(
    benchmark_query,
    mock_model_service,
    model_service_factory,
):
    mock_model_service.generate.return_value = LLMResponse(
        text="Test response",
        model="gemini/gemini-3.5-flash-lite",
        provider="gemini",
        input_tokens=10,
        output_tokens=5,
        latency_ms=200.0,
        estimated_cost_usd=0.00002,
    )

    runner = BenchmarkRunner(model_service_factory)

    model_names = [
        "gemini/gemini-3.5-flash-lite",
        "groq/openai/gpt-oss-120b",
    ]

    results = await runner.run_benchmark(
        queries=[benchmark_query],
        model_names=model_names,
    )

    assert len(results) == 2
    assert all(result.success for result in results)
    assert [result.model for result in results] == model_names
    assert mock_model_service.generate.await_count == 2
    assert model_service_factory.call_count == 2

@pytest.mark.asyncio
async def test_run_benchmark_multiple_queries(
    mock_model_service,
    model_service_factory,
):
    queries = [
        BenchmarkQuery(
            id="simple_001",
            category="simple",
            prompt="What is an API?",
            evaluation_criteria=["Explains the concept correctly"],
        ),
        BenchmarkQuery(
            id="reasoning_001",
            category="reasoning",
            prompt="Calculate 20% of 500.",
            evaluation_criteria=["Returns 100"],
        ),
    ]

    mock_model_service.generate.return_value = LLMResponse(
        text="Test response",
        model="gemini/gemini-3.5-flash-lite",
        provider="gemini",
        input_tokens=10,
        output_tokens=5,
        latency_ms=200.0,
        estimated_cost_usd=0.00002,
    )

    runner = BenchmarkRunner(model_service_factory)

    model_names = [
        "gemini/gemini-3.5-flash-lite",
        "groq/openai/gpt-oss-120b",
    ]

    results = await runner.run_benchmark(
        queries=queries,
        model_names=model_names,
    )

    assert len(results) == 4
    assert all(result.success for result in results)
    assert mock_model_service.generate.await_count == 4
    assert model_service_factory.call_count == 4