from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_result import BenchmarkResult

from app.core.config import settings
from app.main import app
from app.main import get_model_service

client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        (settings.gemini_model, "gemini"),
        (settings.groq_model, "groq"),
    ],
)
def test_generate_success(model, provider):
    mock_response = {
        "text": "A test response.",
        "model": model,
        "provider": provider,
        "input_tokens": 5,
        "output_tokens": 3,
        "latency_ms": 100.0,
        "estimated_cost_usd":None
    }

    with (patch("app.main.get_model_service") as mock_get_service,
    patch("app.main.UsageService.log_usage") as mock_log_usage):
        mock_service = mock_get_service.return_value
        mock_service.generate = AsyncMock(return_value=mock_response)

        response = client.post(
            "/generate",
            json={
                "model": model,
                "prompt": "Explain APIs.",
            },
        )
    assert response.status_code == 200
    assert response.json()["text"] == "A test response."
    assert response.json()["model"] == model
    mock_log_usage.assert_called_once()


def test_generate_unsupported_model():
    response = client.post(
        "/generate",
        json={
            "model": "unknown/model",
            "prompt": "Explain APIs.",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model."


def test_generate_rejects_empty_prompt():
    response = client.post(
        "/generate",
        json={
            "model": settings.gemini_model,
            "prompt": "",
        },
    )

    assert response.status_code == 422


def test_generate_handles_provider_failure():
    with patch("app.main.get_model_service") as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.generate = AsyncMock(
            side_effect=RuntimeError("Internal provider error")
        )

        response = client.post(
            "/generate",
            json={
                "model": settings.groq_model,
                "prompt": "Explain APIs.",
            },
        )
    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The model provider request failed."
    )
@pytest.mark.parametrize(
    ("model_name", "expected_provider", "expected_api_key"),
    [
        (settings.gemini_model, "gemini", settings.gemini_api_key),
        (settings.groq_model, "groq", settings.groq_api_key),
    ],
)
def test_get_model_service_uses_registry(
    model_name,
    expected_provider,
    expected_api_key,
):
    with patch("app.main.LiteLLMProvider") as mock_provider:
        service = get_model_service(model_name)

    mock_provider.assert_called_once_with(
        provider=expected_provider,
        api_key=expected_api_key,
    )

    assert service.provider is mock_provider.return_value

def test_analytics_summary_success():
    mock_summary = {
        "total_requests": 10,
        "total_input_tokens": 1500,
        "total_output_tokens": 800,
        "total_estimated_cost_usd": 0.025,
        "average_latency_ms": 250.0,
    }

    with patch(
        "app.main.AnalyticsService.get_summary",
        return_value=mock_summary,
    ):
        response = client.get("/analytics/summary")

    assert response.status_code == 200
    assert response.json() == mock_summary


def test_analytics_summary_empty_database():
    mock_summary = {
        "total_requests": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_estimated_cost_usd": None,
        "average_latency_ms": None,
    }

    with patch(
        "app.main.AnalyticsService.get_summary",
        return_value=mock_summary,
    ):
        response = client.get("/analytics/summary")

    assert response.status_code == 200
    assert response.json() == mock_summary

def test_model_analytics_success():
    mock_results = [
        {
            "model": "gemini/test-model",
            "provider": "gemini",
            "total_requests": 5,
            "total_input_tokens": 500,
            "total_output_tokens": 250,
            "total_estimated_cost_usd": 0.005,
            "average_latency_ms": 200.0,
        },
        {
            "model": "groq/test-model",
            "provider": "groq",
            "total_requests": 3,
            "total_input_tokens": 300,
            "total_output_tokens": 150,
            "total_estimated_cost_usd": None,
            "average_latency_ms": 150.0,
        },
    ]

    with patch(
        "app.main.AnalyticsService.get_model_summary",
        return_value=mock_results,
    ):
        response = client.get("/analytics/models")

    assert response.status_code == 200
    assert response.json() == mock_results


def test_model_analytics_empty_database():
    with patch(
        "app.main.AnalyticsService.get_model_summary",
        return_value=[],
    ):
        response = client.get("/analytics/models")

    assert response.status_code == 200
    assert response.json() == []

def test_run_benchmark_success(monkeypatch):
    queries = [
        BenchmarkQuery(
            id="simple_001",
            category="simple",
            prompt="What is an API?",
            evaluation_criteria=["Explains the concept correctly"],
        )
    ]

    results = [
        BenchmarkResult(
            query_id="simple_001",
            category="simple",
            model="gemini/gemini-3.5-flash-lite",
            response_text="An API enables software communication.",
            success=True,
        )
    ]

    monkeypatch.setattr(
        "app.main.BenchmarkService.load_queries",
        lambda self: queries,
    )

    mock_runner = MagicMock()
    mock_runner.run_benchmark = AsyncMock(return_value=results)

    monkeypatch.setattr(
        "app.main.BenchmarkRunner",
        lambda model_service_factory: mock_runner,
    )

    response = client.post(
        "/benchmark/run",
        json=["gemini/gemini-3.5-flash-lite"],
    )

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["query_id"] == "simple_001"
    mock_runner.run_benchmark.assert_awaited_once()


def test_run_benchmark_rejects_unsupported_model():
    response = client.post(
        "/benchmark/run",
        json=["unsupported/model"],
    )

    assert response.status_code == 400
    assert "Unsupported models" in response.json()["detail"]


def test_run_benchmark_rejects_duplicate_models():
    response = client.post(
        "/benchmark/run",
        json=[
            "gemini/gemini-3.5-flash-lite",
            "gemini/gemini-3.5-flash-lite",
        ],
    )

    assert response.status_code == 400
    assert "Duplicate models" in response.json()["detail"]
def test_benchmark_analyze_success():
    payload = [
        {
            "query_id": "q1",
            "category": "simple",
            "model": "gemini/test",
            "response_text": "Answer",
            "input_tokens": 10,
            "output_tokens": 20,
            "latency_ms": 100.0,
            "estimated_cost_usd": 0.001,
            "success": True,
        }
    ]

    response = client.post("/benchmark/analyze", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["total_queries"] == 1
    assert data["successful_queries"] == 1
    assert data["failed_queries"] == 0
    assert data["success_rate"] == 100.0
    assert data["average_latency_ms"] == 100.0
    assert data["total_estimated_cost_usd"] == 0.001
    assert data["categories"]["simple"]["total_queries"] == 1


def test_benchmark_analyze_empty_input():
    response = client.post("/benchmark/analyze", json=[])

    assert response.status_code == 200

    data = response.json()
    assert data["total_queries"] == 0
    assert data["successful_queries"] == 0
    assert data["failed_queries"] == 0
    assert data["success_rate"] == 0.0
    assert data["average_latency_ms"] is None
    assert data["categories"] == {}