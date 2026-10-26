import pytest

from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_analysis_service import BenchmarkAnalysisService


@pytest.fixture
def service():
    return BenchmarkAnalysisService()

def test_analyze_successful_results(service):
    results = [
        BenchmarkResult(
            query_id="q1",
            category="simple",
            model="gemini/test",
            response_text="Answer 1",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100.0,
            estimated_cost_usd=0.001,
            success=True,
        ),
        BenchmarkResult(
            query_id="q2",
            category="simple",
            model="gemini/test",
            response_text="Answer 2",
            input_tokens=15,
            output_tokens=25,
            latency_ms=200.0,
            estimated_cost_usd=0.002,
            success=True,
        ),
    ]

    analysis = service.analyze(results)

    assert analysis.total_queries == 2
    assert analysis.successful_queries == 2
    assert analysis.failed_queries == 0
    assert analysis.success_rate == 100.0
    assert analysis.average_latency_ms == 150.0
    assert analysis.total_estimated_cost_usd == pytest.approx(0.003)
    assert analysis.average_cost_per_successful_query_usd == pytest.approx(
        0.0015
    )
    assert analysis.categories["simple"].total_queries == 2


def test_analyze_failed_requests(service):
    results = [
        BenchmarkResult(
            query_id="q1",
            category="coding",
            model="gemini/test",
            response_text="",
            success=False,
            error="Provider unavailable",
        ),
        BenchmarkResult(
            query_id="q2",
            category="coding",
            model="gemini/test",
            response_text="Answer",
            latency_ms=100.0,
            estimated_cost_usd=0.001,
            success=True,
        ),
    ]

    analysis = service.analyze(results)

    assert analysis.total_queries == 2
    assert analysis.successful_queries == 1
    assert analysis.failed_queries == 1
    assert analysis.success_rate == 50.0
    assert analysis.average_latency_ms == 100.0
    assert analysis.categories["coding"].failed_queries == 1


def test_analyze_missing_metrics(service):
    results = [
        BenchmarkResult(
            query_id="q1",
            category="reasoning",
            model="groq/test",
            response_text="Answer 1",
            latency_ms=100.0,
            success=True,
        ),
        BenchmarkResult(
            query_id="q2",
            category="reasoning",
            model="groq/test",
            response_text="Answer 2",
            latency_ms=200.0,
            estimated_cost_usd=0.002,
            success=True,
        ),
    ]

    analysis = service.analyze(results)

    assert analysis.average_latency_ms == 150.0
    assert analysis.total_estimated_cost_usd is None
    assert analysis.average_cost_per_successful_query_usd is None
    assert (
        analysis.categories["reasoning"].total_estimated_cost_usd
        is None
    )


def test_analyze_empty_results(service):
    analysis = service.analyze([])

    assert analysis.total_queries == 0
    assert analysis.average_latency_ms is None
    assert analysis.failed_queries == 0
    assert analysis.success_rate == 0.0
    assert analysis.average_latency_ms is None
    assert analysis.total_estimated_cost_usd is None
    assert analysis.average_cost_per_successful_query_usd is None
    assert analysis.categories == {}