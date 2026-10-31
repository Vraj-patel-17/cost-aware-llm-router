from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base
from app.core.dependencies import get_db
from app.main import app
from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_evaluation import (
    BenchmarkEvaluation,
    CriterionEvaluation,
)
from app.schemas.benchmark_result import BenchmarkResult

MODEL = settings.gemini_model
JUDGE = "test/judge"

LOAD_QUERIES = "app.services.benchmark_service.BenchmarkService.load_queries"
RUNNER = "app.routers.benchmark_runs.BenchmarkRunner"
EVALUATOR = "app.routers.benchmark_runs.BenchmarkEvaluationService"

QUERIES = [
    BenchmarkQuery(
        id=f"q{i}",
        category="simple",
        prompt=f"Prompt {i}",
        evaluation_criteria=["Criterion"],
    )
    for i in (1, 2, 3)
]


class FakeEvaluator:
    judge_model = JUDGE

    async def evaluate(self, query, result) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(
            query_id=query.id,
            model=result.model,
            score=5,
            passed=True,
            reasoning="Good.",
            criteria_evaluations=[
                CriterionEvaluation(
                    criterion="Criterion",
                    passed=True,
                    explanation="Met.",
                )
            ],
            judge_model=JUDGE,
            judge_input_tokens=100,
            judge_output_tokens=50,
            judge_cost_usd=0.002,
        )


@pytest.fixture
def client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # q1 and q2 succeed; q3 fails. All three exist in QUERIES, because
    # evaluation rejects results whose query is not in the dataset.
    results = [
        BenchmarkResult(
            query_id=query.id,
            category=query.category,
            model=MODEL,
            response_text="answer",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            estimated_cost_usd=0.001,
            success=True,
        )
        for query in QUERIES[:2]
    ] + [
        BenchmarkResult(
            query_id=QUERIES[2].id,
            category=QUERIES[2].category,
            model=MODEL,
            response_text="",
            success=False,
            error="Provider unavailable",
        )
    ]

    runner = MagicMock()
    runner.run_benchmark = AsyncMock(return_value=results)

    monkeypatch.setattr(LOAD_QUERIES, lambda self: QUERIES)
    monkeypatch.setattr(
        RUNNER,
        lambda model_service_factory: runner,
    )
    monkeypatch.setattr(EVALUATOR, lambda: FakeEvaluator())

    yield TestClient(app)

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_benchmark_run_and_judge_calls_appear_in_analytics(client):
    assert client.get("/analytics/summary").json()["total_requests"] == 0

    response = client.post(
        "/benchmark/runs",
        json={"model_names": [MODEL]},
    )
    assert response.status_code == 201
    run_id = response.json()["run_id"]

    summary = client.get("/analytics/summary").json()

    # Two successful generations are logged; the failed one is not.
    assert summary["total_requests"] == 2
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.002)

    evaluation = client.post(f"/benchmark/runs/{run_id}/evaluate")

    assert evaluation.status_code == 200
    assert evaluation.json()["failed_generations"] == 1
    assert evaluation.json()["evaluated_results"] == 2

    summary = client.get("/analytics/summary").json()

    assert summary["total_requests"] == 4
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.006)

    models = {
        item["model"]: item for item in client.get("/analytics/models").json()
    }

    assert models[MODEL]["total_requests"] == 2
    assert models[JUDGE]["total_requests"] == 2
    assert models[JUDGE]["total_estimated_cost_usd"] == pytest.approx(0.004)