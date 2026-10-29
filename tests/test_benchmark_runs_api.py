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

QUERIES = [
    BenchmarkQuery(
        id="q1",
        category="simple",
        prompt="Prompt 1",
        evaluation_criteria=["Criterion 1"],
    ),
    BenchmarkQuery(
        id="q2",
        category="simple",
        prompt="Prompt 2",
        evaluation_criteria=["Criterion 2"],
    ),
]


class FakeEvaluator:
    judge_model = JUDGE

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, query, result) -> BenchmarkEvaluation:
        self.calls += 1

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
def client():
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

    yield TestClient(app)

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def stub_benchmark(monkeypatch):
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
        for query in QUERIES
    ]

    runner = MagicMock()
    runner.run_benchmark = AsyncMock(return_value=results)

    monkeypatch.setattr(
        "app.main.BenchmarkService.load_queries",
        lambda self: QUERIES,
    )
    monkeypatch.setattr(
        "app.main.BenchmarkRunner",
        lambda model_service_factory: runner,
    )


@pytest.fixture
def evaluator(monkeypatch):
    fake = FakeEvaluator()
    monkeypatch.setattr(
        "app.main.BenchmarkEvaluationService",
        lambda: fake,
    )
    return fake


def create_run(client) -> str:
    response = client.post(
        "/benchmark/runs",
        json={"model_names": [MODEL]},
    )

    assert response.status_code == 201

    return response.json()["run_id"]


def test_create_and_fetch_run(client, stub_benchmark):
    response = client.post(
        "/benchmark/runs",
        json={"model_names": [MODEL]},
    )

    assert response.status_code == 201

    summary = response.json()
    assert summary["total_results"] == 2
    assert summary["successful_results"] == 2
    assert summary["failed_results"] == 0
    assert summary["model_names"] == [MODEL]

    detail = client.get(f"/benchmark/runs/{summary['run_id']}")

    assert detail.status_code == 200
    assert len(detail.json()["results"]) == 2

    listing = client.get("/benchmark/runs")

    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_get_unknown_run_returns_404(client):
    assert client.get("/benchmark/runs/missing").status_code == 404
    assert (
        client.post("/benchmark/runs/missing/evaluate").status_code == 404
    )
    assert (
        client.get("/benchmark/runs/missing/comparison").status_code == 404
    )


def test_create_run_rejects_unsupported_model(client):
    response = client.post(
        "/benchmark/runs",
        json={"model_names": ["unknown/model"]},
    )

    assert response.status_code == 400
    assert "Unsupported models" in response.json()["detail"]


def test_create_run_rejects_empty_model_list(client):
    response = client.post(
        "/benchmark/runs",
        json={"model_names": []},
    )

    assert response.status_code == 422


def test_evaluate_run_persists_and_skips_existing(
    client,
    stub_benchmark,
    evaluator,
):
    run_id = create_run(client)

    first = client.post(f"/benchmark/runs/{run_id}/evaluate")

    assert first.status_code == 200
    assert first.json()["evaluated_results"] == 2
    assert evaluator.calls == 2

    second = client.post(f"/benchmark/runs/{run_id}/evaluate")

    assert second.status_code == 200
    assert second.json()["total_results"] == 0
    assert evaluator.calls == 2

    forced = client.post(f"/benchmark/runs/{run_id}/evaluate?force=true")

    assert forced.status_code == 200
    assert forced.json()["evaluated_results"] == 2
    assert evaluator.calls == 4


def test_comparison_reflects_stored_evaluations(
    client,
    stub_benchmark,
    evaluator,
):
    run_id = create_run(client)
    client.post(f"/benchmark/runs/{run_id}/evaluate")

    response = client.get(
        f"/benchmark/runs/{run_id}/comparison",
        params={"judge_model": JUDGE},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["judge_model"] == JUDGE
    assert body["judge_cost_usd"] == pytest.approx(0.004)
    assert len(body["models"]) == 1
    assert body["models"][0]["model"] == MODEL
    assert body["models"][0]["evaluated_queries"] == 2
    assert body["models"][0]["pass_rate"] == 100.0
    assert body["models"][0]["self_judged"] is False


def test_evaluate_rejects_changed_dataset(
    client,
    stub_benchmark,
    evaluator,
    monkeypatch,
):
    run_id = create_run(client)

    changed = [
        QUERIES[0].model_copy(
            update={"evaluation_criteria": ["A different criterion"]}
        ),
        QUERIES[1],
    ]
    monkeypatch.setattr(
        "app.main.BenchmarkService.load_queries",
        lambda self: changed,
    )

    response = client.post(f"/benchmark/runs/{run_id}/evaluate")

    assert response.status_code == 409
    assert evaluator.calls == 0