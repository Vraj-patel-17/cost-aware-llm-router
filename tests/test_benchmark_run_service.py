import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.schemas.benchmark_evaluation import (
    BenchmarkEvaluation,
    CriterionEvaluation,
)
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_run_service import BenchmarkRunService


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def make_result(
    query_id: str,
    model: str = "m1",
    success: bool = True,
) -> BenchmarkResult:
    return BenchmarkResult(
        query_id=query_id,
        category="simple",
        model=model,
        response_text="answer" if success else "",
        input_tokens=10 if success else None,
        output_tokens=5 if success else None,
        latency_ms=100.0 if success else None,
        estimated_cost_usd=0.001 if success else None,
        success=success,
        error=None if success else "Provider unavailable",
    )


def make_evaluation(
    query_id: str,
    model: str = "m1",
    judge_model: str = "judge/a",
    score: int = 5,
    passed: bool = True,
) -> BenchmarkEvaluation:
    return BenchmarkEvaluation(
        query_id=query_id,
        model=model,
        score=score,
        passed=passed,
        reasoning="Reasoning.",
        criteria_evaluations=[
            CriterionEvaluation(
                criterion="Criterion",
                passed=passed,
                explanation="Explanation.",
            )
        ],
        judge_model=judge_model,
        judge_input_tokens=100,
        judge_output_tokens=50,
        judge_cost_usd=0.001,
    )


def test_create_run_round_trips_results(db_session):
    results = [
        make_result("q1"),
        make_result("q2", success=False),
    ]
    service = BenchmarkRunService(db_session)

    run = service.create_run(
        model_names=["m1"],
        dataset_hash="abc",
        results=results,
    )

    assert run.id
    assert run.created_at is not None
    assert service.get_run(run.id) is not None
    assert service.get_results(run.id) == results


def test_build_summary_counts_successes_and_failures(db_session):
    service = BenchmarkRunService(db_session)
    run = service.create_run(
        model_names=["m1"],
        dataset_hash="abc",
        results=[
            make_result("q1"),
            make_result("q2"),
            make_result("q3", success=False),
        ],
    )

    summary = service.build_summary(run)

    assert summary.run_id == run.id
    assert summary.model_names == ["m1"]
    assert summary.dataset_hash == "abc"
    assert summary.total_results == 3
    assert summary.successful_results == 2
    assert summary.failed_results == 1


def test_list_runs_returns_all_runs(db_session):
    service = BenchmarkRunService(db_session)
    first = service.create_run(["m1"], "h1", [make_result("q1")])
    second = service.create_run(["m1"], "h2", [make_result("q1")])

    run_ids = {run.id for run in service.list_runs()}
    summaries = service.build_summaries(service.list_runs())

    assert run_ids == {first.id, second.id}
    assert len(summaries) == 2
    assert service.build_summaries([]) == []


def test_get_run_returns_none_for_unknown_id(db_session):
    assert BenchmarkRunService(db_session).get_run("missing") is None


def test_save_evaluations_round_trips(db_session):
    service = BenchmarkRunService(db_session)
    run = service.create_run(["m1"], "abc", [make_result("q1")])
    evaluation = make_evaluation("q1")

    service.save_evaluations(run.id, [evaluation])

    assert service.get_evaluations(run.id) == [evaluation]


def test_save_evaluations_replaces_same_judge_and_keeps_others(db_session):
    service = BenchmarkRunService(db_session)
    run = service.create_run(["m1"], "abc", [make_result("q1")])

    service.save_evaluations(
        run.id,
        [
            make_evaluation("q1", judge_model="judge/a", score=2, passed=False),
            make_evaluation("q1", judge_model="judge/b", score=4),
        ],
    )
    service.save_evaluations(
        run.id,
        [make_evaluation("q1", judge_model="judge/a", score=5)],
    )

    judge_a = service.get_evaluations(run.id, judge_model="judge/a")
    judge_b = service.get_evaluations(run.id, judge_model="judge/b")

    assert len(judge_a) == 1
    assert judge_a[0].score == 5
    assert len(judge_b) == 1
    assert judge_b[0].score == 4
    assert len(service.get_evaluations(run.id)) == 2


def test_get_evaluated_keys_is_scoped_to_judge(db_session):
    service = BenchmarkRunService(db_session)
    run = service.create_run(
        ["m1"],
        "abc",
        [make_result("q1"), make_result("q2")],
    )
    service.save_evaluations(
        run.id,
        [make_evaluation("q1", judge_model="judge/a")],
    )

    assert service.get_evaluated_keys(run.id, "judge/a") == {("q1", "m1")}
    assert service.get_evaluated_keys(run.id, "judge/b") == set()