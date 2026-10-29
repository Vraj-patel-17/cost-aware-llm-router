import pytest

from app.schemas.benchmark_evaluation import (
    BenchmarkEvaluation,
    CriterionEvaluation,
)
from app.schemas.benchmark_result import BenchmarkResult
from app.services.benchmark_comparison_service import (
    BenchmarkComparisonService,
)


def make_result(
    query_id: str,
    model: str,
    success: bool = True,
) -> BenchmarkResult:
    return BenchmarkResult(
        query_id=query_id,
        category="simple",
        model=model,
        response_text="answer" if success else "",
        latency_ms=100.0 if success else None,
        estimated_cost_usd=0.001 if success else None,
        success=success,
        error=None if success else "failed",
    )


def make_evaluation(
    query_id: str,
    model: str,
    score: int,
    passed: bool,
    judge_cost: float | None = 0.001,
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
        judge_model="judge/x",
        judge_cost_usd=judge_cost,
    )


def test_compare_summarizes_each_model():
    results = [
        make_result("q1", "m1"),
        make_result("q2", "m1"),
        make_result("q1", "m2", success=False),
    ]
    evaluations = [
        make_evaluation("q1", "m1", score=5, passed=True),
        make_evaluation("q2", "m1", score=2, passed=False),
    ]

    comparison = BenchmarkComparisonService().compare(
        run_id="run-1",
        judge_model="judge/x",
        results=results,
        evaluations=evaluations,
    )

    m1, m2 = comparison.models

    assert comparison.run_id == "run-1"
    assert comparison.judge_cost_usd == pytest.approx(0.002)

    assert m1.model == "m1"
    assert m1.success_rate == 100.0
    assert m1.evaluated_queries == 2
    assert m1.passed_queries == 1
    assert m1.pass_rate == 50.0
    assert m1.average_score == 3.5
    assert m1.total_generation_cost_usd == pytest.approx(0.002)
    assert m1.cost_per_passed_query_usd == pytest.approx(0.002)
    assert m1.self_judged is False

    assert m2.model == "m2"
    assert m2.success_rate == 0.0
    assert m2.evaluated_queries == 0
    assert m2.pass_rate is None
    assert m2.average_score is None
    assert m2.cost_per_passed_query_usd is None


def test_compare_flags_self_judged_model():
    comparison = BenchmarkComparisonService().compare(
        run_id="run-1",
        judge_model="m1",
        results=[make_result("q1", "m1"), make_result("q1", "m2")],
        evaluations=[],
    )

    flags = {item.model: item.self_judged for item in comparison.models}

    assert flags == {"m1": True, "m2": False}
    assert comparison.judge_cost_usd is None


def test_compare_judge_cost_is_none_when_any_cost_missing():
    comparison = BenchmarkComparisonService().compare(
        run_id="run-1",
        judge_model="judge/x",
        results=[make_result("q1", "m1"), make_result("q2", "m1")],
        evaluations=[
            make_evaluation("q1", "m1", 5, True, judge_cost=0.001),
            make_evaluation("q2", "m1", 5, True, judge_cost=None),
        ],
    )

    assert comparison.judge_cost_usd is None