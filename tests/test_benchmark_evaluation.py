import pytest
from pydantic import ValidationError

from app.schemas.benchmark_evaluation import (
    BenchmarkEvaluation,
    CriterionEvaluation,
)

def test_criterion_evaluation_valid_data():
    evaluation = CriterionEvaluation(
        criterion="Identifies logarithmic time complexity.",
        passed=True,
        explanation="Correctly identifies O(log n).",
    )

    assert evaluation.passed is True
    assert evaluation.criterion == "Identifies logarithmic time complexity."

def test_criterion_evaluation_rejects_empty_criterion():
    with pytest.raises(ValidationError):
        CriterionEvaluation(
            criterion="",
            passed=True,
            explanation="Correct answer.",
        )

def test_benchmark_evaluation_valid_data():
    evaluation = BenchmarkEvaluation(
        query_id="simple_01",
        model="gemini/gemini-3.5-flash-lite",
        score=5,
        passed=True,
        reasoning="The response correctly explains binary search.",
        criteria_evaluations=[
            CriterionEvaluation(
                criterion="Identifies logarithmic time complexity.",
                passed=True,
                explanation="Correctly identifies O(log n).",
            )
        ],
    )

    assert evaluation.score == 5
    assert evaluation.passed is True
    assert len(evaluation.criteria_evaluations) == 1


@pytest.mark.parametrize("score", [0, 6, -1])
def test_benchmark_evaluation_rejects_invalid_score(score):
    with pytest.raises(ValidationError):
        BenchmarkEvaluation(
            query_id="simple_01",
            model="gemini/gemini-3.5-flash-lite",
            score=score,
            passed=True,
            reasoning="Example evaluation.",
            criteria_evaluations=[
                CriterionEvaluation(
                    criterion="Identifies logarithmic time complexity.",
                    passed=True,
                    explanation="Correctly identifies O(log n).",
                )
            ],
        )

def test_benchmark_evaluation_rejects_empty_criteria():
    with pytest.raises(ValidationError):
        BenchmarkEvaluation(
            query_id="simple_01",
            model="gemini/gemini-3.5-flash-lite",
            score=5,
            passed=True,
            reasoning="Example evaluation.",
            criteria_evaluations=[],
        )