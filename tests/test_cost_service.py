from app.core.model_pricing import MODEL_PRICING
from app.services.cost_service import calculate_cost


def test_calculates_cost_correctly(monkeypatch):
    pricing = {
        "input": 2.0,
        "output": 8.0,
    }

    monkeypatch.setitem(
        MODEL_PRICING,
        "test/model",
        pricing,
    )

    result = calculate_cost(
        model="test/model",
        input_tokens=1000,
        output_tokens=500,
    )

    assert result == 0.006


def test_returns_zero_for_zero_tokens(monkeypatch):
    monkeypatch.setitem(
        MODEL_PRICING,
        "test/model",
        {"input": 2.0, "output": 8.0},
    )

    result = calculate_cost(
        model="test/model",
        input_tokens=0,
        output_tokens=0,
    )

    assert result == 0.0


def test_returns_none_for_unknown_model():
    result = calculate_cost(
        model="unknown/model",
        input_tokens=100,
        output_tokens=100,
    )

    assert result is None