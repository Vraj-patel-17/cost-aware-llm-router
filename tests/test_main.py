from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("gemini", settings.gemini_model),
        ("groq", settings.groq_model),
    ],
)
def test_generate_success(provider, expected_model):
    mock_response = {
        "text": "A test response.",
        "model": expected_model,
        "provider": provider,
        "input_tokens": 5,
        "output_tokens": 3,
        "latency_ms": 100.0,
    }

    with patch("app.main.get_model_service") as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.generate = AsyncMock(return_value=mock_response)

        response = client.post(
            "/generate",
            json={
                "provider": provider,
                "prompt": "Explain APIs.",
            },
        )

        assert response.status_code == 200
        assert response.json()["text"] == "A test response."
        assert response.json()["provider"] == provider
        assert response.json()["model"] == expected_model

        mock_service.generate.assert_awaited_once_with(
            prompt="Explain APIs.",
            model=expected_model,
        )
    mock_response = {
        "text": "A test response.",
        "model": expected_model,
        "provider": provider,
        "input_tokens": 5,
        "output_tokens": 3,
        "latency_ms": 100.0,
    }

    with patch(
        "app.main.get_model_service"
    ) as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.generate = AsyncMock(return_value=mock_response)

        response = client.post(
            "/generate",
            json={
                "provider": provider,
                "prompt": "Explain APIs.",
            },
        )

    assert response.status_code == 200
    assert response.json()["text"] == "A test response."
    assert response.json()["provider"] == provider
    assert response.json()["model"] == expected_model

    mock_service.generate.assert_awaited_once_with(
        prompt="Explain APIs.",
        model=expected_model,
    )


def test_generate_unsupported_provider():
    response = client.post(
        "/generate",
        json={
            "provider": "unsupported",
            "prompt": "Explain APIs.",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported provider. Choose 'gemini' or 'groq'."
    )


def test_generate_rejects_empty_prompt():
    response = client.post(
        "/generate",
        json={
            "provider": "gemini",
            "prompt": "",
        },
    )

    assert response.status_code == 422


def test_generate_handles_provider_failure():
    with patch(
        "app.main.get_model_service"
    ) as mock_get_service:
        mock_service = mock_get_service.return_value
        mock_service.generate = AsyncMock(
            side_effect=RuntimeError("Internal provider error")
        )

        response = client.post(
            "/generate",
            json={
                "provider": "groq",
                "prompt": "Explain APIs.",
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The model provider request failed."
    )