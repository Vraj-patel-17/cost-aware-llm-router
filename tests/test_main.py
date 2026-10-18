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
    }

    with patch("app.main.get_model_service") as mock_get_service:
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
    assert response.json()["provider"] == provider
    assert response.json()["model"] == model

    mock_service.generate.assert_awaited_once_with(
        prompt="Explain APIs.",
        model=model,
    )


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