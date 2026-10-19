from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.llm import LLMResponse
from app.services.model_service import ModelService


@pytest.mark.asyncio
async def test_generate_delegates_to_provider():
    expected_response = LLMResponse(
        text="Hello, world!",
        model="test-model",
        provider="test-provider",
        input_tokens=5,
        output_tokens=3,
        latency_ms=100.0,
    )

    fake_provider = AsyncMock()
    fake_provider.generate.return_value = expected_response

    service = ModelService(provider=fake_provider)

    result = await service.generate(
        prompt="Hello, world!",
        model="test-model",
    )

    fake_provider.generate.assert_awaited_once_with(
        prompt="Hello, world!",
        model="test-model",
    )
    assert result == expected_response


@pytest.mark.asyncio
async def test_generate_adds_estimated_cost():
    provider_response = LLMResponse(
        text="Hello!",
        model="test/model",
        provider="test-provider",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=100.0,
    )

    fake_provider = AsyncMock()
    fake_provider.generate.return_value = provider_response

    service = ModelService(provider=fake_provider)

    with patch(
        "app.services.model_service.calculate_cost",
        return_value=0.006,
    ) as mock_calculate_cost:
        result = await service.generate(
            prompt="Hello",
            model="test/model",
        )

    mock_calculate_cost.assert_called_once_with(
        model="test/model",
        input_tokens=1000,
        output_tokens=500,
    )
    assert result.estimated_cost_usd == 0.006


@pytest.mark.asyncio
async def test_generate_skips_cost_when_token_usage_missing():
    provider_response = LLMResponse(
        text="Hello!",
        model="test/model",
        provider="test-provider",
        input_tokens=None,
        output_tokens=500,
        latency_ms=100.0,
    )

    fake_provider = AsyncMock()
    fake_provider.generate.return_value = provider_response

    service = ModelService(provider=fake_provider)

    with patch(
        "app.services.model_service.calculate_cost"
    ) as mock_calculate_cost:
        result = await service.generate(
            prompt="Hello",
            model="test/model",
        )

    mock_calculate_cost.assert_not_called()
    assert result.estimated_cost_usd is None