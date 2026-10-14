from unittest.mock import AsyncMock
import pytest
from app.schemas.llm import  LLMResponse
from app.services.model_service import ModelService
@pytest.mark.asyncio
async def test_generate_delegates_to_provider():
    except_response = LLMResponse(
        text="Hello, world!",
        model="test-model",
        provider="test-provider",
        input_tokens=5,
        output_tokens=3,
        latency_ms=100.0,
    )
    fake_provider = AsyncMock()
    fake_provider.generate.return_value = except_response
    service = ModelService(provider=fake_provider)
    result = await service.generate(prompt="Hello, world!", model="test-model")
    assert result is except_response