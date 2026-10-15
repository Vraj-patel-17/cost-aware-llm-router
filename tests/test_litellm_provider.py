from types import SimpleNamespace
from unittest.mock import patch,AsyncMock
import pytest
import asyncio
from app.providers.litellm_provider import LiteLLMProvider
@pytest.mark.asyncio
async def test_generate_returns_normalized_response():
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello, world!"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),)
    provider = LiteLLMProvider(provider="gemini")
    with patch("app.providers.litellm_provider.litellm.acompletion",
        new_callable=AsyncMock, return_value=mock_response) as mock_acompletion:
        result = await provider.generate(prompt="Hello!", model="gemini-test-model")
        mock_acompletion.assert_called_once_with(
            model="gemini-test-model",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    assert result.text == "Hello, world!"
    assert result.model == "gemini-test-model"
    assert result.provider == "gemini"
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.latency_ms >= 0.0

@pytest.mark.asyncio
async def test_generate_handles_missing_usage():
    mock_response = SimpleNamespace(
    choices=[
    SimpleNamespace(
    message=SimpleNamespace(content="Hello!")
    )
    ],
    usage=None,
    )

    provider = LiteLLMProvider(provider="groq")

    with patch(
        "app.providers.litellm_provider.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await provider.generate(
            prompt="Say hello.",
            model="groq-test-model",
        )

    assert result.text == "Hello!"
    assert result.input_tokens is None
    assert result.output_tokens is None

def test_generate_forwards_api_key():
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello!")
            )
        ],
        usage=None,
    )

    provider = LiteLLMProvider(
        provider="gemini",
        api_key="test-api-key",
    )

    async def run_test():
        with patch(
            "app.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_completion:
            await provider.generate(
                prompt="Say hello.",
                model="gemini/gemini-3.5-flash-lite",
            )

            mock_completion.assert_awaited_once_with(
                model="gemini/gemini-3.5-flash-lite",
                messages=[{"role": "user", "content": "Say hello."}],
                api_key="test-api-key",
            )

    asyncio.run(run_test())