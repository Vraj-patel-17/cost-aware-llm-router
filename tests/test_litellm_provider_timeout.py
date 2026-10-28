import asyncio
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.providers.litellm_provider import LiteLLMProvider


@pytest.mark.asyncio
async def test_generate_times_out(monkeypatch):
    async def slow_completion(**kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(settings, "request_timeout_seconds", 0.01)
    provider = LiteLLMProvider(provider="gemini")

    with patch(
        "app.providers.litellm_provider.litellm.acompletion",
        new=slow_completion,
    ):
        with pytest.raises(TimeoutError):
            await provider.generate(
                prompt="Hello",
                model="gemini-test-model",
            )