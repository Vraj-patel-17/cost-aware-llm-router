from app.schemas.llm import LLMResponse

def test_valid_llm_request():
    response=LLMResponse(
        text="Hello, world!",
        model="gpt-3.5-turbo",
        provider="OpenAI",
        input_tokens=5,
        output_tokens=3,
        latency_ms=150.0
    )
    assert response.text == "Hello, world!"
    assert response.model == "gpt-3.5-turbo"
    assert response.provider == "OpenAI"
    assert response.input_tokens == 5
    assert response.output_tokens == 3
    assert response.latency_ms == 150.0

def test_token_counts_can_be_missing():
    response = LLMResponse(
        text="Hello!",
        model="example-model",
        provider="example-provider",
        latency_ms=50.0,
    )

    assert response.input_tokens is None
    assert response.output_tokens is None

def test_negative_latency_is_rejected():
    try:
        LLMResponse(
            text="Hello!",
            model="example-model",
            provider="example-provider",
            latency_ms=-1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Negative latency should be rejected")