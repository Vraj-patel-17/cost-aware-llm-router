import asyncio
from app.core.config import settings
from app.providers.litellm_provider import LiteLLMProvider
async def main()-> None:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment variables.")
    provider = LiteLLMProvider(
        provider="groq",
        api_key=settings.groq_api_key
    )
    response = await provider.generate(
        prompt="Explain what an api is in one line",
        model=settings.groq_model
    )
    print("\n--- Groq Response ---")
    print(f"Text: {response.text}")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}") 
    print(f"Input tokens: {response.input_tokens}") 
    print(f"Output tokens: {response.output_tokens}")
    print(f"Latency: {response.latency_ms:.2f} ms")

if __name__=="__main__":
    asyncio.run(main())