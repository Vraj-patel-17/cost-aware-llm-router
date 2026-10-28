from app.core.config import settings

_API_KEYS: dict[str, str | None] = {
    "gemini": settings.gemini_api_key,
    "groq": settings.groq_api_key,
}

MODEL_REGISTRY: dict[str, dict[str, str | None]] = {
    settings.gemini_model: {
        "provider": "gemini",
        "api_key": settings.gemini_api_key,
    },
    settings.groq_model: {
        "provider": "groq",
        "api_key": settings.groq_api_key,
    },
}

if settings.judge_model and settings.judge_model not in MODEL_REGISTRY:
    _judge_provider = settings.judge_model.split("/", 1)[0]
    MODEL_REGISTRY[settings.judge_model] = {
        "provider": _judge_provider,
        "api_key": _API_KEYS.get(_judge_provider),
    }