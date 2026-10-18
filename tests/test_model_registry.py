from app.core.config import settings
from app.core.model_registry import MODEL_REGISTRY


def test_gemini_model_is_registered():
    model_config = MODEL_REGISTRY[settings.gemini_model]

    assert model_config["provider"] == "gemini"
    assert model_config["api_key"] == settings.gemini_api_key


def test_groq_model_is_registered():
    model_config = MODEL_REGISTRY[settings.groq_model]

    assert model_config["provider"] == "groq"
    assert model_config["api_key"] == settings.groq_api_key


def test_registry_contains_only_configured_models():
    assert settings.gemini_model in MODEL_REGISTRY
    assert settings.groq_model in MODEL_REGISTRY