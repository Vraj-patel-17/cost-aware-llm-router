from app.core.config import settings
MODEL_REGISTRY:dict[str,dict[str,str | None]]={
    settings.gemini_model:{
        "provider":"gemini",
        "api_key":settings.gemini_api_key,
    },
    settings.groq_model:{
        "provider":"groq",
        "api_key":settings.groq_api_key,
    },
}