from pydantic_settings import BaseSettings,SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = "Cost-Aware LLM Router"
    app_env: str = "development"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini/gemini-3.5-flash-lite"
    groq_api_key:str | None = None
    groq_model: str = "groq/openai/gpt-oss-120b"
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore")
settings = Settings()