from pydantic_settings import BaseSettings,SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = "Cost-Aware LLM Router"
    app_env: str = "development"
    model_config=SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore")
settings = Settings()