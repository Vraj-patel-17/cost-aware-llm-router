from app.core.config import Settings

def test_settings_have_default_values():
    settings = Settings(_env_file=None)  # Disable loading from .env for this test
    assert settings.app_name == "Cost-Aware LLM Router"
    assert settings.app_env == "development"

def test_settings_can_be_overridden():
    settings = Settings(app_name="Test Router", app_env="testing", _env_file=None)
    assert settings.app_name == "Test Router"
    assert settings.app_env == "testing"