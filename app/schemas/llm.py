from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    model: str


class LLMResponse(BaseModel):
    """Represents a normalized response from a language model."""
    text: str
    model: str
    provider: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)