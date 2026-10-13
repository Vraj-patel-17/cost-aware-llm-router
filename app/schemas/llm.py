from pydantic import BaseModel, Field
class LLMResponse(BaseModel):
    """
    LLMResponse is a Pydantic model that represents the response from a language model.
    """
    text: str
    model: str
    provider: str
    input_tokens: int | None =Field(default=None,ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(ge=0.0)