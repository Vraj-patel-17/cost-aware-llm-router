from pydantic import BaseModel, Field

class ModelAnalytics(BaseModel):
    model: str
    provider: str
    total_requests: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_estimated_cost_usd: float | None = Field(default=None, ge=0)
    average_latency_ms: float | None = Field(default=None, ge=0)