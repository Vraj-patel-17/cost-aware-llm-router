from pydantic import BaseModel, Field


class BenchmarkResult(BaseModel):
    query_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    model: str = Field(min_length=1)

    response_text: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)

    success: bool
    error: str | None = None