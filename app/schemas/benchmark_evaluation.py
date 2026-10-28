from pydantic import BaseModel, Field


class CriterionEvaluation(BaseModel):
    criterion: str = Field(min_length=1)
    passed: bool
    explanation: str = Field(min_length=1)

class BenchmarkEvaluation(BaseModel):
    query_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    score: int = Field(ge=1, le=5)
    passed: bool
    reasoning: str = Field(min_length=1)
    criteria_evaluations: list[CriterionEvaluation] = Field(min_length=1)

    # Judge metadata
    judge_model: str | None = None
    judge_input_tokens: int | None = Field(default=None, ge=0)
    judge_output_tokens: int | None = Field(default=None, ge=0)
    judge_cost_usd: float | None = Field(default=None, ge=0.0)