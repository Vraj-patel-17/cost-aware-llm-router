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