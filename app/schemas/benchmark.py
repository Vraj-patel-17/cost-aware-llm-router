from pydantic import BaseModel, Field

class BenchmarkQuery(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    evaluation_criteria: list[str] = Field(min_length=1)