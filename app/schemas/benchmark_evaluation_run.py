from pydantic import BaseModel, Field

from app.schemas.benchmark import BenchmarkQuery
from app.schemas.benchmark_evaluation import BenchmarkEvaluation
from app.schemas.benchmark_result import BenchmarkResult


class BenchmarkEvaluationRequest(BaseModel):
    queries: list[BenchmarkQuery] = Field(min_length=1)
    results: list[BenchmarkResult] = Field(min_length=1)


class BenchmarkEvaluationRecord(BaseModel):
    generation_result: BenchmarkResult
    evaluation: BenchmarkEvaluation | None = None
    evaluation_error: str | None = None


class BenchmarkEvaluationRun(BaseModel):
    total_results: int = Field(ge=0)
    evaluated_results: int = Field(ge=0)
    failed_generations: int = Field(ge=0)
    evaluation_errors: int = Field(ge=0)
    records: list[BenchmarkEvaluationRecord]