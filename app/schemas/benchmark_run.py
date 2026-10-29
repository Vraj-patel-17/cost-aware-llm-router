from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.benchmark_result import BenchmarkResult

class BenchmarkRunCreate(BaseModel):
    model_names: list[str] = Field(min_length=1)

class BenchmarkRunSummary(BaseModel):
    run_id: str
    created_at: datetime
    model_names: list[str]
    dataset_hash: str
    total_results: int = Field(ge=0)
    successful_results: int = Field(ge=0)
    failed_results: int = Field(ge=0)

class BenchmarkRunDetail(BenchmarkRunSummary):
    results: list[BenchmarkResult]

class ModelRunSummary(BaseModel):
    model: str
    total_queries: int
    successful_queries: int
    success_rate: float
    average_latency_ms: float | None
    total_generation_cost_usd: float | None
    evaluated_queries: int
    passed_queries: int
    pass_rate: float | None
    average_score: float | None
    cost_per_passed_query_usd: float | None
    self_judged: bool

class BenchmarkRunComparison(BaseModel):
    run_id: str
    judge_model: str
    judge_cost_usd: float | None
    models: list[ModelRunSummary]