from pydantic import BaseModel

class CategoryAnalysis(BaseModel):
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    average_latency_ms: float | None
    total_estimated_cost_usd: float | None

class BenchmarkAnalysis(BaseModel):
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    average_latency_ms: float | None
    total_estimated_cost_usd: float | None
    average_cost_per_successful_query_usd: float | None
    categories: dict[str, CategoryAnalysis]