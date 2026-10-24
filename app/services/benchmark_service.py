import json
from pathlib import Path

from app.schemas.benchmark import BenchmarkQuery

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "benchmark_queries.json"
EXPECTED_CATEGORIES = {"simple", "reasoning", "coding", "extraction"}

class BenchmarkService:
    def load_queries(self) -> list[BenchmarkQuery]:
        with DATASET_PATH.open(encoding="utf-8") as file:
            raw_queries = json.load(file)

        queries = [
            BenchmarkQuery.model_validate(query)
            for query in raw_queries
        ]

        query_ids = [query.id for query in queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Benchmark query IDs must be unique")

        categories = {query.category for query in queries}
        unexpected_categories = categories - EXPECTED_CATEGORIES
        if unexpected_categories:
            raise ValueError(
                f"Unexpected benchmark categories: {unexpected_categories}"
            )

        return queries