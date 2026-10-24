import json
from pathlib import Path
from app.services.benchmark_service import BenchmarkService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "benchmark_queries.json"

EXPECTED_CATEGORIES = {"simple", "reasoning", "coding", "extraction"}


def test_benchmark_dataset_is_valid():
    with DATASET_PATH.open(encoding="utf-8") as file:
        queries = json.load(file)

    assert len(queries) == 20

    query_ids = [query["id"] for query in queries]
    assert len(query_ids) == len(set(query_ids)), "Query IDs must be unique"

    assert all(
        {"id", "category", "prompt", "evaluation_criteria"} <= query.keys()
        for query in queries
    )

    categories = {query["category"] for query in queries}
    assert categories == EXPECTED_CATEGORIES

    for category in EXPECTED_CATEGORIES:
        assert sum(query["category"] == category for query in queries) == 5

    assert all(query["prompt"].strip() for query in queries)
    assert all(query["evaluation_criteria"] for query in queries)

def test_benchmark_service_loads_queries():
    service = BenchmarkService()

    queries = service.load_queries()

    assert len(queries) == 20
    assert all(query.id for query in queries)
    assert all(query.prompt for query in queries)
    assert all(query.evaluation_criteria for query in queries)