import json
from collections import Counter
from pathlib import Path

from app.services.benchmark_service import BenchmarkService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "benchmark_queries.json"

EXPECTED_CATEGORIES = {"simple", "reasoning", "coding", "extraction"}
EXPECTED_DIFFICULTIES = {"easy", "medium", "hard"}
EXPECTED_SPLITS = {"dev", "test"}
REQUIRED_FIELDS = {
    "id",
    "category",
    "prompt",
    "evaluation_criteria",
    "difficulty",
    "split",
}

MIN_QUERIES = 40
MIN_PER_CATEGORY = 10
MIN_DEV_PER_CATEGORY = 3
MIN_TEST_PER_CATEGORY = 2
MIN_NON_EASY_SHARE = 0.4


def load_raw_queries() -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_benchmark_dataset_has_required_fields_and_unique_ids():
    queries = load_raw_queries()

    assert len(queries) >= MIN_QUERIES

    query_ids = [query["id"] for query in queries]
    assert len(query_ids) == len(set(query_ids)), "Query IDs must be unique"

    assert all(REQUIRED_FIELDS <= query.keys() for query in queries)
    assert all(query["difficulty"] in EXPECTED_DIFFICULTIES for query in queries)
    assert all(query["split"] in EXPECTED_SPLITS for query in queries)
    assert {query["category"] for query in queries} == EXPECTED_CATEGORIES


def test_query_ids_match_their_category():
    for query in load_raw_queries():
        assert query["id"].startswith(f"{query['category']}_"), query["id"]


def test_every_category_has_dev_and_test_queries():
    queries = load_raw_queries()

    per_category = Counter(query["category"] for query in queries)
    per_category_split = Counter(
        (query["category"], query["split"]) for query in queries
    )

    for category in EXPECTED_CATEGORIES:
        assert per_category[category] >= MIN_PER_CATEGORY, category
        assert (
            per_category_split[(category, "dev")] >= MIN_DEV_PER_CATEGORY
        ), category
        assert (
            per_category_split[(category, "test")] >= MIN_TEST_PER_CATEGORY
        ), category


def test_difficulty_mix_is_not_all_easy():
    queries = load_raw_queries()
    difficulties = Counter(query["difficulty"] for query in queries)

    assert set(difficulties) == EXPECTED_DIFFICULTIES

    non_easy = len(queries) - difficulties["easy"]
    assert non_easy / len(queries) >= MIN_NON_EASY_SHARE


def test_prompts_and_criteria_are_unique_and_non_empty():
    queries = load_raw_queries()

    prompts = [query["prompt"].strip() for query in queries]
    assert all(prompts)
    assert len(prompts) == len(set(prompts)), "Prompts must be unique"

    for query in queries:
        criteria = query["evaluation_criteria"]

        assert criteria, query["id"]
        assert all(criterion.strip() for criterion in criteria), query["id"]
        assert len(criteria) == len(set(criteria)), query["id"]


def test_benchmark_service_loads_queries():
    queries = BenchmarkService().load_queries()

    assert len(queries) >= MIN_QUERIES
    assert all(query.id for query in queries)
    assert all(query.prompt for query in queries)
    assert all(query.evaluation_criteria for query in queries)
    assert {query.difficulty for query in queries} == EXPECTED_DIFFICULTIES
    assert {query.split for query in queries} == EXPECTED_SPLITS