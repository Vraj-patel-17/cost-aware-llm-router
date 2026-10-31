from app.core.database import Base
from app.main import app

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

EXPECTED_ROUTES = {
    ("GET", "/health"),
    ("POST", "/generate"),
    ("GET", "/analytics/summary"),
    ("GET", "/analytics/models"),
    ("POST", "/benchmark/run"),
    ("POST", "/benchmark/analyze"),
    ("POST", "/benchmark/evaluate"),
    ("POST", "/benchmark/evaluate-all"),
    ("POST", "/benchmark/runs"),
    ("GET", "/benchmark/runs"),
    ("GET", "/benchmark/runs/{run_id}"),
    ("POST", "/benchmark/runs/{run_id}/evaluate"),
    ("GET", "/benchmark/runs/{run_id}/comparison"),
}

EXPECTED_TABLES = {
    "usage_logs",
    "benchmark_runs",
    "benchmark_results",
    "benchmark_evaluations",
}


def test_all_expected_routes_are_registered():
    registered = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in HTTP_METHODS
    }

    missing = EXPECTED_ROUTES - registered

    assert not missing, f"Missing routes: {sorted(missing)}"


def test_importing_main_registers_all_tables():
    assert EXPECTED_TABLES <= set(Base.metadata.tables)