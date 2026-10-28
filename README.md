# Cost-Aware LLM Router

A system that routes queries to cost-effective language models while maintaining an acceptable level of response quality.

## Problem

Using the most powerful language model for every request can increase inference costs unnecessarily. Many queries can potentially be handled by cheaper models.

This project explores whether intelligent model selection can reduce inference costs without significantly compromising answer quality.

## Project Goals

* Integrate multiple language models through a common interface.
* Establish an always-use-the-strongest-model baseline.
* Develop routing strategies based on query characteristics.
* Measure inference costs, response quality, and latency.
* Compare routing strategies using a reproducible evaluation dataset.
* Evaluate the trade-off between cost savings and answer quality.

## Technology Stack

* **Language:** Python 3.14
* **API:** FastAPI
* **Validation:** Pydantic
* **LLM integration:** LiteLLM
* **Configuration:** Pydantic Settings
* **Persistence:** SQLAlchemy with SQLite
* **Testing:** pytest, pytest-asyncio
* **Dependency management:** uv

Planned technologies include scikit-learn for learned routing and Streamlit for benchmark visualization.

## Current Implementation

* FastAPI application with health-check, model-generation, analytics, and benchmark endpoints.
* Centralized configuration using Pydantic Settings.
* Configurable model registry mapping supported model IDs to providers and API keys.
* Asynchronous model execution through LiteLLM, with a configurable per-request timeout.
* Gemini and Groq integrations verified through real API calls.
* Normalized model responses containing generated text, provider, token usage, latency, and estimated inference cost.
* Provider-independent cost estimation using configured input and output token prices.
* SQLite persistence for model usage records, including model, provider, token usage, latency, estimated cost, and timestamp.
* Aggregate usage analytics through `GET /analytics/summary`.
* Per-model usage analytics through `GET /analytics/models`.
* A benchmark dataset containing 20 queries across four categories: simple questions, reasoning, coding, and information extraction.
* Concurrent benchmark execution through `POST /benchmark/run`, with a configurable concurrency limit and support for running selected models against the same query dataset.
* Benchmark analysis through `POST /benchmark/analyze`, calculating success rates, average latency, and estimated costs overall and by category.
* LLM-as-judge quality evaluation through `POST /benchmark/evaluate`, using evaluation criteria defined for each benchmark query.
* A configurable judge model. The candidate model's name is never shown to the judge, and the judge model, token usage, and estimated cost are recorded on every evaluation.
* Concurrent batch quality evaluation through `POST /benchmark/evaluate-all`, evaluating previously generated benchmark results and preserving generation failures and individual evaluation errors.
* Structured evaluation results containing quality scores, pass/fail outcomes, criterion-level assessments, and explanations.
* Automated tests covering API behavior, model registration, cost calculations, usage persistence, analytics aggregation, benchmark execution and concurrency, analysis, quality evaluation, and provider timeouts.
* Mocked provider tests that do not require live LLM API calls.

## Current Limitations

* Model selection is explicit; automatic cost-aware routing has not yet been implemented.
* Only Gemini and Groq integrations have been verified against real APIs.
* Configured token prices are estimates and must be rechecked when provider pricing changes.
* Estimated costs may differ from actual charges because of free-tier allowances, discounts, provider-specific billing rules, and how each provider counts reasoning tokens.
* Usage records are stored locally in SQLite.
* The benchmark dataset currently contains 20 queries, which is insufficient for strong general conclusions about model quality. Most queries are easy enough that capable models are likely to pass nearly all of them, so the dataset cannot yet separate models well.
* Quality evaluation relies on an LLM judge. Its scores may be subjective or inconsistent and should not be treated as ground truth. Several criteria (sentence counts, JSON shape, exact values) could be checked programmatically instead.
* By default the judge is the Gemini model, which is also a benchmark candidate, so scores for that model may be inflated. Set `JUDGE_MODEL` to a model that is not being benchmarked.
* Batch evaluation processes previously generated results; generation and evaluation are separate operations.
* Judge cost is recorded on each evaluation result but is not written to the usage table, and benchmark generation calls are not logged there either, so `/analytics` does not include benchmark traffic.
* Benchmark results and quality evaluations are returned by the API but are not yet persisted as benchmark runs.
* The API has no authentication or rate limiting and should not be exposed publicly.
* Response caching, automatic fallback and retries, a learned routing strategy, and a Streamlit dashboard have not yet been implemented.

## Evaluation Plan

### Evaluation Strategy

The project aims to compare cost-aware routing against an always-use-the-strongest-model baseline.

The current benchmark dataset contains 20 queries distributed across four categories:

* **Simple:** Straightforward questions.
* **Reasoning:** Questions requiring logical reasoning.
* **Coding:** Programming-related tasks.
* **Extraction:** Tasks that extract specific information from supplied text.

Each query includes evaluation criteria used to assess generated responses.

### Evaluation Workflow

1. **Generate:** Run selected models against the same benchmark queries using `POST /benchmark/run`.
2. **Analyze:** Calculate generation success rates, latency, and estimated costs using `POST /benchmark/analyze`.
3. **Evaluate quality:** Assess individual responses against their criteria using `POST /benchmark/evaluate`.
4. **Batch evaluation:** Evaluate previously generated results using `POST /benchmark/evaluate-all`.
5. **Compare:** Use generation costs, latency, and quality outcomes to understand the trade-offs between models.

### Key Metrics

* **Response quality:** LLM-judge scores and criterion-level pass/fail results.
* **Success rate:** Percentage of benchmark generations completed successfully.
* **Inference cost:** Estimated generation cost per request and across the benchmark.
* **Judge cost:** Estimated evaluation cost per response, tracked separately from generation cost.
* **Latency:** Time required to generate responses.
* **Cost savings:** Reduction in estimated costs relative to the baseline.
* **Quality retention:** Whether a cheaper model maintains acceptable response quality compared with the baseline.

The current evaluation infrastructure provides the foundation for comparing models. Establishing a reliable baseline and implementing cost-aware routing remain future milestones. The 20-query dataset is an initial development benchmark, not sufficient evidence of general model performance.

## Project Status

**Current phase:** Benchmarking and quality evaluation.

The project currently supports concurrent model execution, estimated cost calculation, SQLite usage logging, usage analytics, benchmark execution, aggregate benchmark analysis, and individual and batch quality evaluation with a configurable judge.

The next major milestone is to implement and evaluate a cost-aware routing strategy using measured quality, latency, and cost rather than assuming that cheaper models are sufficient for every query.

## Roadmap

* [x] FastAPI foundation and health endpoint
* [x] Multi-provider execution through LiteLLM
* [x] Configurable model registry
* [x] Token usage and estimated cost calculation
* [x] SQLite usage logging
* [x] Aggregate and per-model usage analytics
* [x] Initial benchmark dataset
* [x] Benchmark execution and result analysis
* [x] LLM-as-judge quality evaluation
* [x] Batch benchmark evaluation
* [x] Concurrent benchmark execution with request timeouts
* [x] Track evaluator token usage and costs separately
* [ ] Use a judge model that is not a benchmark candidate
* [ ] Persist benchmark runs and evaluation results
* [ ] Expand and validate the benchmark dataset (harder prompts, dev/test split)
* [ ] Score objective criteria programmatically and calibrate the judge against human grading
* [ ] Establish an always-use-the-strongest-model baseline
* [ ] Compute an oracle router (cheapest passing model per query) as a savings ceiling
* [ ] Implement rule-based cost-aware routing
* [ ] Compare routing strategies against the baseline
* [ ] Evaluate a learned routing strategy
* [ ] Add authentication and rate limiting
* [ ] Add response caching, retries, and fallback handling
* [ ] Build a Streamlit analytics dashboard
* [ ] Improve deployment and CI/CD

## API Endpoints

| Method | Endpoint                  | Purpose                                                       |
| ------ | ------------------------- | ------------------------------------------------------------- |
| GET    | `/health`                 | Check API health                                              |
| POST   | `/generate`               | Generate a response using a configured model                  |
| GET    | `/analytics/summary`      | Retrieve aggregate usage metrics                              |
| GET    | `/analytics/models`       | Retrieve usage metrics grouped by model                       |
| POST   | `/benchmark/run`          | Generate benchmark responses using selected models            |
| POST   | `/benchmark/analyze`      | Calculate aggregate and category-level benchmark metrics      |
| POST   | `/benchmark/evaluate`     | Evaluate one generated response against its criteria          |
| POST   | `/benchmark/evaluate-all` | Evaluate multiple previously generated results in one request |

### Example Requests

Generate a response:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "groq/openai/gpt-oss-120b", "prompt": "Explain what an API is in one sentence."}'
```

Run the benchmark (the request body is a JSON list of model IDs):

```bash
curl -X POST http://127.0.0.1:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '["gemini/gemini-3.5-flash-lite", "groq/openai/gpt-oss-120b"]'
```

## Configuration

Settings are read from environment variables or a `.env` file (see `.env.example`).

| Variable                  | Default                          | Purpose                                                              |
| ------------------------- | -------------------------------- | -------------------------------------------------------------------- |
| `GEMINI_API_KEY`          | none                             | API key for Gemini models                                            |
| `GROQ_API_KEY`            | none                             | API key for Groq models                                              |
| `GEMINI_MODEL`            | `gemini/gemini-3.5-flash-lite`   | Gemini model ID                                                      |
| `GROQ_MODEL`              | `groq/openai/gpt-oss-120b`       | Groq model ID                                                        |
| `JUDGE_MODEL`             | falls back to `GEMINI_MODEL`     | Model used as the LLM judge. Prefer one that is not being benchmarked |
| `DATABASE_URL`            | `sqlite:///./router.db`          | Database used for usage records                                      |
| `REQUEST_TIMEOUT_SECONDS` | `60`                             | Timeout for each provider request                                    |
| `BENCHMARK_CONCURRENCY`   | `4`                              | Maximum concurrent provider calls during benchmarks and evaluation   |

If `JUDGE_MODEL` is a model other than the two defaults, the provider is taken from its ID prefix (for example `gemini/...` or `groq/...`). Add its token prices to `app/core/model_pricing.py` to track judge cost.

## Project Structure

```text
app/
├── core/
│   ├── config.py
│   ├── database.py
│   ├── dependencies.py
│   ├── model_pricing.py
│   └── model_registry.py
├── models/
│   └── usage_log.py
├── providers/
│   └── litellm_provider.py
├── schemas/
│   ├── llm.py
│   ├── analytics.py
│   ├── model_analytics.py
│   ├── benchmark.py
│   ├── benchmark_result.py
│   ├── benchmark_analysis.py
│   ├── benchmark_evaluation.py
│   └── benchmark_evaluation_run.py
├── services/
│   ├── model_service.py
│   ├── cost_service.py
│   ├── usage_service.py
│   ├── analytics_service.py
│   ├── benchmark_service.py
│   ├── benchmark_runner.py
│   ├── benchmark_analysis_service.py
│   ├── benchmark_evaluation_service.py
│   └── benchmark_evaluation_run_service.py
└── main.py

data/
└── benchmark_queries.json

scripts/
├── smoke_test_gemini.py
└── smoke_test_groq.py

tests/
├── test_main.py
├── test_schemas.py
├── test_config.py
├── test_litellm_provider.py
├── test_litellm_provider_timeout.py
├── test_model_service.py
├── test_model_registry.py
├── test_cost_service.py
├── test_usage_service.py
├── test_analytics_service.py
├── test_benchmark_dataset.py
├── test_benchmark_runner.py
├── test_benchmark_analysis_service.py
├── test_benchmark_evaluation.py
├── test_benchmark_evaluation_service.py
├── test_benchmark_evaluation_judge.py
├── test_benchmark_evaluation_run_service.py
└── test_concurrency.py

pyproject.toml
uv.lock
.python-version
.env.example
.gitignore
README.md
```

## Getting Started

### Prerequisites

* Python compatible with the version specified in `.python-version`
* [uv](https://docs.astral.sh/uv/)
* API keys for the providers you want to use (Gemini and/or Groq)

### Install Dependencies

Clone the repository, navigate to its root directory, and sync the dependencies:

```bash
uv sync
```

### Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

### Run the API

```bash
uv run uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs` and the health check at `http://127.0.0.1:8000/health`.

### Run Tests

```bash
uv run pytest
```

Tests mock all provider calls and do not require API keys.

### Verify Provider Connectivity

These scripts make real API calls and require valid keys in `.env`:

```bash
uv run python -m scripts.smoke_test_gemini
uv run python -m scripts.smoke_test_groq
```