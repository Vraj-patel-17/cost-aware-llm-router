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

* **Language:** Python
* **API:** FastAPI
* **Validation:** Pydantic
* **LLM integration:** LiteLLM
* **Configuration:** Pydantic Settings
* **Testing:** pytest, pytest-asyncio
* **Dependency management:** uv

Planned technologies include SQLite for persistence, scikit-learn for learned routing, and Streamlit for benchmark visualization.

### Current Implementation

* FastAPI application with health-check, model-generation, analytics, and benchmark endpoints.
* Centralized configuration using Pydantic Settings.
* Configurable model registry mapping supported model IDs to providers and API keys.
* Asynchronous multi-model execution through LiteLLM.
* Gemini and Groq integrations verified through real API calls.
* Normalized model responses containing generated text, provider, token usage, latency, and estimated inference cost.
* Provider-independent cost estimation using configured input and output token prices.
* SQLite persistence for model usage records, including model, provider, token usage, latency, estimated cost, and timestamp.
* Aggregate usage analytics through `GET /analytics/summary`.
* Per-model usage analytics through `GET /analytics/models`.
* A benchmark dataset containing 20 queries across four categories: simple questions, reasoning, coding, and information extraction.
* Benchmark execution through `POST /benchmark/run`, with support for evaluating selected models against the same query dataset.
* Benchmark analysis through `POST /benchmark/analyze`, calculating success rates, average latency, and estimated costs overall and by category.
* LLM-as-judge quality evaluation through `POST /benchmark/evaluate`, using evaluation criteria defined for each benchmark query.
* Batch quality evaluation through `POST /benchmark/evaluate-all`, evaluating previously generated benchmark results and preserving generation failures and individual evaluation errors.
* Structured evaluation results containing quality scores, pass/fail outcomes, criterion-level assessments, and explanations.
* Automated tests covering API behavior, model registration, cost calculations, usage persistence, analytics aggregation, benchmark execution, analysis, and quality evaluation.
* Mocked provider tests that do not require live LLM API calls.

### Current Limitations

* Model selection is explicit; automatic cost-aware routing has not yet been implemented.
* Only Gemini and Groq integrations have been verified against real APIs.
* Configured token prices are estimates and must be rechecked when provider pricing changes.
* Estimated costs may differ from actual charges because of free-tier allowances, discounts, and provider-specific billing rules.
* Usage records are stored locally in SQLite.
* The benchmark dataset currently contains 20 queries, which is insufficient for strong general conclusions about model quality.
* Quality evaluation relies on an LLM judge. Its scores may be subjective or inconsistent and should not be treated as ground truth.
* Batch evaluation processes previously generated results; generation and evaluation are separate operations.
* Evaluator usage costs are not yet tracked separately from generation costs.
* Benchmark results and quality evaluations are returned by the API but are not yet persisted as benchmark runs.
* Response caching, automatic fallback, a learned routing strategy, and a Streamlit dashboard have not yet been implemented.

### Evaluation Plan

### Evaluation Strategy

The project aims to compare cost-aware routing against an always-use-the-strongest-model baseline.

The current benchmark dataset contains 20 queries distributed across four categories:

* **Simple:** Straightforward questions.
* **Reasoning:** Questions requiring logical reasoning.
* **Coding:** Programming-related tasks.
* **Extraction:** Tasks that extract specific information from supplied text.

Each query includes evaluation criteria used to assess generated responses.

#### Evaluation Workflow

1. **Generate:** Run selected models against the same benchmark queries using `POST /benchmark/run`.
2. **Analyze:** Calculate generation success rates, latency, and estimated costs using `POST /benchmark/analyze`.
3. **Evaluate quality:** Assess individual responses against their criteria using `POST /benchmark/evaluate`.
4. **Batch evaluation:** Evaluate previously generated results using `POST /benchmark/evaluate-all`.
5. **Compare:** Use generation costs, latency, and quality outcomes to understand the trade-offs between models.

#### Key Metrics

* **Response quality:** LLM-judge scores and criterion-level pass/fail results.
* **Success rate:** Percentage of benchmark generations completed successfully.
* **Inference cost:** Estimated generation cost per request and across the benchmark.
* **Latency:** Time required to generate responses.
* **Cost savings:** Reduction in estimated costs relative to the baseline.
* **Quality retention:** Whether a cheaper model maintains acceptable response quality compared with the baseline.

The current evaluation infrastructure provides the foundation for comparing models. Establishing a reliable baseline and implementing cost-aware routing remain future milestones. The 20-query dataset is an initial development benchmark, not sufficient evidence of general model performance.


### Project Status

**Current phase:** Benchmarking and quality evaluation.

The project currently supports asynchronous execution across registered models, estimated cost calculation, SQLite usage logging, usage analytics, benchmark execution, aggregate benchmark analysis, and individual and batch quality evaluation.

The next major milestone is to implement and evaluate a cost-aware routing strategy using measured quality, latency, and cost rather than assuming that cheaper models are sufficient for every query.

### Roadmap

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
* [ ] Track evaluator token usage and costs separately
* [ ] Persist benchmark runs and evaluation results
* [ ] Expand and validate the benchmark dataset
* [ ] Establish an always-use-the-strongest-model baseline
* [ ] Implement rule-based cost-aware routing
* [ ] Compare routing strategies against the baseline
* [ ] Evaluate a learned routing strategy
* [ ] Add response caching and fallback handling
* [ ] Build a Streamlit analytics dashboard
* [ ] Improve deployment and CI/CD


## API Endpoints

| Method | Endpoint             | Purpose                                      |
| ------ | -------------------- | -------------------------------------------- |
| GET    | `/health`            | Check API health                             |
| POST   | `/generate`          | Generate a response using a configured model |
| GET    | `/analytics/summary` | Retrieve aggregate usage metrics             |
| GET    | `/analytics/models`  | Retrieve usage metrics grouped by model      |
| POST   | `/benchmark/run`          | Generate benchmark responses using selected models            |
| POST   | `/benchmark/analyze`      | Calculate aggregate and category-level benchmark metrics      |
| POST   | `/benchmark/evaluate`     | Evaluate one generated response against its criteria          |
| POST   | `/benchmark/evaluate-all` | Evaluate multiple previously generated results in one request |

## Project Structure

```text
app/
├── core/
│   ├── config.py
│   ├── database.py
│   ├── dependencies.py
│   └── model_registry.py
├── models/
│   └── usage_log.py
├── providers/
│   └── litellm_provider.py
├── schemas/
│   ├── llm.py
│   ├── analytics.py
│   └── model_analytics.py
├── services/
│   ├── model_service.py
│   ├── cost_service.py
│   ├── usage_service.py
│   └── analytics_service.py
└── main.py

scripts/
├── smoke_test_gemini.py
└── smoke_test_groq.py

tests/
├── test_main.py
├── test_schemas.py
├── test_litellm_provider.py
├── test_model_service.py
├── test_config.py
├── test_model_registry.py
├── test_cost_service.py
├── test_usage_service.py
└── test_analytics_service.py

pyproject.toml
uv.lock
.python-version
.env.example
.gitignore
README.md
```


Some directories are reserved for components that will be implemented in later phases.

## Getting Started

### Prerequisites

* Python compatible with the version specified in `.python-version`
* [uv](https://docs.astral.sh/uv/)

### Install Dependencies

Clone the repository and navigate

