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

* FastAPI application with health-check and model-generation endpoints.
* Centralized configuration using Pydantic Settings.
* Model registry mapping supported model IDs to providers and API keys.
* Explicit model selection through registered model IDs.
* Asynchronous model execution through LiteLLM.
* Normalized responses containing model output, provider, token usage, latency, and estimated cost when available.
* Gemini and Groq integrations verified through real API calls.
* Provider-independent cost estimation using configurable input and output token prices.
* Configured and tested pricing entries for Gemini 3.5 Flash Lite and Groq GPT-OSS 120B.
* SQLite persistence for model usage records, including model, provider, token usage, latency, estimated cost, and timestamp.
* Aggregate usage analytics through `GET /analytics/summary`.
* Per-model usage analytics through `GET /analytics/models`.
* Verified usage persistence and data flow into aggregate and per-model analytics using an isolated in-memory SQLite database.
* Automated tests covering API behavior, model registration, cost calculations, usage persistence, and analytics aggregation.

### Current Limitations

* Model selection is explicit; automatic cost-aware routing has not been implemented.
* Only Gemini and Groq integrations have been verified against real APIs.
* Automated tests use mocked responses where appropriate and do not require live API credentials.
* Configured prices are estimates based on the supplied published rates and should be rechecked when provider pricing changes.
* Estimated costs may differ from actual provider charges due to free-tier allowances, discounts, and billing rules.
* Usage data is stored locally in SQLite.
* A reproducible benchmark, response-quality evaluation, caching, and fallback handling have not yet been implemented.

### Evaluation Plan

The project will be evaluated against an always-use-the-strongest-model baseline.

The benchmark will cover different query categories, including simple questions, reasoning, coding, and information extraction.

Key evaluation metrics:

* **Response quality:** Correctness and task-specific answer quality.
* **Inference cost:** Estimated cost per request and across the evaluation dataset.
* **Latency:** Time required to generate a response.
* **Cost savings:** Reduction in estimated inference costs relative to the baseline.
* **Quality retention:** How much response quality is preserved after introducing routing.

The objective is to quantify the trade-off between inference cost and response quality rather than assume that cheaper models are always sufficient.

### Project Status

**Current phase:** Usage tracking and analytics.

The project currently supports model execution through LiteLLM, a configurable model registry, cost-estimation logic, SQLite usage logging, and aggregate and per-model analytics.

The next milestones are to configure verified model prices, build a reproducible benchmark, and implement and evaluate cost-aware routing strategies.

### Roadmap

* [x] FastAPI foundation and health endpoint
* [x] Multi-provider model execution through LiteLLM
* [x] Configurable model registry
* [x] Provider-independent cost-estimation logic
* [x] Persistent usage logging with SQLite
* [x] Aggregate usage analytics
* [x] Per-model usage analytics
* [ ] Configure verified pricing for supported models
* [ ] Build a reproducible benchmark dataset
* [ ] Establish an always-use-the-strongest-model baseline
* [ ] Implement rule-based routing
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

