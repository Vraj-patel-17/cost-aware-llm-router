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

## Current Implementation

### Implemented

* FastAPI application with a health-check endpoint.
* Pydantic schema for normalized LLM responses, including token usage and latency.
* LiteLLM provider layer for asynchronous model execution.
* Model service that delegates generation requests to the provider.
* Unit tests using mocked model responses to verify provider and service behavior.
* Centralized application configuration using Pydantic Settings, with environment-based overrides and sensible defaults.
**Note:** Model execution is currently tested using mocks. Real API-provider integration and benchmark evaluation are planned for later phases.


### In Progress

* Model execution service.
* Integration with real model providers.
* Multi-model evaluation and benchmarking.
* Rule-based and learned routing strategies.

The provider tests currently use mocked responses. Successful tests do not yet establish that real external model calls work.

## Project Structure

```text
app/
├── core/
│   └── config.py
├── providers/
│   └── litellm_provider.py
├── schemas/
│   └── llm.py
├── services/
│   └── model_service.py
└── main.py
tests/
├── test_main.py
├── test_schemas.py
├── test_litellm_provider.py
├── test_model_service.py
└── test_config.py
datasets/
scripts/
pyproject.toml
uv.lock
```

Some directories are reserved for components that will be implemented in later phases.

## Getting Started

### Prerequisites

* Python compatible with the version specified in `.python-version`
* [uv](https://docs.astral.sh/uv/)

### Install Dependencies

Clone the repository and navigate
