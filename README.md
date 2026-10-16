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

* FastAPI application with a health-check endpoint.
* Centralized configuration using Pydantic Settings.
* Pydantic schema for normalized LLM responses, including token usage and latency.
* Asynchronous model execution through LiteLLM.
* Model service that delegates generation requests to the provider.
* Gemini API integration verified with a successful real model call.
* Groq API integration verified with a real model call.
* Automated tests using mocked responses to verify provider behavior, API-key forwarding, configuration, and service delegation.

### Current Limitations
* Only Gemini and Groq integrations have been verified against real APIs.
* Automated tests use mocked model responses and do not require API credentials.
* The routing algorithm and cost-aware model selection are not implemented yet.
* Cost tracking, benchmark evaluation, caching, and fallback handling remain planned features.
### Evaluation Plan

The project's effectiveness will be evaluated against an always-use-the-strongest-model baseline.

The planned benchmark will cover different query categories, including simple questions, reasoning, coding, and information extraction.

Key evaluation metrics:

Response quality: Correctness and task-specific answer quality.
Inference cost: Estimated cost per request and across the evaluation dataset.
Latency: Time required to generate a response.
Cost savings: Reduction in estimated inference costs relative to the baseline.
Quality retention: How much response quality is preserved after introducing routing.

The objective is to quantify the trade-off between inference cost and response quality rather than assuming that a cheaper model is always sufficient.

Project Status

Current phase: Multi-provider model execution.

Gemini and Groq have been integrated through LiteLLM, and their real API calls have been verified. Automated tests cover the existing provider and service behavior.

The next milestones are to establish the model execution interface, implement routing strategies, and build a reproducible benchmark for evaluating cost and quality.
### In Progress

* Model execution service.
* Integration with real model providers.
* Multi-model evaluation and benchmarking.
* Rule-based and learned routing strategies.

The provider tests currently use mocked responses. Successful tests do not yet establish that real external model calls work.

## Project Structure

```text
Project Structure
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

scripts/
├── __init__.py
├── smoke_test_gemini.py
└── smoke_test_groq.py

tests/
├── __init__.py
├── test_main.py
├── test_schemas.py
├── test_litellm_provider.py
├── test_model_service.py
└── test_config.py

datasets/

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

