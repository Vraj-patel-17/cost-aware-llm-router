# Cost-Aware LLM Router

A backend system that routes incoming queries to cost-effective language models while maintaining an acceptable level of response quality.

## Problem

Using the most powerful language model for every request can increase inference costs unnecessarily. Many requests may be handled adequately by cheaper models.

This project explores whether intelligent model selection can reduce costs without significantly compromising response quality.

## Goals

* Integrate multiple language models through a common interface.
* Establish an always-use-the-strongest-model baseline.
* Develop routing strategies that select models based on query characteristics.
* Measure model costs and response quality.
* Compare routing strategies against the baseline using a reproducible evaluation dataset.
* Evaluate the trade-off between cost savings and answer quality.

## Technology Stack

* Python
* FastAPI
* Pydantic
* LiteLLM
* SQLite
* pytest
* scikit-learn
* Streamlit

Some components will be introduced as the project develops.

## Development Setup

Install [uv](https://docs.astral.sh/uv/), then clone the repository and navigate to its root directory.

Sync the project dependencies:

```bash
uv sync
```

Start the development server:

```bash
uv run uvicorn app.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

The health endpoint is available at `http://127.0.0.1:8000/health`.

## Tests

Run the test suite:

```bash
uv run pytest
```

## Project Status

The initial FastAPI application and health endpoint are implemented. Model integration, routing strategies, evaluation, and benchmarking will be developed in subsequent phases.
