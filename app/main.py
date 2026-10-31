from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.models as _models  # noqa: F401  (registers tables for create_all)
from app.core.database import Base, engine
from app.routers import (
    analytics,
    benchmark,
    benchmark_runs,
    generate,
    health,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Cost-Aware LLM Router",
    description="A system for routing queries to cost-effective language models.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(generate.router)
app.include_router(analytics.router)
app.include_router(benchmark.router)
app.include_router(benchmark_runs.router)