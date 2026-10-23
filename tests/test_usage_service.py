import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.usage_log import UsageLog
from app.schemas.llm import LLMResponse
from app.services.usage_service import UsageService
from app.services.analytics_service import AnalyticsService


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        yield session

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_log_usage_saves_record(db_session: Session):
    response = LLMResponse(
        text="Test response",
        model="gemini/gemini-3.5-flash-lite",
        provider="gemini",
        input_tokens=10,
        output_tokens=20,
        latency_ms=150.0,
        estimated_cost_usd=0.001,
    )

    usage_service = UsageService(db_session)
    saved_record = usage_service.log_usage(response)

    assert saved_record.id is not None

    records = db_session.scalars(
        select(UsageLog)
    ).all()

    assert len(records) == 1
    assert records[0].model == response.model
    assert records[0].provider == response.provider
    assert records[0].input_tokens == 10
    assert records[0].output_tokens == 20
    assert records[0].latency_ms == 150.0
    assert records[0].estimated_cost_usd == 0.001


def test_usage_logging_flows_into_analytics(db_session: Session):
    responses = [
        LLMResponse(
            text="Gemini response",
            model="gemini/gemini-3.5-flash-lite",
            provider="gemini",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=200.0,
            estimated_cost_usd=0.00155,
        ),
        LLMResponse(
            text="Groq response",
            model="groq/openai/gpt-oss-120b",
            provider="groq",
            input_tokens=2000,
            output_tokens=1000,
            latency_ms=100.0,
            estimated_cost_usd=0.0009,
        ),
    ]

    usage_service = UsageService(db_session)

    for response in responses:
        usage_service.log_usage(response)

    analytics_service = AnalyticsService(db_session)

    summary = analytics_service.get_summary()

    assert summary["total_requests"] == 2
    assert summary["total_input_tokens"] == 3000
    assert summary["total_output_tokens"] == 1500
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.00245)
    assert summary["average_latency_ms"] == pytest.approx(150.0)

    model_summaries = analytics_service.get_model_summary()

    assert len(model_summaries) == 2

    summaries_by_model = {
        item["model"]: item for item in model_summaries
    }

    assert (
        summaries_by_model["gemini/gemini-3.5-flash-lite"][
            "total_estimated_cost_usd"
        ]
        == pytest.approx(0.00155)
    )

    assert (
        summaries_by_model["groq/openai/gpt-oss-120b"][
            "total_estimated_cost_usd"
        ]
        == pytest.approx(0.0009)
    )