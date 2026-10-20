import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.usage_log import UsageLog
from app.schemas.llm import LLMResponse
from app.services.usage_service import UsageService


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