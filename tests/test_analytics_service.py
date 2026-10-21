from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.usage_log import UsageLog
from app.services.analytics_service import AnalyticsService


def test_get_summary_returns_usage_totals():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    try:
        with Session(engine) as db:
            db.add_all(
                [
                    UsageLog(
                        model="gemini/test-model",
                        provider="gemini",
                        input_tokens=100,
                        output_tokens=50,
                        latency_ms=200.0,
                        estimated_cost_usd=0.001,
                    ),
                    UsageLog(
                        model="groq/test-model",
                        provider="groq",
                        input_tokens=200,
                        output_tokens=100,
                        latency_ms=400.0,
                        estimated_cost_usd=0.002,
                    ),
                ]
            )
            db.commit()

            summary = AnalyticsService(db).get_summary()

            assert summary["total_requests"] == 2
            assert summary["total_input_tokens"] == 300
            assert summary["total_output_tokens"] == 150
            assert abs(summary["total_estimated_cost_usd"] - 0.003) < 1e-9
            assert summary["average_latency_ms"] == 300.0
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()