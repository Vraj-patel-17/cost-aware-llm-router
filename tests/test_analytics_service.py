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

def test_get_model_summary_groups_by_model():
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
                        model="gemini/test-model",
                        provider="gemini",
                        input_tokens=200,
                        output_tokens=100,
                        latency_ms=400.0,
                        estimated_cost_usd=0.002,
                    ),
                    UsageLog(
                        model="groq/test-model",
                        provider="groq",
                        input_tokens=50,
                        output_tokens=25,
                        latency_ms=100.0,
                        estimated_cost_usd=None,
                    ),
                ]
            )
            db.commit()

            results = AnalyticsService(db).get_model_summary()

            assert len(results) == 2

            gemini = next(
                row for row in results
                if row["provider"] == "gemini"
            )
            assert gemini["total_requests"] == 2
            assert gemini["total_input_tokens"] == 300
            assert gemini["total_output_tokens"] == 150
            assert abs(
                gemini["total_estimated_cost_usd"] - 0.003
            ) < 1e-9
            assert gemini["average_latency_ms"] == 300.0

            groq = next(
                row for row in results
                if row["provider"] == "groq"
            )
            assert groq["total_requests"] == 1
            assert groq["total_input_tokens"] == 50
            assert groq["total_output_tokens"] == 25
            assert groq["total_estimated_cost_usd"] is None
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()