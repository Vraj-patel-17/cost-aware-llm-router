from datetime import datetime
from sqlalchemy import JSON,Boolean,DateTime,Float,ForeignKey,Integer,String,Text,UniqueConstraint,func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base

class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    model_names: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
    )
    dataset_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

class BenchmarkResultLog(Base):
    __tablename__ = "benchmark_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "query_id",
            "model",
            name="uq_benchmark_result",
        ),
    )
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runs.id"),
        index=True,
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    response_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    output_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    latency_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    estimated_cost_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

class BenchmarkEvaluationLog(Base):
    __tablename__ = "benchmark_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "query_id",
            "model",
            "judge_model",
            name="uq_benchmark_evaluation",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runs.id"),
        index=True,
        nullable=False,
    )
    query_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    judge_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    reasoning: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    criteria_evaluations: Mapped[list[dict]] = mapped_column(
        JSON,
        nullable=False,
    )
    judge_input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    judge_output_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    judge_cost_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )