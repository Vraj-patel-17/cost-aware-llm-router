import uuid

from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session

from app.models.benchmark_run import BenchmarkEvaluationLog,BenchmarkResultLog,BenchmarkRun
from app.schemas.benchmark_evaluation import BenchmarkEvaluation
from app.schemas.benchmark_result import BenchmarkResult
from app.schemas.benchmark_run import BenchmarkRunSummary

UNKNOWN_JUDGE = "unknown"


class BenchmarkRunService:
    def __init__(self, db: Session):
        self.db = db

    def create_run(
        self,
        model_names: list[str],
        dataset_hash: str,
        results: list[BenchmarkResult],
    ) -> BenchmarkRun:
        run = BenchmarkRun(
            id=str(uuid.uuid4()),
            model_names=list(model_names),
            dataset_hash=dataset_hash,
        )
        self.db.add(run)

        self.db.add_all(
            [
                BenchmarkResultLog(
                    run_id=run.id,
                    query_id=result.query_id,
                    category=result.category,
                    model=result.model,
                    response_text=result.response_text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    estimated_cost_usd=result.estimated_cost_usd,
                    success=result.success,
                    error=result.error,
                )
                for result in results
            ]
        )

        self.db.commit()
        self.db.refresh(run)

        return run

    def get_run(self, run_id: str) -> BenchmarkRun | None:
        return self.db.get(BenchmarkRun, run_id)

    def list_runs(self) -> list[BenchmarkRun]:
        statement = select(BenchmarkRun).order_by(
            BenchmarkRun.created_at.desc()
        )

        return list(self.db.scalars(statement).all())

    def get_results(self, run_id: str) -> list[BenchmarkResult]:
        statement = (
            select(BenchmarkResultLog)
            .where(BenchmarkResultLog.run_id == run_id)
            .order_by(BenchmarkResultLog.id)
        )

        return [
            BenchmarkResult(
                query_id=row.query_id,
                category=row.category,
                model=row.model,
                response_text=row.response_text,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                latency_ms=row.latency_ms,
                estimated_cost_usd=row.estimated_cost_usd,
                success=row.success,
                error=row.error,
            )
            for row in self.db.scalars(statement).all()
        ]

    def build_summary(self, run: BenchmarkRun) -> BenchmarkRunSummary:
        return self.build_summaries([run])[0]

    def build_summaries(
        self,
        runs: list[BenchmarkRun],
    ) -> list[BenchmarkRunSummary]:
        if not runs:
            return []

        counts = self._result_counts([run.id for run in runs])
        summaries = []

        for run in runs:
            total, successful = counts.get(run.id, (0, 0))

            summaries.append(
                BenchmarkRunSummary(
                    run_id=run.id,
                    created_at=run.created_at,
                    model_names=list(run.model_names),
                    dataset_hash=run.dataset_hash,
                    total_results=total,
                    successful_results=successful,
                    failed_results=total - successful,
                )
            )

        return summaries

    def _result_counts(
        self,
        run_ids: list[str],
    ) -> dict[str, tuple[int, int]]:
        statement = (
            select(
                BenchmarkResultLog.run_id,
                func.count(BenchmarkResultLog.id),
                func.sum(
                    case(
                        (BenchmarkResultLog.success.is_(True), 1),
                        else_=0,
                    )
                ),
            )
            .where(BenchmarkResultLog.run_id.in_(run_ids))
            .group_by(BenchmarkResultLog.run_id)
        )

        return {
            run_id: (total, successful or 0)
            for run_id, total, successful in self.db.execute(statement).all()
        }

    def save_evaluations(
        self,
        run_id: str,
        evaluations: list[BenchmarkEvaluation],
    ) -> None:
        for evaluation in evaluations:
            judge_model = evaluation.judge_model or UNKNOWN_JUDGE

            self.db.execute(
                delete(BenchmarkEvaluationLog).where(
                    BenchmarkEvaluationLog.run_id == run_id,
                    BenchmarkEvaluationLog.query_id == evaluation.query_id,
                    BenchmarkEvaluationLog.model == evaluation.model,
                    BenchmarkEvaluationLog.judge_model == judge_model,
                )
            )

            self.db.add(
                BenchmarkEvaluationLog(
                    run_id=run_id,
                    query_id=evaluation.query_id,
                    model=evaluation.model,
                    judge_model=judge_model,
                    score=evaluation.score,
                    passed=evaluation.passed,
                    reasoning=evaluation.reasoning,
                    criteria_evaluations=[
                        item.model_dump()
                        for item in evaluation.criteria_evaluations
                    ],
                    judge_input_tokens=evaluation.judge_input_tokens,
                    judge_output_tokens=evaluation.judge_output_tokens,
                    judge_cost_usd=evaluation.judge_cost_usd,
                )
            )

        self.db.commit()

    def get_evaluations(
        self,
        run_id: str,
        judge_model: str | None = None,
    ) -> list[BenchmarkEvaluation]:
        statement = (
            select(BenchmarkEvaluationLog)
            .where(BenchmarkEvaluationLog.run_id == run_id)
            .order_by(BenchmarkEvaluationLog.id)
        )

        if judge_model is not None:
            statement = statement.where(
                BenchmarkEvaluationLog.judge_model == judge_model
            )

        return [
            BenchmarkEvaluation(
                query_id=row.query_id,
                model=row.model,
                score=row.score,
                passed=row.passed,
                reasoning=row.reasoning,
                criteria_evaluations=row.criteria_evaluations,
                judge_model=row.judge_model,
                judge_input_tokens=row.judge_input_tokens,
                judge_output_tokens=row.judge_output_tokens,
                judge_cost_usd=row.judge_cost_usd,
            )
            for row in self.db.scalars(statement).all()
        ]

    def get_evaluated_keys(
        self,
        run_id: str,
        judge_model: str,
    ) -> set[tuple[str, str]]:
        statement = select(
            BenchmarkEvaluationLog.query_id,
            BenchmarkEvaluationLog.model,
        ).where(
            BenchmarkEvaluationLog.run_id == run_id,
            BenchmarkEvaluationLog.judge_model == judge_model,
        )

        return {
            (query_id, model)
            for query_id, model in self.db.execute(statement).all()
        }