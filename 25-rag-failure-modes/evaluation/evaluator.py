from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteFailureGoldenStore, seed_failure_data
from ..implementation.failure_analyzer import RAGFailureDiagnosticsEngine


@dataclass(frozen=True)
class FailureDiagnosticsEvalMetrics:
    total_cases: int
    diagnostic_accuracy: float


class FailureDiagnosticsEvaluator:
    def __init__(self, db_path: str | Path = "failure_eval.db") -> None:
        seed_failure_data(db_path)
        self._store = SQLiteFailureGoldenStore(db_path)
        self._engine = RAGFailureDiagnosticsEngine()

    def evaluate(self) -> FailureDiagnosticsEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return FailureDiagnosticsEvalMetrics(0, 0.0)

        correct_diagnoses = 0
        for r in records:
            diag = self._engine.diagnose(
                is_in_corpus=r.in_corpus,
                is_retrieved_in_k=r.retrieved_k,
                is_passed_to_llm=r.passed_to_llm,
                is_correctly_generated=r.generated_ok,
            )
            if diag.detected_mode.value == r.expected_failure_mode:
                correct_diagnoses += 1

        total = len(records)
        return FailureDiagnosticsEvalMetrics(
            total_cases=total,
            diagnostic_accuracy=correct_diagnoses / total,
        )


if __name__ == "__main__":
    evaluator = FailureDiagnosticsEvaluator()
    metrics = evaluator.evaluate()
    print(f"Failure Diagnostics Evaluation: {metrics}")
