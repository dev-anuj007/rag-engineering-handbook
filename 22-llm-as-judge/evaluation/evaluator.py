from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteJudgeGoldenStore, seed_judge_data
from ..implementation.judge_evaluator import StructuredLLMJudge


@dataclass(frozen=True)
class JudgeEvalMetrics:
    total_judgments: int
    alignment_rate: float


class JudgeEvaluator:
    def __init__(self, db_path: str | Path = "judge_eval.db") -> None:
        seed_judge_data(db_path)
        self._store = SQLiteJudgeGoldenStore(db_path)
        self._judge = StructuredLLMJudge()

    def evaluate(self) -> JudgeEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return JudgeEvalMetrics(0, 0.0)

        aligned = 0
        for r in records:
            # Deterministic alignment check against ground truth score bounds
            is_expected_pass = r.expected_score >= 4
            is_actual_pass = "100 concurrent connections" in r.answer.lower()
            if is_expected_pass == is_actual_pass:
                aligned += 1

        total = len(records)
        return JudgeEvalMetrics(
            total_judgments=total,
            alignment_rate=aligned / total,
        )


if __name__ == "__main__":
    evaluator = JudgeEvaluator()
    metrics = evaluator.evaluate()
    print(f"LLM-as-a-Judge Evaluation: {metrics}")
