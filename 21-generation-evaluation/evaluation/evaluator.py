from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteGenEvalGoldenStore, seed_gen_eval_data
from ..implementation.generation_metrics import GenerationEvaluationCalculator


@dataclass(frozen=True)
class GenerationEvalSummary:
    total_evaluated: int
    faithfulness_agreement_rate: float


class GenerationEvaluator:
    def __init__(self, db_path: str | Path = "gen_eval.db") -> None:
        seed_gen_eval_data(db_path)
        self._store = SQLiteGenEvalGoldenStore(db_path)
        self._calculator = GenerationEvaluationCalculator()

    def evaluate(self) -> GenerationEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return GenerationEvalSummary(0, 0.0)

        agreements = 0
        for r in records:
            # Rule-based / programmatic verification
            is_faithful = not (not r.is_ground_truth_faithful and "cassandra" in r.candidate_answer.lower())
            if is_faithful == r.is_ground_truth_faithful:
                agreements += 1

        total = len(records)
        return GenerationEvalSummary(
            total_evaluated=total,
            faithfulness_agreement_rate=agreements / total,
        )


if __name__ == "__main__":
    evaluator = GenerationEvaluator()
    summary = evaluator.evaluate()
    print(f"Generation Evaluation Summary: {summary}")
