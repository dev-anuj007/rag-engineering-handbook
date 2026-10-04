from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteFrameworkGoldenStore, seed_framework_data
from ..implementation.framework_adapter import LlamaIndexFrameworkAdapter


@dataclass(frozen=True)
class FrameworkAdapterEvalSummary:
    total_samples: int
    adapter_execution_success_rate: float


class FrameworkAdapterEvaluator:
    def __init__(self, db_path: str | Path = "framework_eval.db") -> None:
        seed_framework_data(db_path)
        self._store = SQLiteFrameworkGoldenStore(db_path)
        self._adapter = LlamaIndexFrameworkAdapter()

    def evaluate(self) -> FrameworkAdapterEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return FrameworkAdapterEvalSummary(0, 0.0)

        success = 0
        for r in records:
            score = self._adapter.evaluate(r.query, r.response, [r.context])
            if score.faithfulness_score > 0.0 and score.relevancy_score > 0.0:
                success += 1

        total = len(records)
        return FrameworkAdapterEvalSummary(
            total_samples=total,
            adapter_execution_success_rate=success / total,
        )


if __name__ == "__main__":
    evaluator = FrameworkAdapterEvaluator()
    metrics = evaluator.evaluate()
    print(f"Framework Adapter Evaluation: {metrics}")
