from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteSystemDesignGoldenStore, seed_system_design_data
from ..implementation.sizing_calculator import SystemCapacityEstimation


@dataclass(frozen=True)
class SystemDesignEvalSummary:
    total_scenarios: int
    calculation_within_bounds: bool


class SystemDesignEvaluator:
    def __init__(self, db_path: str | Path = "system_design_eval.db") -> None:
        seed_system_design_data(db_path)
        self._store = SQLiteSystemDesignGoldenStore(db_path)

    def evaluate(self) -> SystemDesignEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return SystemDesignEvalSummary(0, False)

        all_ok = True
        for r in records:
            estimation = SystemCapacityEstimation(
                num_documents=r.doc_count,
                avg_tokens_per_doc=250,
                vector_dimension=r.dimension,
                target_qps=r.target_qps,
            )
            ram_gb = estimation.compute_vector_memory_gb()
            if not (r.min_expected_ram_gb <= ram_gb <= r.max_expected_ram_gb):
                all_ok = False

        total = len(records)
        return SystemDesignEvalSummary(
            total_scenarios=total,
            calculation_within_bounds=all_ok,
        )


if __name__ == "__main__":
    evaluator = SystemDesignEvaluator()
    summary = evaluator.evaluate()
    print(f"System Design Evaluation: {summary}")
