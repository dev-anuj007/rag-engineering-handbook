from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteDatasetStore, seed_golden_dataset_store
from ..implementation.dataset_generator import SyntheticQAGenerator


@dataclass(frozen=True)
class DatasetSplitSummary:
    test_set_size: int
    train_set_size: int
    synthetic_generation_verified: bool


class DatasetEvaluator:
    def __init__(self, db_path: str | Path = "golden_benchmark_store.db") -> None:
        seed_golden_dataset_store(db_path)
        self._store = SQLiteDatasetStore(db_path)
        self._generator = SyntheticQAGenerator()

    def evaluate(self) -> DatasetSplitSummary:
        test_entries = self._store.fetch_by_split("test")
        train_entries = self._store.fetch_by_split("train")

        sample_chunk = "Backup recovery time objective (RTO) is guaranteed at 4 hours."
        qa_pair = self._generator.generate_from_chunk(sample_chunk)
        gen_ok = len(qa_pair.query) > 0 and len(qa_pair.ground_truth_answer) > 0

        return DatasetSplitSummary(
            test_set_size=len(test_entries),
            train_set_size=len(train_entries),
            synthetic_generation_verified=gen_ok,
        )


if __name__ == "__main__":
    evaluator = DatasetEvaluator()
    summary = evaluator.evaluate()
    print(f"Evaluation Dataset Summary: {summary}")
