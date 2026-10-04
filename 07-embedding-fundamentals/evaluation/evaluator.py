from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteVectorMetricGoldenStore, seed_vector_metric_data
from ..implementation.vector_math import (
    CosineSimilarity,
    EmbeddingVectorStoreManager,
)


@dataclass(frozen=True)
class VectorMathEvalMetrics:
    total_samples: int
    metric_precision_rate: float


class VectorMathEvaluator:
    def __init__(self, db_path: str | Path = "vector_metric_eval.db") -> None:
        seed_vector_metric_data(db_path)
        self._store = SQLiteVectorMetricGoldenStore(db_path)
        self._cosine_calc = CosineSimilarity()
        self._manager = EmbeddingVectorStoreManager()

    def evaluate(self) -> VectorMathEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return VectorMathEvalMetrics(0, 0.0)

        correct_count = 0

        for r in records:
            vec_a = [float(x.strip()) for x in r.vector_a_csv.split(",")]
            vec_b = [float(x.strip()) for x in r.vector_b_csv.split(",")]

            computed = self._cosine_calc.calculate(vec_a, vec_b)
            if math.isclose(computed, r.expected_cosine, abs_tol=1e-4):
                correct_count += 1

        total = len(records)
        return VectorMathEvalMetrics(
            total_samples=total,
            metric_precision_rate=correct_count / total,
        )


if __name__ == "__main__":
    evaluator = VectorMathEvaluator()
    metrics = evaluator.evaluate()
    print(f"Vector Math Evaluation: {metrics}")
