from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteCRAGGoldenStore, seed_crag_data
from ..implementation.crag_pipeline import (
    CorrectiveRAGPipeline,
    RetrievalQuality,
)


@dataclass(frozen=True)
class CRAGEvalMetrics:
    total_queries: int
    correct_grade_rate: float


class CRAGEvaluator:
    def __init__(self, db_path: str | Path = "crag_eval.db") -> None:
        seed_crag_data(db_path)
        self._store = SQLiteCRAGGoldenStore(db_path)
        self._pipeline = CorrectiveRAGPipeline()

    def evaluate(self) -> CRAGEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return CRAGEvalMetrics(0, 0.0)

        docs = [
            Document(text="Internal API rate limit is 10,000 requests per minute per tenant.")
        ]
        self._pipeline.index(docs)

        graded_correct = 0
        for r in records:
            ans, quality = self._pipeline.run(r.query)
            if quality.value == r.expected_quality:
                graded_correct += 1

        total = len(records)
        return CRAGEvalMetrics(
            total_queries=total,
            correct_grade_rate=graded_correct / total,
        )


if __name__ == "__main__":
    evaluator = CRAGEvaluator()
    metrics = evaluator.evaluate()
    print(f"CRAG Evaluation: {metrics}")
