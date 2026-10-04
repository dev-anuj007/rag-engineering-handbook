from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteBatchGoldenStore, seed_batch_data
from ..implementation.async_batch_optimizer import (
    AsyncBatchProcessor,
    BatchOptimizationConfig,
)


@dataclass(frozen=True)
class BatchEvalSummary:
    total_documents: int
    batch_throughput_success: bool


class PerformanceEvaluator:
    def __init__(self, db_path: str | Path = "batch_eval.db") -> None:
        seed_batch_data(db_path)
        self._store = SQLiteBatchGoldenStore(db_path)
        self._processor = AsyncBatchProcessor(
            config=BatchOptimizationConfig(batch_size=5, max_concurrency=4)
        )

    def evaluate(self) -> BatchEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return BatchEvalSummary(0, False)

        docs = [Document(text=r.doc_text, metadata={"id": r.id}) for r in records]
        inserted = self._processor.batch_insert(docs)

        # Run async retrieval test
        queries = ["Benchmark document 1", "Benchmark document 5"]
        results = asyncio.run(self._processor.async_multi_retrieve(queries))

        return BatchEvalSummary(
            total_documents=inserted,
            batch_throughput_success=(len(results) == len(queries)),
        )


if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    summary = evaluator.evaluate()
    print(f"Performance Optimization Evaluation: {summary}")
