from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteIngestionBenchmarkStore, seed_ingestion_data
from ..implementation.ingestion_manager import (
    DocumentIngestionService,
    InMemoryDeduplicationStore,
    RawDocumentPayload,
)


@dataclass(frozen=True)
class IngestionEvaluationMetrics:
    total_records: int
    deduplication_accuracy: float
    indexing_accuracy: float


class IngestionEvaluator:
    def __init__(self, db_path: str | Path = "ingestion_benchmark.db") -> None:
        seed_ingestion_data(db_path)
        self._store = SQLiteIngestionBenchmarkStore(db_path)

    def evaluate(self) -> IngestionEvaluationMetrics:
        records = self._store.fetch_all()
        if not records:
            return IngestionEvaluationMetrics(0, 0.0, 0.0)

        payloads = [
            RawDocumentPayload(
                doc_id=r.doc_id,
                source_uri=r.uri,
                content=r.content,
            )
            for r in records
        ]

        service = DocumentIngestionService(dedup_store=InMemoryDeduplicationStore())
        result = service.process(payloads)

        expected_duplicates = sum(1 for r in records if r.is_expected_duplicate)
        expected_inserts = len(records) - expected_duplicates

        dedup_accuracy = 1.0 if result.skipped_duplicates == expected_duplicates else 0.0
        indexing_accuracy = 1.0 if result.inserted_count == expected_inserts else 0.0

        return IngestionEvaluationMetrics(
            total_records=len(records),
            deduplication_accuracy=dedup_accuracy,
            indexing_accuracy=indexing_accuracy,
        )


if __name__ == "__main__":
    evaluator = IngestionEvaluator()
    metrics = evaluator.evaluate()
    print(f"Ingestion Evaluation: {metrics}")
