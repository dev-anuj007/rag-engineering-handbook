from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteAsymmetricGoldenStore, seed_asymmetric_data
from ..implementation.embedding_providers import (
    AsymmetricVectorStore,
    EmbeddingTaskType,
    GeminiAsymmetricEmbeddingProvider,
)


@dataclass(frozen=True)
class AsymmetricEvalMetrics:
    total_records: int
    retrieval_hit_rate: float


class AsymmetricModelEvaluator:
    def __init__(self, db_path: str | Path = "asymmetric_eval.db") -> None:
        seed_asymmetric_data(db_path)
        self._store = SQLiteAsymmetricGoldenStore(db_path)
        self._provider = GeminiAsymmetricEmbeddingProvider()
        self._vector_store = AsymmetricVectorStore()

    def evaluate(self) -> AsymmetricEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return AsymmetricEvalMetrics(0, 0.0)

        docs = [
            Document(
                text=r.doc_text,
                metadata={"id": r.id, "query_target": r.query},
            )
            for r in records
        ]
        index = self._vector_store.index(docs)
        retriever = index.as_retriever(similarity_top_k=1)

        hits = 0
        for r in records:
            results = retriever.retrieve(r.query)
            if results and results[0].node.metadata.get("id") == r.id:
                hits += 1

        total = len(records)
        return AsymmetricEvalMetrics(
            total_records=total,
            retrieval_hit_rate=hits / total,
        )


if __name__ == "__main__":
    evaluator = AsymmetricModelEvaluator()
    metrics = evaluator.evaluate()
    print(f"Asymmetric Model Evaluation: {metrics}")
