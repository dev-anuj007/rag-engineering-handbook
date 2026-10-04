from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteRetrievalGoldenStore, seed_retrieval_data
from ..implementation.dense_retriever import (
    DenseRetrievalConfig,
    ThresholdedDenseRetriever,
)


@dataclass(frozen=True)
class RetrievalEvalMetrics:
    total_queries: int
    hit_rate_at_1: float
    precision_at_k: float


class DenseRetrievalEvaluator:
    def __init__(self, db_path: str | Path = "retrieval_eval.db") -> None:
        seed_retrieval_data(db_path)
        self._store = SQLiteRetrievalGoldenStore(db_path)
        self._retriever = ThresholdedDenseRetriever(
            DenseRetrievalConfig(similarity_top_k=2, score_threshold=0.5)
        )

    def evaluate(self) -> RetrievalEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return RetrievalEvalMetrics(0, 0.0, 0.0)

        docs = []
        for r in records:
            docs.append(Document(text=r.target_content, metadata={"id": f"target-{r.id}"}))
            docs.append(Document(text=r.distractor_content, metadata={"id": f"dist-{r.id}"}))

        self._retriever.index_corpus(docs)

        hits_at_1 = 0
        hits_in_k = 0

        for r in records:
            results = self._retriever.retrieve(r.query)
            if results:
                if results[0].node.metadata.get("id") == f"target-{r.id}":
                    hits_at_1 += 1
                if any(n.node.metadata.get("id") == f"target-{r.id}" for n in results):
                    hits_in_k += 1

        total = len(records)
        return RetrievalEvalMetrics(
            total_queries=total,
            hit_rate_at_1=hits_at_1 / total,
            precision_at_k=hits_in_k / total,
        )


if __name__ == "__main__":
    evaluator = DenseRetrievalEvaluator()
    metrics = evaluator.evaluate()
    print(f"Retrieval Fundamentals Evaluation: {metrics}")
