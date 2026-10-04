from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteRerankGoldenStore, seed_rerank_data
from ..implementation.reranker import (
    SemanticRelevanceReranker,
    TwoStagePipelineConfig,
    TwoStageRetrievalPipeline,
)


@dataclass(frozen=True)
class RerankingEvalMetrics:
    total_samples: int
    top_1_accuracy: float


class RerankingEvaluator:
    def __init__(self, db_path: str | Path = "rerank_eval.db") -> None:
        seed_rerank_data(db_path)
        self._store = SQLiteRerankGoldenStore(db_path)
        self._pipeline = TwoStageRetrievalPipeline(
            reranker=SemanticRelevanceReranker(),
            config=TwoStagePipelineConfig(coarse_top_k=5, fine_top_k=1),
        )

    def evaluate(self) -> RerankingEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return RerankingEvalMetrics(0, 0.0)

        docs = [
            Document(
                text="Python async await event loop enables asynchronous concurrent I/O operations.",
                metadata={"id": "doc-precise-async"},
            ),
            Document(
                text="Python multithreading uses the Global Interpreter Lock (GIL) for CPU-bound tasks.",
                metadata={"id": "doc-gil-threading"},
            ),
            Document(
                text="JavaScript promises and async functions execute on V8 runtime microtask queues.",
                metadata={"id": "doc-js-async"},
            ),
        ]
        self._pipeline.index(docs)

        top_1_hits = 0
        for r in records:
            results = self._pipeline.retrieve_and_rerank(r.query)
            if results and results[0].node.metadata.get("id") == r.best_doc_id:
                top_1_hits += 1

        total = len(records)
        return RerankingEvalMetrics(
            total_samples=total,
            top_1_accuracy=top_1_hits / total,
        )


if __name__ == "__main__":
    evaluator = RerankingEvaluator()
    metrics = evaluator.evaluate()
    print(f"Reranking Evaluation: {metrics}")
