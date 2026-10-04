from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteGraphGoldenStore, seed_graph_data
from ..implementation.graph_retriever import (
    HybridGraphVectorRetriever,
    KnowledgeGraphTriplet,
)


@dataclass(frozen=True)
class GraphRAGEvalSummary:
    total_queries: int
    multi_hop_recall: float


class GraphRAGEvaluator:
    def __init__(self, db_path: str | Path = "graph_eval.db") -> None:
        seed_graph_data(db_path)
        self._store = SQLiteGraphGoldenStore(db_path)
        self._retriever = HybridGraphVectorRetriever()

    def evaluate(self) -> GraphRAGEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return GraphRAGEvalSummary(0, 0.0)

        triplets = [
            KnowledgeGraphTriplet("FastAPI", "isBuiltOn", "Starlette"),
            KnowledgeGraphTriplet("Starlette", "uses", "AnyIO"),
            KnowledgeGraphTriplet("PostgreSQL", "implements", "WAL"),
        ]
        self._retriever.populate_graph(triplets)

        hits = 0
        for r in records:
            results = self._retriever.retrieve_with_graph_expansion(r.entity_query)
            if any(r.expected_related_entity in fact.lower() for fact in results):
                hits += 1

        total = len(records)
        return GraphRAGEvalSummary(
            total_queries=total,
            multi_hop_recall=hits / total,
        )


if __name__ == "__main__":
    evaluator = GraphRAGEvaluator()
    summary = evaluator.evaluate()
    print(f"Graph RAG Evaluation: {summary}")
