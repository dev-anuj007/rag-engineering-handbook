from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core.schema import TextNode

from ..data.golden_dataset import SQLiteHybridGoldenStore, seed_hybrid_data
from ..implementation.hybrid_retriever import (
    HybridRRFRetriever,
    HybridRetrievalConfig,
)


@dataclass(frozen=True)
class HybridEvalMetrics:
    total_queries: int
    keyword_exact_hit_rate: float
    semantic_hit_rate: float
    overall_hit_rate: float


class HybridEvaluator:
    def __init__(self, db_path: str | Path = "hybrid_eval.db") -> None:
        seed_hybrid_data(db_path)
        self._store = SQLiteHybridGoldenStore(db_path)
        nodes = [
            TextNode(
                text="Gateway returns ERR_CONNECTION_TIMED_OUT code 504 when upstream proxy fails.",
                id_="node-err-504",
            ),
            TextNode(
                text="Horizontal pod autoscaler increases replica count dynamically when CPU exceeds 80%.",
                id_="node-autoscale-workers",
            ),
        ]
        self._retriever = HybridRRFRetriever(
            nodes=nodes,
            config=HybridRetrievalConfig(final_top_k=2),
        )

    def evaluate(self) -> HybridEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return HybridEvalMetrics(0, 0.0, 0.0, 0.0)

        keyword_hits = 0
        semantic_hits = 0
        keyword_total = 0
        semantic_total = 0

        for r in records:
            results = self._retriever.retrieve(r.query)
            hit = any(n.node.node_id == r.target_node_id for n in results)
            if r.query_type == "keyword_exact":
                keyword_total += 1
                if hit:
                    keyword_hits += 1
            else:
                semantic_total += 1
                if hit:
                    semantic_hits += 1

        total = len(records)
        return HybridEvalMetrics(
            total_queries=total,
            keyword_exact_hit_rate=keyword_hits / keyword_total if keyword_total else 0.0,
            semantic_hit_rate=semantic_hits / semantic_total if semantic_total else 0.0,
            overall_hit_rate=(keyword_hits + semantic_hits) / total,
        )


if __name__ == "__main__":
    evaluator = HybridEvaluator()
    metrics = evaluator.evaluate()
    print(f"Hybrid Retrieval Evaluation: {metrics}")
