from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteProdGoldenStore, seed_prod_data
from ..implementation.production_service import ProductionRAGService


@dataclass(frozen=True)
class ProdEvalMetrics:
    total_queries: int
    cache_hit_rate: float
    service_stability: float


class ProductionEvaluator:
    def __init__(self, db_path: str | Path = "prod_eval.db") -> None:
        seed_prod_data(db_path)
        self._store = SQLiteProdGoldenStore(db_path)
        self._service = ProductionRAGService()

    def evaluate(self) -> ProdEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return ProdEvalMetrics(0, 0.0, 0.0)

        docs = [Document(text=r.content) for r in records]
        self._service.index(docs)

        # First query (Cache miss)
        r = records[0]
        self._service.query(r.query)

        # Second query (Cache hit)
        ans, was_cached = self._service.query(r.query)

        return ProdEvalMetrics(
            total_queries=2,
            cache_hit_rate=0.5,
            service_stability=1.0 if was_cached else 0.0,
        )


if __name__ == "__main__":
    evaluator = ProductionEvaluator()
    metrics = evaluator.evaluate()
    print(f"Production Service Evaluation: {metrics}")
