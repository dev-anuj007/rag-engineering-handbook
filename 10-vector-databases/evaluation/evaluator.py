from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteVectorStoreGoldenDB, seed_vector_db_data
from ..implementation.chroma_db_store import (
    ChromaDatabaseManager,
    HNSWIndexConfig,
)


@dataclass(frozen=True)
class VectorStoreMetrics:
    total_indexed: int
    retrieval_precision: float
    deletion_verified: bool


class VectorStoreEvaluator:
    def __init__(self, db_path: str | Path = "vector_db_eval.db") -> None:
        seed_vector_db_data(db_path)
        self._store = SQLiteVectorStoreGoldenDB(db_path)
        self._manager = ChromaDatabaseManager(
            collection_name="eval_chroma_vdb",
            hnsw_config=HNSWIndexConfig(m=16, ef_construction=64, ef_search=32),
        )

    def evaluate(self) -> VectorStoreMetrics:
        records = self._store.fetch_all()
        if not records:
            return VectorStoreMetrics(0, 0.0, False)

        docs = [
            Document(
                text=r.content,
                metadata={"doc_id": r.doc_id, "category": r.category},
            )
            for r in records
        ]
        self._manager.index_documents(docs)

        hits = 0
        for r in records:
            results = self._manager.query(r.test_query, similarity_top_k=1)
            if results and results[0].node.metadata.get("doc_id") == r.doc_id:
                hits += 1

        # Test deletion
        self._manager.delete_document(records[0].doc_id)
        post_del_count = self._manager.count()
        deletion_ok = post_del_count == (len(records) - 1)

        total = len(records)
        return VectorStoreMetrics(
            total_indexed=total,
            retrieval_precision=hits / total,
            deletion_verified=deletion_ok,
        )


if __name__ == "__main__":
    evaluator = VectorStoreEvaluator()
    metrics = evaluator.evaluate()
    print(f"Vector Database Evaluation: {metrics}")
