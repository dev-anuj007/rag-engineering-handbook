from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteChunkingGoldenStore, seed_chunking_data
from ..implementation.chunker import (
    ChunkIndexer,
    ChunkingConfig,
    SentenceAwareChunker,
)


@dataclass(frozen=True)
class ChunkingEvalMetrics:
    total_samples: int
    boundary_compliance_rate: float


class ChunkingEvaluator:
    def __init__(self, db_path: str | Path = "chunking_eval.db") -> None:
        seed_chunking_data(db_path)
        self._store = SQLiteChunkingGoldenStore(db_path)
        self._chunker = SentenceAwareChunker(ChunkingConfig(chunk_size=128, chunk_overlap=20))
        self._indexer = ChunkIndexer()

    def evaluate(self) -> ChunkingEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return ChunkingEvalMetrics(0, 0.0)

        compliant_count = 0

        for r in records:
            docs = [Document(text=r.text, metadata={"id": r.id})]
            nodes = self._chunker.chunk(docs)

            if r.expected_min_nodes <= len(nodes) <= r.expected_max_nodes:
                compliant_count += 1

            self._indexer.index(nodes)

        total = len(records)
        return ChunkingEvalMetrics(
            total_samples=total,
            boundary_compliance_rate=compliant_count / total,
        )


if __name__ == "__main__":
    evaluator = ChunkingEvaluator()
    metrics = evaluator.evaluate()
    print(f"Chunking Evaluation: {metrics}")
