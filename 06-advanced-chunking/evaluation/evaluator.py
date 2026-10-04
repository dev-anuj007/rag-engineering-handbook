from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteHierarchicalGoldenStore, seed_hierarchical_data
from ..implementation.advanced_chunker import (
    AdvancedChunkIndexer,
    MarkdownHeaderChunker,
)


@dataclass(frozen=True)
class AdvancedChunkingMetrics:
    total_records: int
    header_alignment_rate: float


class AdvancedChunkingEvaluator:
    def __init__(self, db_path: str | Path = "hierarchical_eval.db") -> None:
        seed_hierarchical_data(db_path)
        self._store = SQLiteHierarchicalGoldenStore(db_path)
        self._chunker = MarkdownHeaderChunker()
        self._indexer = AdvancedChunkIndexer()

    def evaluate(self) -> AdvancedChunkingMetrics:
        records = self._store.fetch_all()
        if not records:
            return AdvancedChunkingMetrics(0, 0.0)

        aligned_count = 0

        for r in records:
            docs = [Document(text=r.markdown_content, metadata={"id": r.id})]
            nodes = self._chunker.parse_nodes(docs)

            if len(nodes) >= r.expected_header_count:
                aligned_count += 1

            self._indexer.index(nodes)

        total = len(records)
        return AdvancedChunkingMetrics(
            total_records=total,
            header_alignment_rate=aligned_count / total,
        )


if __name__ == "__main__":
    evaluator = AdvancedChunkingEvaluator()
    metrics = evaluator.evaluate()
    print(f"Advanced Chunking Evaluation: {metrics}")
