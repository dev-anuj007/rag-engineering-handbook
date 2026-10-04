from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core.schema import NodeWithScore, TextNode

from ..data.golden_dataset import SQLiteContextGoldenStore, seed_context_data
from ..implementation.context_compressor import (
    ContextEngineeringConfig,
    ProductionContextPacker,
)


@dataclass(frozen=True)
class ContextEvalMetrics:
    total_records: int
    budget_adherence_rate: float
    source_attribution_rate: float


class ContextEngineeringEvaluator:
    def __init__(self, db_path: str | Path = "context_eval.db") -> None:
        seed_context_data(db_path)
        self._store = SQLiteContextGoldenStore(db_path)
        self._packer = ProductionContextPacker(
            ContextEngineeringConfig(max_token_budget=300, enable_reordering=True)
        )

    def evaluate(self) -> ContextEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return ContextEvalMetrics(0, 0.0, 0.0)

        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Database client pool timeout is set to 3000ms by default.",
                    metadata={"doc_id": "db_config"},
                ),
                score=0.95,
            ),
            NodeWithScore(
                node=TextNode(
                    text="HTTP keep-alive max idle timeout is configured at 60 seconds.",
                    metadata={"doc_id": "http_config"},
                ),
                score=0.72,
            ),
        ]

        budget_ok = 0
        attrib_ok = 0

        for r in records:
            packed_prompt = self._packer.pack(r.query, nodes)
            if len(packed_prompt) <= r.max_budget_chars:
                budget_ok += 1
            if r.target_source_tag in packed_prompt:
                attrib_ok += 1

        total = len(records)
        return ContextEvalMetrics(
            total_records=total,
            budget_adherence_rate=budget_ok / total,
            source_attribution_rate=attrib_ok / total,
        )


if __name__ == "__main__":
    evaluator = ContextEngineeringEvaluator()
    metrics = evaluator.evaluate()
    print(f"Context Engineering Evaluation: {metrics}")
