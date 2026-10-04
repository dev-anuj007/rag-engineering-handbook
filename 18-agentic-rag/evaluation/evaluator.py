from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteAgentGoldenStore, seed_agent_data
from ..implementation.agentic_router import AgenticRAGManager


@dataclass(frozen=True)
class AgenticEvalMetrics:
    total_queries: int
    correct_routing_rate: float


class AgenticRAGEvaluator:
    def __init__(self, db_path: str | Path = "agent_eval.db") -> None:
        seed_agent_data(db_path)
        self._store = SQLiteAgentGoldenStore(db_path)
        self._manager = AgenticRAGManager()

    def evaluate(self) -> AgenticEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return AgenticEvalMetrics(0, 0.0)

        finance_docs = [
            Document(text="Total Q4 revenue increased by $4.2M year-over-year.")
        ]
        tech_docs = [
            Document(text="Worker pod memory limit is capped at 4GiB per container.")
        ]

        router = self._manager.build_router(finance_docs, tech_docs)

        correct = 0
        for r in records:
            response = router.query(r.query)
            if r.expected_keyword.lower() in str(response).lower():
                correct += 1

        total = len(records)
        return AgenticEvalMetrics(
            total_queries=total,
            correct_routing_rate=correct / total,
        )


if __name__ == "__main__":
    evaluator = AgenticRAGEvaluator()
    metrics = evaluator.evaluate()
    print(f"Agentic RAG Evaluation: {metrics}")
