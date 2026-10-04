from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteSelectionGoldenStore, seed_selection_data
from ..implementation.model_selector import (
    DomainProfile,
    EmbeddingModelCandidate,
    EmbeddingModelSelector,
    ModelSelectionCriteria,
)


@dataclass(frozen=True)
class SelectionMetrics:
    total_evaluations: int
    optimal_selection_rate: float


class EmbeddingSelectionEvaluator:
    def __init__(self, db_path: str | Path = "selection_eval.db") -> None:
        seed_selection_data(db_path)
        self._store = SQLiteSelectionGoldenStore(db_path)
        candidates = [
            EmbeddingModelCandidate(
                name="code-bert-base",
                dimension=768,
                max_tokens=512,
                avg_latency_ms=25.0,
                mteb_score=64.2,
                supported_domains=[DomainProfile.CODE, DomainProfile.GENERAL],
            ),
            EmbeddingModelCandidate(
                name="gemini-embedding-001",
                dimension=768,
                max_tokens=2048,
                avg_latency_ms=45.0,
                mteb_score=71.8,
                supported_domains=[DomainProfile.GENERAL, DomainProfile.FINANCIAL, DomainProfile.LEGAL],
            ),
            EmbeddingModelCandidate(
                name="finance-embed-v1",
                dimension=1024,
                max_tokens=1024,
                avg_latency_ms=90.0,
                mteb_score=68.5,
                supported_domains=[DomainProfile.FINANCIAL],
            ),
        ]
        self._selector = EmbeddingModelSelector(candidates)

    def evaluate(self) -> SelectionMetrics:
        records = self._store.fetch_all()
        if not records:
            return SelectionMetrics(0, 0.0)

        correct = 0
        for r in records:
            criteria = ModelSelectionCriteria(
                domain=DomainProfile(r.domain),
                max_latency_ms=r.max_latency_ms,
                max_memory_dim=r.max_dim,
                context_window_tokens=256,
            )
            selected = self._selector.select_best_model(criteria)
            if selected.name == r.expected_model_name:
                correct += 1

        total = len(records)
        return SelectionMetrics(
            total_evaluations=total,
            optimal_selection_rate=correct / total,
        )


if __name__ == "__main__":
    evaluator = EmbeddingSelectionEvaluator()
    metrics = evaluator.evaluate()
    print(f"Embedding Selection Evaluation: {metrics}")
