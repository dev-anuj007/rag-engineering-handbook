from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvaluationTaxonomyReport:
    retrieval_hit_rate: float
    retrieval_mrr: float
    generation_faithfulness: float
    generation_relevance: float
    end_to_end_accuracy: float


class ComponentWiseEvaluationEngine:
    @staticmethod
    def compute_retrieval_metrics(
        ground_truth_ids: Sequence[str],
        retrieved_ids: Sequence[str],
    ) -> tuple[float, float]:
        if not ground_truth_ids:
            return 0.0, 0.0

        hits = sum(1 for doc_id in ground_truth_ids if doc_id in retrieved_ids)
        hit_rate = 1.0 if hits > 0 else 0.0

        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                mrr = 1.0 / (rank + 1)
                break

        return hit_rate, mrr

    @staticmethod
    def compute_generation_faithfulness(
        claims: Sequence[str],
        source_context: str,
    ) -> float:
        if not claims:
            return 1.0
        supported = sum(1 for c in claims if c.lower() in source_context.lower())
        return supported / len(claims)
