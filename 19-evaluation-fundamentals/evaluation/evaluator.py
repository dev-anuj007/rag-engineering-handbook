from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteTaxonomyGoldenStore, seed_taxonomy_data
from ..implementation.metrics_engine import (
    ComponentWiseEvaluationEngine,
    EvaluationTaxonomyReport,
)


class TaxonomyEvaluator:
    def __init__(self, db_path: str | Path = "taxonomy_eval.db") -> None:
        seed_taxonomy_data(db_path)
        self._store = SQLiteTaxonomyGoldenStore(db_path)
        self._engine = ComponentWiseEvaluationEngine()

    def evaluate(self) -> EvaluationTaxonomyReport:
        records = self._store.fetch_all()
        if not records:
            return EvaluationTaxonomyReport(0.0, 0.0, 0.0, 0.0, 0.0)

        hit_rates: list[float] = []
        mrrs: list[float] = []
        faithfulness_scores: list[float] = []

        for r in records:
            # Mock retrieved list with target in rank 1
            retrieved = [r.target_doc_id, "doc-distractor"]
            hr, mrr = self._engine.compute_retrieval_metrics([r.target_doc_id], retrieved)
            hit_rates.append(hr)
            mrrs.append(mrr)

            faith = self._engine.compute_generation_faithfulness([r.claim_1, r.claim_2], r.source_context)
            faithfulness_scores.append(faith)

        avg_hr = sum(hit_rates) / len(hit_rates)
        avg_mrr = sum(mrrs) / len(mrrs)
        avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)

        return EvaluationTaxonomyReport(
            retrieval_hit_rate=avg_hr,
            retrieval_mrr=avg_mrr,
            generation_faithfulness=avg_faith,
            generation_relevance=1.0,
            end_to_end_accuracy=(avg_hr + avg_faith) / 2.0,
        )


if __name__ == "__main__":
    evaluator = TaxonomyEvaluator()
    report = evaluator.evaluate()
    print(f"Evaluation Taxonomy Report: {report}")
