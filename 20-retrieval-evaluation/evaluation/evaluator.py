from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteRetrievalEvalGoldenStore, seed_retrieval_ranking_data
from ..implementation.retrieval_metrics import (
    ChromaRetrievalBenchmarker,
    IRankingMetricCalculator,
    RankingMetricsResult,
)


@dataclass(frozen=True)
class OverallRetrievalMetricsSummary:
    avg_hit_rate_at_3: float
    avg_mrr: float
    avg_ndcg_at_3: float
    total_evaluated: int


class RetrievalRankingEvaluator:
    def __init__(self, db_path: str | Path = "retrieval_ranking_eval.db") -> None:
        seed_retrieval_ranking_data(db_path)
        self._store = SQLiteRetrievalEvalGoldenStore(db_path)
        self._benchmarker = ChromaRetrievalBenchmarker()

    def evaluate(self) -> OverallRetrievalMetricsSummary:
        records = self._store.fetch_all()
        if not records:
            return OverallRetrievalMetricsSummary(0.0, 0.0, 0.0, 0)

        docs = [
            Document(text=r.corpus_text, metadata={"doc_id": r.relevant_doc_id})
            for r in records
        ]
        self._benchmarker.index(docs)

        hr_list: list[float] = []
        mrr_list: list[float] = []
        ndcg_list: list[float] = []

        for r in records:
            retrieved_ids = self._benchmarker.retrieve_doc_ids(r.query, top_k=3)
            metrics = IRankingMetricCalculator.calculate_metrics(
                ground_truth_ids=[r.relevant_doc_id],
                retrieved_ids=retrieved_ids,
                k=3,
            )
            hr_list.append(metrics.hit_rate_at_k)
            mrr_list.append(metrics.mrr)
            ndcg_list.append(metrics.ndcg_at_k)

        total = len(records)
        return OverallRetrievalMetricsSummary(
            avg_hit_rate_at_3=sum(hr_list) / total,
            avg_mrr=sum(mrr_list) / total,
            avg_ndcg_at_3=sum(ndcg_list) / total,
            total_evaluated=total,
        )


if __name__ == "__main__":
    evaluator = RetrievalRankingEvaluator()
    summary = evaluator.evaluate()
    print(f"Retrieval Evaluation Summary: {summary}")
