from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from dotenv import load_dotenv

from ..data.golden_dataset import GoldenDatasetRepository, GoldenRecord, seed_default_dataset
from ..implementation.basic_rag import BasicRAG, build_rag


@dataclass(frozen=True)
class EvalMetrics:
    hit_rate: float
    exact_match_ratio: float
    total_evaluated: int


class FoundationsEvaluator:
    def __init__(self, rag_system: BasicRAG, dataset_repo: GoldenDatasetRepository) -> None:
        self._rag = rag_system
        self._dataset_repo = dataset_repo

    def evaluate(self) -> EvalMetrics:
        records: Sequence[GoldenRecord] = self._dataset_repo.get_all_records()
        if not records:
            return EvalMetrics(hit_rate=0.0, exact_match_ratio=0.0, total_evaluated=0)

        hits = 0
        exact_matches = 0

        for record in records:
            # 1. Retrieve & evaluate hit
            results = self._rag._retriever.retrieve(record.query)
            retrieved_texts = [res.node.get_content() for res in results]

            if any(record.expected_context in text for text in retrieved_texts):
                hits += 1

            # 2. Generation answer check
            answer = self._rag._generator.generate(record.query, results)
            if record.expected_answer.strip().lower() in answer.strip().lower():
                exact_matches += 1

        total = len(records)
        return EvalMetrics(
            hit_rate=hits / total,
            exact_match_ratio=exact_matches / total,
            total_evaluated=total,
        )


def run_evaluation(db_path: str | Path = "eval_dataset.db") -> EvalMetrics:
    seed_default_dataset(db_path)
    repo = GoldenDatasetRepository(db_path)
    rag = build_rag()
    evaluator = FoundationsEvaluator(rag_system=rag, dataset_repo=repo)
    return evaluator.evaluate()


if __name__ == "__main__":
    load_dotenv()
    metrics = run_evaluation()
    print(f"Evaluation Results on SQLite Golden Dataset:")
    print(f"Total Records: {metrics.total_evaluated}")
    print(f"Hit Rate: {metrics.hit_rate:.2%}")
    print(f"Answer Match Ratio: {metrics.exact_match_ratio:.2%}")
