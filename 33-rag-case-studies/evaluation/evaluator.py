from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteCaseStudyGoldenStore, seed_case_study_data
from ..implementation.case_study_runner import (
    DomainCaseStudyConfig,
    DomainCaseStudyPipeline,
)


@dataclass(frozen=True)
class CaseStudyEvalSummary:
    total_case_studies: int
    clause_extraction_accuracy: float


class CaseStudyEvaluator:
    def __init__(self, db_path: str | Path = "case_study_eval.db") -> None:
        seed_case_study_data(db_path)
        self._store = SQLiteCaseStudyGoldenStore(db_path)

    def evaluate(self) -> CaseStudyEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return CaseStudyEvalSummary(0, 0.0)

        correct = 0
        for r in records:
            pipeline = DomainCaseStudyPipeline(DomainCaseStudyConfig(domain_name=r.domain))
            pipeline.load_case_corpus([Document(text=r.content)])
            ans = pipeline.execute_domain_query(r.query)
            if r.expected_clause.lower() in ans.lower():
                correct += 1

        total = len(records)
        return CaseStudyEvalSummary(
            total_case_studies=total,
            clause_extraction_accuracy=correct / total,
        )


if __name__ == "__main__":
    evaluator = CaseStudyEvaluator()
    summary = evaluator.evaluate()
    print(f"Case Study Evaluation: {summary}")
