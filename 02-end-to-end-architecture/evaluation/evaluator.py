from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteGoldenStore, seed_chapter_dataset
from ..implementation.pipeline import PipelineConfig, create_pipeline


@dataclass(frozen=True)
class EvaluationSummary:
    total: int
    keyword_match_rate: float
    retrieval_success_rate: float


class EndToEndEvaluator:
    def __init__(self, db_path: str | Path = "golden_eval.db") -> None:
        seed_chapter_dataset(db_path)
        self._store = SQLiteGoldenStore(db_path)

    def run_eval(self) -> EvaluationSummary:
        records = self._store.fetch_all()
        if not records:
            return EvaluationSummary(0, 0.0, 0.0)

        # Mock source corpus for initialization
        documents = [
            Document(
                text="Tier-1 incident response time SLA is 15 minutes across all production services.",
                metadata={"doc_type": "sla_policy"},
            ),
            Document(
                text="Budget requests exceeding $50,000 require formal CFO approval before commitment.",
                metadata={"doc_type": "finance_policy"},
            ),
        ]

        pipeline = create_pipeline(PipelineConfig())
        pipeline.initialize(documents)

        matched_keywords = 0
        successful_retrievals = 0

        for record in records:
            answer, nodes = pipeline.query(record.query)
            if len(nodes) > 0:
                successful_retrievals += 1

            required_terms = [k.strip().lower() for k in record.expected_keywords.split(",")]
            if all(term in answer.lower() or any(term in n.node.get_content().lower() for n in nodes) for term in required_terms):
                matched_keywords += 1

        total = len(records)
        return EvaluationSummary(
            total=total,
            keyword_match_rate=matched_keywords / total,
            retrieval_success_rate=successful_retrievals / total,
        )


if __name__ == "__main__":
    evaluator = EndToEndEvaluator()
    summary = evaluator.run_eval()
    print(f"End-to-End Evaluation Complete: {summary}")
