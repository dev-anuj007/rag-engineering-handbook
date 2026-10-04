from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteDecisionGoldenStore, seed_decision_data
from ..implementation.decision_engine import (
    ArchitecturalDecisionEngine,
    WorkloadProfile,
)


@dataclass(frozen=True)
class DecisionEvalSummary:
    total_decisions: int
    optimal_architecture_match_rate: float


class DecisionFrameworkEvaluator:
    def __init__(self, db_path: str | Path = "decision_eval.db") -> None:
        seed_decision_data(db_path)
        self._store = SQLiteDecisionGoldenStore(db_path)

    def evaluate(self) -> DecisionEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return DecisionEvalSummary(0, 0.0)

        matches = 0
        for r in records:
            profile = WorkloadProfile(
                has_exact_part_numbers=r.has_exact_part_numbers,
                requires_multi_hop_relational_reasoning=r.requires_multi_hop,
                has_multiple_isolated_departments=r.has_isolated_departments,
                requires_style_adaptation_only=r.requires_style_only,
            )
            recommendation = ArchitecturalDecisionEngine.recommend_architecture(profile)
            if recommendation.value == r.expected_pattern:
                matches += 1

        total = len(records)
        return DecisionEvalSummary(
            total_decisions=total,
            optimal_architecture_match_rate=matches / total,
        )


if __name__ == "__main__":
    evaluator = DecisionFrameworkEvaluator()
    summary = evaluator.evaluate()
    print(f"Decision Framework Evaluation: {summary}")
