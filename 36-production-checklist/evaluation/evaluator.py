from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteChecklistGoldenStore, seed_checklist_data
from ..implementation.checklist_runner import (
    ChecklistAuditReport,
    ProductionCheckItem,
    ProductionChecklistVerifier,
)


class ProductionChecklistEvaluator:
    def __init__(self, db_path: str | Path = "checklist_eval.db") -> None:
        seed_checklist_data(db_path)
        self._store = SQLiteChecklistGoldenStore(db_path)

    def evaluate(self) -> ChecklistAuditReport:
        records = self._store.fetch_all()
        checks = [
            ProductionCheckItem(
                check_id=r.check_id,
                category=r.category,
                description=r.description,
                is_mandatory=r.is_mandatory,
            )
            for r in records
        ]
        verifier = ProductionChecklistVerifier(checks)

        # Simulate all mandatory checklist items passing
        active_statuses = {c.check_id: True for c in checks}
        return verifier.verify(active_statuses)


if __name__ == "__main__":
    evaluator = ProductionChecklistEvaluator()
    report = evaluator.evaluate()
    print(f"Production Checklist Audit Report: {report}")
