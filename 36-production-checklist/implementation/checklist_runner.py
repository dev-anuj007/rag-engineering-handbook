from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProductionCheckItem:
    check_id: str
    category: str  # 'Security', 'Reliability', 'Evaluation', 'Performance'
    description: str
    is_mandatory: bool


@dataclass(frozen=True)
class ChecklistAuditReport:
    total_checks: int
    mandatory_passed: bool
    readiness_percentage: float


class ProductionChecklistVerifier:
    def __init__(self, checks: Sequence[ProductionCheckItem]) -> None:
        self._checks = checks

    def verify(self, active_statuses: dict[str, bool]) -> ChecklistAuditReport:
        total = len(self._checks)
        if total == 0:
            return ChecklistAuditReport(0, True, 1.0)

        passed = 0
        mandatory_ok = True

        for c in self._checks:
            status = active_statuses.get(c.check_id, False)
            if status:
                passed += 1
            elif c.is_mandatory:
                mandatory_ok = False

        return ChecklistAuditReport(
            total_checks=total,
            mandatory_passed=mandatory_ok,
            readiness_percentage=passed / total,
        )
