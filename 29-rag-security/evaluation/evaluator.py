from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteSecurityGoldenStore, seed_security_data
from ..implementation.security_guard import SecuritySanitizer


@dataclass(frozen=True)
class SecurityDefenseSummary:
    total_payloads: int
    pii_redaction_accuracy: float
    injection_defense_rate: float


class SecurityEvaluator:
    def __init__(self, db_path: str | Path = "security_eval.db") -> None:
        seed_security_data(db_path)
        self._store = SQLiteSecurityGoldenStore(db_path)

    def evaluate(self) -> SecurityDefenseSummary:
        records = self._store.fetch_all()
        if not records:
            return SecurityDefenseSummary(0, 0.0, 0.0)

        pii_ok = 0
        inj_ok = 0

        for r in records:
            result = SecuritySanitizer.sanitize_text(r.raw_payload)
            if (result.pii_redacted_count > 0) == r.contains_pii:
                pii_ok += 1
            if result.injection_detected == r.is_malicious_injection:
                inj_ok += 1

        total = len(records)
        return SecurityDefenseSummary(
            total_payloads=total,
            pii_redaction_accuracy=pii_ok / total,
            injection_defense_rate=inj_ok / total,
        )


if __name__ == "__main__":
    evaluator = SecurityEvaluator()
    summary = evaluator.evaluate()
    print(f"Security Defense Evaluation: {summary}")
