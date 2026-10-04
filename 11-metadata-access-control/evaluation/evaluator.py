from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteRBACGoldenStore, seed_rbac_data
from ..implementation.access_control import (
    AccessControlledRetriever,
    SecurityClearance,
    UserContext,
)


@dataclass(frozen=True)
class RBACEvalMetrics:
    total_checks: int
    security_enforcement_accuracy: float
    zero_leakage_verified: bool


class RBACEvaluator:
    def __init__(self, db_path: str | Path = "rbac_eval.db") -> None:
        seed_rbac_data(db_path)
        self._store = SQLiteRBACGoldenStore(db_path)
        self._retriever = AccessControlledRetriever()

    def evaluate(self) -> RBACEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return RBACEvalMetrics(0, 0.0, False)

        docs = [
            Document(
                text=r.content,
                metadata={
                    "doc_id": r.id,
                    "tenant_id": r.tenant_id,
                    "clearance": r.clearance,
                },
            )
            for r in records
        ]
        self._retriever.index_secure_documents(docs)

        correct_enforcements = 0
        leaks = 0

        for r in records:
            user_ctx = UserContext(
                user_id="test-user",
                tenant_id=r.user_tenant,
                clearance=SecurityClearance(r.user_clearance),
            )
            nodes = self._retriever.retrieve_with_rbac(
                query=r.content,
                user_context=user_ctx,
                similarity_top_k=5,
            )
            retrieved_doc_ids = [n.node.metadata.get("doc_id") for n in nodes]
            is_present = r.id in retrieved_doc_ids

            if is_present == r.should_permit:
                correct_enforcements += 1
            else:
                if not r.should_permit and is_present:
                    leaks += 1

        total = len(records)
        return RBACEvalMetrics(
            total_checks=total,
            security_enforcement_accuracy=correct_enforcements / total,
            zero_leakage_verified=(leaks == 0),
        )


if __name__ == "__main__":
    evaluator = RBACEvaluator()
    metrics = evaluator.evaluate()
    print(f"RBAC Security Evaluation: {metrics}")
