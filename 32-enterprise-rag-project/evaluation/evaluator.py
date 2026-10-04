from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteEnterpriseGoldenStore, seed_enterprise_data
from ..implementation.enterprise_service import (
    EnterpriseQueryRequest,
    EnterpriseRAGService,
)


@dataclass(frozen=True)
class EnterpriseEvalSummary:
    total_queries: int
    tenant_isolation_success: bool
    accuracy: float


class EnterpriseRAGEvaluator:
    def __init__(self, db_path: str | Path = "enterprise_eval.db") -> None:
        seed_enterprise_data(db_path)
        self._store = SQLiteEnterpriseGoldenStore(db_path)
        self._service = EnterpriseRAGService()

    def evaluate(self) -> EnterpriseEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return EnterpriseEvalSummary(0, False, 0.0)

        for r in records:
            self._service.ingest_tenant_document(r.tenant_id, r.doc_id, r.content)

        # 1. Test Authorized Query
        rec = records[0]
        req_auth = EnterpriseQueryRequest(tenant_id=rec.tenant_id, user_id="user_1", query=rec.query)
        resp_auth = self._service.query_tenant(req_auth)
        auth_success = rec.expected_keyword.lower() in resp_auth.answer.lower()

        # 2. Test Unauthorized Foreign Tenant Query
        req_foreign = EnterpriseQueryRequest(tenant_id="tenant-foreign", user_id="user_2", query=rec.query)
        resp_foreign = self._service.query_tenant(req_foreign)
        isolation_ok = len(resp_foreign.sources) == 0

        return EnterpriseEvalSummary(
            total_queries=2,
            tenant_isolation_success=isolation_ok,
            accuracy=1.0 if auth_success else 0.0,
        )


if __name__ == "__main__":
    evaluator = EnterpriseRAGEvaluator()
    summary = evaluator.evaluate()
    print(f"Enterprise RAG Project Evaluation: {summary}")
