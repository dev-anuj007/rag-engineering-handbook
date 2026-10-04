from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteTraceGoldenStore, seed_trace_data
from ..implementation.tracer import ObservabilityTracer


@dataclass(frozen=True)
class ObservabilityEvalSummary:
    total_spans_checked: int
    sla_compliance_rate: float


class ObservabilityEvaluator:
    def __init__(self, db_path: str | Path = "traces_eval.db") -> None:
        seed_trace_data(db_path)
        self._store = SQLiteTraceGoldenStore(db_path)

    def evaluate(self) -> ObservabilityEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return ObservabilityEvalSummary(0, 0.0)

        tracer = ObservabilityTracer()
        tracer.record_span("chroma_dense_retrieval", 12.5, {"collection": "production"})
        tracer.record_span("reranking_stage", 34.0, {"top_k": "5"})
        trace = tracer.finalize()

        compliant = 0
        for r in records:
            span = next((s for s in trace.spans if s.name == r.stage_name), None)
            if span and span.duration_ms <= r.max_acceptable_ms:
                compliant += 1

        total = len(records)
        return ObservabilityEvalSummary(
            total_spans_checked=total,
            sla_compliance_rate=compliant / total,
        )


if __name__ == "__main__":
    evaluator = ObservabilityEvaluator()
    summary = evaluator.evaluate()
    print(f"Observability Evaluation: {summary}")
