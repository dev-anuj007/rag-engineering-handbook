from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core import Document

from ..data.golden_dataset import SQLiteLlamaIndexGoldenStore, seed_llamaindex_data
from ..implementation.custom_engine import create_custom_rag_engine


@dataclass(frozen=True)
class CustomEngineEvalSummary:
    total_samples: int
    execution_success_rate: float


class LlamaIndexDeepDiveEvaluator:
    def __init__(self, db_path: str | Path = "llamaindex_eval.db") -> None:
        seed_llamaindex_data(db_path)
        self._store = SQLiteLlamaIndexGoldenStore(db_path)

    def evaluate(self) -> CustomEngineEvalSummary:
        records = self._store.fetch_all()
        if not records:
            return CustomEngineEvalSummary(0, 0.0)

        docs = [Document(text=r.content) for r in records]
        engine = create_custom_rag_engine(docs)

        success = 0
        for r in records:
            res = engine.query(r.query)
            if r.expected_ans_substring.lower() in str(res).lower():
                success += 1

        total = len(records)
        return CustomEngineEvalSummary(
            total_samples=total,
            execution_success_rate=success / total,
        )


if __name__ == "__main__":
    evaluator = LlamaIndexDeepDiveEvaluator()
    summary = evaluator.evaluate()
    print(f"LlamaIndex Custom Engine Evaluation: {summary}")
