from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from llama_index.core.schema import NodeWithScore, TextNode

from ..data.golden_dataset import SQLiteSynthesisGoldenStore, seed_synthesis_data
from ..implementation.synthesizer import (
    LlamaIndexResponseSynthesizer,
    SynthesisMode,
)


@dataclass(frozen=True)
class SynthesisEvalMetrics:
    total_queries: int
    containment_rate: float


class SynthesisEvaluator:
    def __init__(self, db_path: str | Path = "synthesis_eval.db") -> None:
        seed_synthesis_data(db_path)
        self._store = SQLiteSynthesisGoldenStore(db_path)
        self._synthesizer = LlamaIndexResponseSynthesizer(mode=SynthesisMode.COMPACT)

    def evaluate(self) -> SynthesisEvalMetrics:
        records = self._store.fetch_all()
        if not records:
            return SynthesisEvalMetrics(0, 0.0)

        contained_count = 0
        for r in records:
            nodes = [
                NodeWithScore(node=TextNode(text=r.context), score=0.95)
            ]
            response = self._synthesizer.synthesize(r.query, nodes)
            if r.expected_answer_contains.lower() in response.lower():
                contained_count += 1

        total = len(records)
        return SynthesisEvalMetrics(
            total_queries=total,
            containment_rate=contained_count / total,
        )


if __name__ == "__main__":
    evaluator = SynthesisEvaluator()
    metrics = evaluator.evaluate()
    print(f"Generation Synthesis Evaluation: {metrics}")
