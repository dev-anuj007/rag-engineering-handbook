from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.golden_dataset import SQLiteParsingGoldenStore, seed_parsing_data
from ..implementation.multimodal_parser import (
    ModalityType,
    MultimodalIndexer,
    StructuredDocumentParser,
)


@dataclass(frozen=True)
class ParsingMetrics:
    total_samples: int
    classification_accuracy: float


class MultimodalParsingEvaluator:
    def __init__(self, db_path: str | Path = "parsing_eval.db") -> None:
        seed_parsing_data(db_path)
        self._store = SQLiteParsingGoldenStore(db_path)
        self._parser = StructuredDocumentParser()
        self._indexer = MultimodalIndexer()

    def evaluate(self) -> ParsingMetrics:
        records = self._store.fetch_all()
        if not records:
            return ParsingMetrics(0, 0.0)

        correct_classifications = 0

        for r in records:
            elements = self._parser.parse(r.raw_document, {"doc_id": r.id})
            text_count = sum(1 for e in elements if e.modality == ModalityType.TEXT)
            table_count = sum(1 for e in elements if e.modality == ModalityType.TABLE)
            img_count = sum(1 for e in elements if e.modality == ModalityType.IMAGE_DESCRIPTION)

            if (
                text_count == r.expected_text_count
                and table_count == r.expected_table_count
                and img_count == r.expected_image_count
            ):
                correct_classifications += 1

            # Validate ChromaDB indexing
            self._indexer.index_elements(elements)

        total = len(records)
        return ParsingMetrics(
            total_samples=total,
            classification_accuracy=correct_classifications / total,
        )


if __name__ == "__main__":
    evaluator = MultimodalParsingEvaluator()
    metrics = evaluator.evaluate()
    print(f"Parsing Evaluation: {metrics}")
