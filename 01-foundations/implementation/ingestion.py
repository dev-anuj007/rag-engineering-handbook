from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from dotenv import load_dotenv

@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50


class DocumentLoader:
    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = documents

    def load(self) -> Sequence[Document]:
        return self._documents


class NodeParser:
    def __init__(self, config: IngestionConfig) -> None:
        self._parser = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def parse(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        return self._parser.get_nodes_from_documents(documents)


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        parser: NodeParser,
    ) -> None:
        self._loader = loader
        self._parser = parser

    def run(self) -> Sequence[BaseNode]:
        documents = self._loader.load()
        return self._parser.parse(documents)


def build_pipeline(config: IngestionConfig) -> IngestionPipeline:
    documents = [
        Document(
            text="Employees receive 24 days of annual leave every year.",
            metadata={
                "document_id": "employee-handbook",
                "section": "annual-leave",
            },
        ),
        Document(
            text="Employees receive 12 days of sick leave every year.",
            metadata={
                "document_id": "employee-handbook",
                "section": "sick-leave",
            },
        ),
    ]

    return IngestionPipeline(
        loader=DocumentLoader(documents),
        parser=NodeParser(config),
    )


if __name__ == "__main__":
    load_dotenv()
    pipeline = build_pipeline(IngestionConfig())
    nodes = pipeline.run()

    print(f"Generated {len(nodes)} nodes")
