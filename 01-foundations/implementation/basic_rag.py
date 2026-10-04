from __future__ import annotations

from dotenv import load_dotenv

from .generation import ResponseGenerator
from .ingestion import IngestionConfig, build_pipeline
from .retrieval import build_retriever


class BasicRAG:
    def __init__(
        self,
        retriever,
        generator: ResponseGenerator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def query(self, question: str) -> str:
        results = self._retriever.retrieve(question)

        return self._generator.generate(
            query=question,
            results=results,
        )


def build_rag() -> BasicRAG:
    nodes = build_pipeline(
        IngestionConfig()
    ).run()

    retriever = build_retriever(nodes)
    generator = ResponseGenerator()

    return BasicRAG(
        retriever=retriever,
        generator=generator,
    )


if __name__ == "__main__":
    load_dotenv()

    rag = build_rag()

    question = "How many annual leave days do employees receive?"

    answer = rag.query(question)

    print(answer)
