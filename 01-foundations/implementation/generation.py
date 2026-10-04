from __future__ import annotations

from typing import Sequence

from dotenv import load_dotenv
from llama_index.core.schema import NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI


class ResponseGenerator:
    def __init__(
        self,
        model: str = "gemini-3.6-flash",
    ) -> None:
        self._llm = GoogleGenAI(
            model=model,
        )

    def generate(
        self,
        query: str,
        results: Sequence[NodeWithScore],
    ) -> str:
        context = "\n\n".join(
            result.node.get_content()
            for result in results
        )

        prompt = f"""
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {query}

        If the context does not contain enough information to answer the question,
        say that the information is not available in the provided context.
        """

        response = self._llm.complete(prompt)

        return response.text


if __name__ == "__main__":
    load_dotenv()

    from .ingestion import IngestionConfig, build_pipeline
    from .retrieval import build_retriever

    nodes = build_pipeline(IngestionConfig()).run()

    retriever = build_retriever(nodes)

    query = "How many annual leave days do employees receive?"
    results = retriever.retrieve(query)

    generator = ResponseGenerator()

    answer = generator.generate(
        query=query,
        results=results,
    )

    print(answer)
