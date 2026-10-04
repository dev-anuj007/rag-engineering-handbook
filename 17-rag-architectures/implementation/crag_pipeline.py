from __future__ import annotations

from enum import Enum
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


class RetrievalQuality(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class RetrievalEvaluatorGrading:
    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self._threshold = confidence_threshold

    def grade_retrieval(self, query: str, nodes: Sequence[NodeWithScore]) -> RetrievalQuality:
        if not nodes:
            return RetrievalQuality.INCORRECT

        best_score = nodes[0].score if nodes[0].score is not None else 0.0
        if best_score >= self._threshold:
            return RetrievalQuality.CORRECT
        elif best_score >= 0.45:
            return RetrievalQuality.AMBIGUOUS
        else:
            return RetrievalQuality.INCORRECT


class CorrectiveRAGPipeline:
    def __init__(
        self,
        collection_name: str = "crag_knowledge_store",
        embedding_model: str = "gemini-embedding-001",
        llm_model: str = "gemini-2.5-flash",
    ) -> None:
        self._grader = RetrievalEvaluatorGrading()
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)
        self._llm = GoogleGenAI(model=llm_model)

    def index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

    def run(self, query: str) -> tuple[str, RetrievalQuality]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)

        quality = self._grader.grade_retrieval(query, nodes)

        if quality == RetrievalQuality.CORRECT:
            context = "\n".join(n.node.get_content() for n in nodes)
            ans = self._llm.complete(f"Context: {context}\nQuery: {query}\nAnswer accurately:").text
            return ans, quality
        elif quality == RetrievalQuality.AMBIGUOUS:
            context = "\n".join(n.node.get_content() for n in nodes)
            ans = self._llm.complete(
                f"Context: {context}\nQuery: {query}\nProvide answer with caution regarding ambiguity:"
            ).text
            return ans, quality
        else:
            return "Unable to answer: Retrieval confidence is too low and external web search fallback is triggered.", quality
