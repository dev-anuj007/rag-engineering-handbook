from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class DenseRetrievalConfig:
    similarity_top_k: int = 5
    score_threshold: float = 0.65
    collection_name: str = "dense_retrieval_fundamentals"
    embedding_model: str = "gemini-embedding-001"


class ThresholdedDenseRetriever:
    def __init__(self, config: DenseRetrievalConfig = DenseRetrievalConfig()) -> None:
        self._config = config
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=config.embedding_model)

    def index_corpus(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=self._config.similarity_top_k)
        raw_results = retriever.retrieve(query)

        # Apply score cutoff filter
        filtered_results = [
            node
            for node in raw_results
            if node.score is not None and node.score >= self._config.score_threshold
        ]
        return filtered_results
