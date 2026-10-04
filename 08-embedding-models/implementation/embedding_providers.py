from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class EmbeddingTaskType(Enum):
    RETRIEVAL_DOCUMENT = "retrieval_document"
    RETRIEVAL_QUERY = "retrieval_query"
    SEMANTIC_SIMILARITY = "semantic_similarity"


class IEmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str, task_type: EmbeddingTaskType) -> list[float]:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass


class GeminiAsymmetricEmbeddingProvider(IEmbeddingProvider):
    def __init__(self, model_name: str = "gemini-embedding-001") -> None:
        self._embed_model = GoogleGenAIEmbedding(model_name=model_name)

    def embed_text(self, text: str, task_type: EmbeddingTaskType) -> list[float]:
        # Prefix handling for asymmetric embedding models
        if task_type == EmbeddingTaskType.RETRIEVAL_DOCUMENT:
            formatted = f"search_document: {text}"
        elif task_type == EmbeddingTaskType.RETRIEVAL_QUERY:
            formatted = f"search_query: {text}"
        else:
            formatted = text
        return self._embed_model.get_text_embedding(formatted)

    def get_dimension(self) -> int:
        return 768


class MatryoshkaTruncator:
    @staticmethod
    def truncate_dimension(embedding: Sequence[float], target_dim: int) -> list[float]:
        truncated = list(embedding[:target_dim])
        # Re-normalize vector to unit length
        norm = sum(x * x for x in truncated) ** 0.5
        if norm > 0:
            return [x / norm for x in truncated]
        return truncated


class AsymmetricVectorStore:
    def __init__(
        self,
        collection_name: str = "asymmetric_embeddings",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )
