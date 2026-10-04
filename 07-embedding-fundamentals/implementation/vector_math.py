from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class ISimilarityMetric(ABC):
    @abstractmethod
    def calculate(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        pass


class CosineSimilarity(ISimilarityMetric):
    def calculate(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


class EuclideanDistance(ISimilarityMetric):
    def calculate(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


class DotProduct(ISimilarityMetric):
    def calculate(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))


class EmbeddingVectorStoreManager:
    def __init__(
        self,
        collection_name: str = "embedding_fundamentals",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def index_documents(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

    def get_embedding(self, text: str) -> list[float]:
        return self._embed_model.get_text_embedding(text)
