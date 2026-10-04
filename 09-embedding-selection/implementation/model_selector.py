from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class DomainProfile(Enum):
    GENERAL = "general"
    CODE = "code"
    FINANCIAL = "financial"
    LEGAL = "legal"


@dataclass(frozen=True)
class ModelSelectionCriteria:
    domain: DomainProfile
    max_latency_ms: float
    max_memory_dim: int
    context_window_tokens: int


@dataclass(frozen=True)
class EmbeddingModelCandidate:
    name: str
    dimension: int
    max_tokens: int
    avg_latency_ms: float
    mteb_score: float
    supported_domains: Sequence[DomainProfile]


class EmbeddingModelSelector:
    def __init__(self, candidates: Sequence[EmbeddingModelCandidate]) -> None:
        self._candidates = candidates

    def select_best_model(self, criteria: ModelSelectionCriteria) -> EmbeddingModelCandidate:
        eligible = [
            c
            for c in self._candidates
            if criteria.domain in c.supported_domains
            and c.dimension <= criteria.max_memory_dim
            and c.avg_latency_ms <= criteria.max_latency_ms
            and c.max_tokens >= criteria.context_window_tokens
        ]
        if not eligible:
            # Fallback to general model with highest MTEB score
            return max(self._candidates, key=lambda x: x.mteb_score)
        return max(eligible, key=lambda x: x.mteb_score)


class SelectedModelVectorIndexer:
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        collection_name: str = "selected_embedding_benchmark",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=model_name)

    def index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )
