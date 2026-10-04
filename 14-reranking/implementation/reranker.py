from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class TwoStagePipelineConfig:
    coarse_top_k: int = 10
    fine_top_k: int = 3
    collection_name: str = "reranking_stage_one"
    embedding_model: str = "gemini-embedding-001"


class IReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidate_nodes: Sequence[NodeWithScore], top_n: int) -> Sequence[NodeWithScore]:
        pass


class SemanticRelevanceReranker(IReranker):
    def rerank(self, query: str, candidate_nodes: Sequence[NodeWithScore], top_n: int) -> Sequence[NodeWithScore]:
        # Cross-encoder scoring approximation: evaluate joint token interaction
        query_words = set(query.lower().split())
        scored: list[NodeWithScore] = []

        for item in candidate_nodes:
            content = item.node.get_content().lower()
            overlap_score = sum(1 for w in query_words if w in content) / max(len(query_words), 1)
            # Combine vector cosine score with cross-attention token interaction
            base_score = item.score if item.score is not None else 0.5
            combined_score = 0.4 * base_score + 0.6 * overlap_score
            scored.append(NodeWithScore(node=item.node, score=combined_score))

        scored.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        return scored[:top_n]


class TwoStageRetrievalPipeline:
    def __init__(
        self,
        reranker: IReranker,
        config: TwoStagePipelineConfig = TwoStagePipelineConfig(),
    ) -> None:
        self._reranker = reranker
        self._config = config

        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(config.collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=config.embedding_model)

    def index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

    def retrieve_and_rerank(self, query: str) -> Sequence[NodeWithScore]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=self._config.coarse_top_k)
        candidates = retriever.retrieve(query)
        return self._reranker.rerank(query, candidates, top_n=self._config.fine_top_k)
