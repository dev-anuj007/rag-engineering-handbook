from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class HybridRetrievalConfig:
    vector_top_k: int = 5
    bm25_top_k: int = 5
    final_top_k: int = 3
    rrf_k: int = 60
    collection_name: str = "hybrid_advanced_retrieval"
    embedding_model: str = "gemini-embedding-001"


class SimpleBM25Retriever:
    def __init__(self, nodes: Sequence[BaseNode]) -> None:
        self._nodes = nodes
        self._corpus = [node.get_content().lower().split() for node in nodes]

    def retrieve(self, query: str, top_k: int) -> Sequence[NodeWithScore]:
        query_terms = query.lower().split()
        scores: list[tuple[float, BaseNode]] = []
        for doc_terms, node in zip(self._corpus, self._nodes):
            score = sum(doc_terms.count(term) for term in query_terms)
            if score > 0:
                scores.append((float(score), node))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [NodeWithScore(node=n, score=s) for s, n in scores[:top_k]]


class HybridRRFRetriever:
    def __init__(
        self,
        nodes: Sequence[BaseNode],
        config: HybridRetrievalConfig = HybridRetrievalConfig(),
    ) -> None:
        self._config = config
        self._bm25_retriever = SimpleBM25Retriever(nodes)

        # ChromaDB setup
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(config.collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=config.embedding_model)

        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )
        self._dense_retriever = self._vector_index.as_retriever(
            similarity_top_k=config.vector_top_k
        )

    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        dense_results = self._dense_retriever.retrieve(query)
        bm25_results = self._bm25_retriever.retrieve(query, top_k=self._config.bm25_top_k)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = defaultdict(float)
        node_map: dict[str, BaseNode] = {}

        for rank, item in enumerate(dense_results):
            node_id = item.node.node_id
            node_map[node_id] = item.node
            rrf_scores[node_id] += 1.0 / (self._config.rrf_k + rank + 1)

        for rank, item in enumerate(bm25_results):
            node_id = item.node.node_id
            node_map[node_id] = item.node
            rrf_scores[node_id] += 1.0 / (self._config.rrf_k + rank + 1)

        sorted_nodes = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            NodeWithScore(node=node_map[node_id], score=score)
            for node_id, score in sorted_nodes[: self._config.final_top_k]
        ]
