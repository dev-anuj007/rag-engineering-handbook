from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class RankingMetricsResult:
    hit_rate_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float


class IRankingMetricCalculator:
    @staticmethod
    def calculate_metrics(
        ground_truth_ids: Sequence[str],
        retrieved_ids: Sequence[str],
        k: int,
    ) -> RankingMetricsResult:
        top_k_retrieved = retrieved_ids[:k]
        if not ground_truth_ids or not top_k_retrieved:
            return RankingMetricsResult(0.0, 0.0, 0.0, 0.0)

        # Hit Rate @ K
        hits = [1 if doc_id in ground_truth_ids else 0 for doc_id in top_k_retrieved]
        hit_rate = 1.0 if sum(hits) > 0 else 0.0

        # Precision @ K
        precision = sum(hits) / k

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                mrr = 1.0 / (rank + 1)
                break

        # NDCG @ K
        dcg = sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(hits))
        ideal_hits = [1] * min(len(ground_truth_ids), k)
        idcg = sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal_hits))
        ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

        return RankingMetricsResult(
            hit_rate_at_k=hit_rate,
            precision_at_k=precision,
            mrr=mrr,
            ndcg_at_k=ndcg,
        )


class ChromaRetrievalBenchmarker:
    def __init__(
        self,
        collection_name: str = "retrieval_benchmark_chroma",
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

    def retrieve_doc_ids(self, query: str, top_k: int) -> list[str]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        return [res.node.metadata.get("doc_id", res.node.node_id) for res in results]
