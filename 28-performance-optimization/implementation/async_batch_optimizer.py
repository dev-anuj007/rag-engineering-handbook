from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class BatchOptimizationConfig:
    batch_size: int = 64
    max_concurrency: int = 8


class AsyncBatchProcessor:
    def __init__(
        self,
        collection_name: str = "optimized_batch_chroma",
        embedding_model: str = "gemini-embedding-001",
        config: BatchOptimizationConfig = BatchOptimizationConfig(),
    ) -> None:
        self._config = config
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def batch_insert(self, documents: Sequence[Document]) -> int:
        total = len(documents)
        for i in range(0, total, self._config.batch_size):
            batch = documents[i : i + self._config.batch_size]
            storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
            VectorStoreIndex.from_documents(
                batch,
                storage_context=storage_ctx,
                embed_model=self._embed_model,
            )
        return total

    async def async_multi_retrieve(self, queries: Sequence[str]) -> list[list[str]]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=2)

        async def _query_single(q: str) -> list[str]:
            # Simulate async task execution
            results = retriever.retrieve(q)
            return [res.node.get_content() for res in results]

        tasks = [_query_single(q) for q in queries]
        return await asyncio.gather(*tasks)
