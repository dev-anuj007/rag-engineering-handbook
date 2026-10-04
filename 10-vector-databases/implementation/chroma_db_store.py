from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class HNSWIndexConfig:
    m: int = 16
    ef_construction: int = 100
    ef_search: int = 50
    space: str = "cosine"


class ChromaDatabaseManager:
    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "hnsw_chroma_collection",
        hnsw_config: HNSWIndexConfig = HNSWIndexConfig(),
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        if persist_directory:
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.EphemeralClient()

        metadata = {
            "hnsw:space": hnsw_config.space,
            "hnsw:construction_ef": hnsw_config.ef_construction,
            "hnsw:M": hnsw_config.m,
            "hnsw:search_ef": hnsw_config.ef_search,
        }
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def index_documents(self, documents: Sequence[Document]) -> VectorStoreIndex:
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context,
            embed_model=self._embed_model,
        )

    def query(
        self,
        query_str: str,
        similarity_top_k: int = 3,
        metadata_filters: dict[str, Any] | None = None,
    ) -> Sequence[NodeWithScore]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(
            similarity_top_k=similarity_top_k,
        )
        return retriever.retrieve(query_str)

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._collection.count()
