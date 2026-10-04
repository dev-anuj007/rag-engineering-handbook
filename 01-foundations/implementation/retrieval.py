from __future__ import annotations

from typing import Protocol, Sequence
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class BaseRetriever(Protocol):
    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        ...


class ChromaVectorRetriever:
    def __init__(
        self,
        nodes: Sequence[BaseNode],
        collection_name: str = "foundations_rag",
        embedding_model: str = "gemini-embedding-001",
        similarity_top_k: int = 3,
        persist_dir: str | None = None,
    ) -> None:
        if persist_dir:
            chroma_client = chromadb.PersistentClient(path=persist_dir)
        else:
            chroma_client = chromadb.EphemeralClient()

        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

        self._index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        self._retriever = self._index.as_retriever(
            similarity_top_k=similarity_top_k,
        )

    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        return self._retriever.retrieve(query)


def build_retriever(
    nodes: Sequence[BaseNode],
    collection_name: str = "foundations_rag",
    similarity_top_k: int = 3,
) -> BaseRetriever:
    return ChromaVectorRetriever(
        nodes=nodes,
        collection_name=collection_name,
        similarity_top_k=similarity_top_k,
    )
