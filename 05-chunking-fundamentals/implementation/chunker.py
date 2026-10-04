from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 256
    chunk_overlap: int = 32


class IChunker(ABC):
    @abstractmethod
    def chunk(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        pass


class FixedTokenChunker(IChunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        return self._splitter.get_nodes_from_documents(documents)


class SentenceAwareChunker(IChunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        return self._splitter.get_nodes_from_documents(documents)


class ChunkIndexer:
    def __init__(
        self,
        collection_name: str = "chunking_fundamentals",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def index(self, nodes: Sequence[BaseNode]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )
