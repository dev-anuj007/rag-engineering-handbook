from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, MarkdownNodeParser
from llama_index.core.schema import BaseNode, IndexNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class HierarchicalConfig:
    chunk_sizes: Sequence[int] = (1024, 256, 64)


class IAdvancedChunker(ABC):
    @abstractmethod
    def parse_nodes(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        pass


class HierarchicalParentChildChunker(IAdvancedChunker):
    def __init__(self, config: HierarchicalConfig = HierarchicalConfig()) -> None:
        self._parser = HierarchicalNodeParser.from_defaults(chunk_sizes=list(config.chunk_sizes))

    def parse_nodes(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        return self._parser.get_nodes_from_documents(documents)


class MarkdownHeaderChunker(IAdvancedChunker):
    def __init__(self) -> None:
        self._parser = MarkdownNodeParser()

    def parse_nodes(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        return self._parser.get_nodes_from_documents(documents)


class AdvancedChunkIndexer:
    def __init__(
        self,
        collection_name: str = "advanced_chunking",
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
