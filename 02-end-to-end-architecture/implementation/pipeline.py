from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class PipelineConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 3
    collection_name: str = "end_to_end_rag"
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"


class IIngestionEngine(ABC):
    @abstractmethod
    def ingest(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        pass


class IIndexingEngine(ABC):
    @abstractmethod
    def index(self, nodes: Sequence[BaseNode]) -> VectorStoreIndex:
        pass


class IRetrievalEngine(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        pass


class IGenerationEngine(ABC):
    @abstractmethod
    def generate(self, query: str, context_nodes: Sequence[NodeWithScore]) -> str:
        pass


class SentenceSplitterIngestionEngine(IIngestionEngine):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest(self, documents: Sequence[Document]) -> Sequence[BaseNode]:
        return self._splitter.get_nodes_from_documents(documents)


class ChromaIndexingEngine(IIndexingEngine):
    def __init__(self, collection_name: str, embedding_model: str) -> None:
        self._collection_name = collection_name
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)
        self._client = chromadb.EphemeralClient()

    def index(self, nodes: Sequence[BaseNode]) -> VectorStoreIndex:
        chroma_collection = self._client.get_or_create_collection(self._collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )


class VectorRetrievalEngine(IRetrievalEngine):
    def __init__(self, index: VectorStoreIndex, similarity_top_k: int) -> None:
        self._retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    def retrieve(self, query: str) -> Sequence[NodeWithScore]:
        return self._retriever.retrieve(query)


class GroundedGenerationEngine(IGenerationEngine):
    def __init__(self, model_name: str) -> None:
        self._llm = GoogleGenAI(model=model_name)

    def generate(self, query: str, context_nodes: Sequence[NodeWithScore]) -> str:
        context_str = "\n\n".join(node.node.get_content() for node in context_nodes)
        prompt = (
            f"Context information:\n{context_str}\n\n"
            f"Given the context above, answer the query accurately:\n{query}\n"
            "If the context is insufficient, state 'Insufficient context'."
        )
        response = self._llm.complete(prompt)
        return response.text


class EndToEndRAGPipeline:
    def __init__(
        self,
        ingestion_engine: IIngestionEngine,
        indexing_engine: IIndexingEngine,
        generation_engine: IGenerationEngine,
        config: PipelineConfig,
    ) -> None:
        self._ingestion = ingestion_engine
        self._indexing = indexing_engine
        self._generation = generation_engine
        self._config = config
        self._retrieval: IRetrievalEngine | None = None

    def initialize(self, documents: Sequence[Document]) -> None:
        nodes = self._ingestion.ingest(documents)
        index = self._indexing.index(nodes)
        self._retrieval = VectorRetrievalEngine(index, self._config.similarity_top_k)

    def query(self, question: str) -> tuple[str, Sequence[NodeWithScore]]:
        if not self._retrieval:
            raise RuntimeError("Pipeline must be initialized with documents before querying.")
        nodes = self._retrieval.retrieve(question)
        answer = self._generation.generate(question, nodes)
        return answer, nodes


def create_pipeline(config: PipelineConfig = PipelineConfig()) -> EndToEndRAGPipeline:
    return EndToEndRAGPipeline(
        ingestion_engine=SentenceSplitterIngestionEngine(config.chunk_size, config.chunk_overlap),
        indexing_engine=ChromaIndexingEngine(config.collection_name, config.embedding_model),
        generation_engine=GroundedGenerationEngine(config.llm_model),
        config=config,
    )
