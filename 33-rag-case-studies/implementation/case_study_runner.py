from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class DomainCaseStudyConfig:
    domain_name: str
    similarity_top_k: int = 2
    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"


class DomainCaseStudyPipeline:
    def __init__(self, config: DomainCaseStudyConfig) -> None:
        self._config = config
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(f"case_study_{config.domain_name}")
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=config.embedding_model)
        self._llm = GoogleGenAI(model=config.llm_model)

    def load_case_corpus(self, docs: Sequence[Document]) -> None:
        storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
        VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_ctx,
            embed_model=self._embed_model,
        )

    def execute_domain_query(self, query: str) -> str:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=self._config.similarity_top_k)
        nodes = retriever.retrieve(query)
        context = "\n".join(n.node.get_content() for n in nodes)
        prompt = (
            f"Domain: {self._config.domain_name}\n"
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Provide domain-specific analysis:"
        )
        return self._llm.complete(prompt).text
