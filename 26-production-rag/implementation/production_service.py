from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    recovery_timeout_sec: float = 30.0


class CircuitBreakerOpenException(Exception):
    pass


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig = CircuitBreakerConfig()) -> None:
        self._config = config
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._is_open = False

    def record_success(self) -> None:
        self._failure_count = 0
        self._is_open = False

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._config.failure_threshold:
            self._is_open = True

    def check_state(self) -> None:
        if self._is_open:
            if time.time() - self._last_failure_time > self._config.recovery_timeout_sec:
                # Half-open state
                self._is_open = False
                self._failure_count = 0
            else:
                raise CircuitBreakerOpenException("Circuit breaker is OPEN. Upstream LLM is unavailable.")


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95) -> None:
        self._cache: dict[str, str] = {}

    def get(self, query: str) -> str | None:
        return self._cache.get(query.strip().lower())

    def put(self, query: str, answer: str) -> None:
        self._cache[query.strip().lower()] = answer


class ProductionRAGService:
    def __init__(
        self,
        collection_name: str = "prod_rag_collection",
        embedding_model: str = "gemini-embedding-001",
        llm_model: str = "gemini-2.5-flash",
    ) -> None:
        self._circuit_breaker = CircuitBreaker()
        self._cache = SemanticCache()

        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)
        self._llm = GoogleGenAI(model=llm_model)

    def index(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

    def query(self, question: str) -> tuple[str, bool]:
        # 1. Check Semantic Cache
        cached_ans = self._cache.get(question)
        if cached_ans:
            return cached_ans, True

        # 2. Check Circuit Breaker
        self._circuit_breaker.check_state()

        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                embed_model=self._embed_model,
            )
            retriever = index.as_retriever(similarity_top_k=2)
            nodes = retriever.retrieve(question)

            context = "\n".join(n.node.get_content() for n in nodes)
            ans = self._llm.complete(f"Context: {context}\nQuestion: {question}\nAnswer:").text

            self._cache.put(question, ans)
            self._circuit_breaker.record_success()
            return ans, False
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise e
