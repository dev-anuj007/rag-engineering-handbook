from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class KnowledgeGraphTriplet:
    subject: str
    predicate: str
    object_: str


class InMemoryKnowledgeGraph:
    def __init__(self) -> None:
        self._adj: dict[str, list[tuple[str, str]]] = {}

    def add_triplet(self, triplet: KnowledgeGraphTriplet) -> None:
        sub = triplet.subject.strip().lower()
        obj = triplet.object_.strip().lower()
        pred = triplet.predicate.strip().lower()

        if sub not in self._adj:
            self._adj[sub] = []
        self._adj[sub].append((pred, obj))

    def get_neighbors(self, entity: str) -> list[tuple[str, str]]:
        return self._adj.get(entity.strip().lower(), [])


class HybridGraphVectorRetriever:
    def __init__(
        self,
        collection_name: str = "graph_rag_vectors",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._kg = InMemoryKnowledgeGraph()
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def populate_graph(self, triplets: Sequence[KnowledgeGraphTriplet]) -> None:
        for t in triplets:
            self._kg.add_triplet(t)
            # Index triplet text description in ChromaDB
            summary_doc = Document(
                text=f"{t.subject} {t.predicate} {t.object_}.",
                metadata={"entity": t.subject.lower()},
            )
            storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
            VectorStoreIndex.from_documents(
                [summary_doc],
                storage_context=storage_ctx,
                embed_model=self._embed_model,
            )

    def retrieve_with_graph_expansion(self, query_entity: str) -> list[str]:
        # 1. Direct Graph 1-hop traversal
        neighbors = self._kg.get_neighbors(query_entity)
        graph_facts = [f"{query_entity} {pred} {obj}" for pred, obj in neighbors]

        # 2. Vector search over related triplet summaries
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(similarity_top_k=2)
        vector_results = retriever.retrieve(query_entity)
        vector_facts = [res.node.get_content() for res in vector_results]

        combined = list(set(graph_facts + vector_facts))
        return combined
