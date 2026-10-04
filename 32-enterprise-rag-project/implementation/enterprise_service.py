from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class EnterpriseQueryRequest:
    tenant_id: str
    user_id: str
    query: str
    top_k: int = 3


@dataclass(frozen=True)
class EnterpriseQueryResponse:
    answer: str
    sources: list[str]
    tenant_id: str


class EnterpriseRAGService:
    def __init__(
        self,
        collection_name: str = "enterprise_rag_cluster",
        embedding_model: str = "gemini-embedding-001",
        llm_model: str = "gemini-2.5-flash",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)
        self._llm = GoogleGenAI(model=llm_model)

    def ingest_tenant_document(self, tenant_id: str, doc_id: str, content: str) -> None:
        doc = Document(
            text=content,
            metadata={"tenant_id": tenant_id, "doc_id": doc_id},
        )
        storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
        VectorStoreIndex.from_documents(
            [doc],
            storage_context=storage_ctx,
            embed_model=self._embed_model,
        )

    def query_tenant(self, request: EnterpriseQueryRequest) -> EnterpriseQueryResponse:
        # Pre-filtering strictly enforces multi-tenant boundary
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="tenant_id", value=request.tenant_id)]
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(
            similarity_top_k=request.top_k,
            filters=filters,
        )
        nodes = retriever.retrieve(request.query)

        if not nodes:
            return EnterpriseQueryResponse(
                answer="No relevant documentation found for your organization.",
                sources=[],
                tenant_id=request.tenant_id,
            )

        context_str = "\n\n".join(n.node.get_content() for n in nodes)
        prompt = (
            f"Tenant Context:\n{context_str}\n\n"
            f"Question: {request.query}\n"
            f"Provide a concise, grounded enterprise answer:"
        )
        ans = self._llm.complete(prompt).text
        sources = [n.node.metadata.get("doc_id", "unknown") for n in nodes]

        return EnterpriseQueryResponse(
            answer=ans,
            sources=sources,
            tenant_id=request.tenant_id,
        )
