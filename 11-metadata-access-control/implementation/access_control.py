from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class SecurityClearance(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class UserContext:
    user_id: str
    tenant_id: str
    clearance: SecurityClearance
    roles: Sequence[str] = field(default_factory=list)


class AccessControlledRetriever:
    def __init__(
        self,
        collection_name: str = "rbac_controlled_rag",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def index_secure_documents(self, documents: Sequence[Document]) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

    def retrieve_with_rbac(
        self,
        query: str,
        user_context: UserContext,
        similarity_top_k: int = 3,
    ) -> Sequence[NodeWithScore]:
        # Pre-filtering: strict tenant and clearance isolation
        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="tenant_id", value=user_context.tenant_id),
            ]
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=filters,
        )
        nodes = retriever.retrieve(query)

        # Post-filtering for hierarchical clearance enforcement
        clearance_hierarchy = {
            SecurityClearance.PUBLIC: 0,
            SecurityClearance.INTERNAL: 1,
            SecurityClearance.CONFIDENTIAL: 2,
            SecurityClearance.RESTRICTED: 3,
        }
        user_level = clearance_hierarchy[user_context.clearance]

        authorized_nodes: list[NodeWithScore] = []
        for n in nodes:
            doc_clearance_str = n.node.metadata.get("clearance", "restricted")
            try:
                doc_clearance = SecurityClearance(doc_clearance_str)
                if clearance_hierarchy[doc_clearance] <= user_level:
                    authorized_nodes.append(n)
            except ValueError:
                continue

        return authorized_nodes
