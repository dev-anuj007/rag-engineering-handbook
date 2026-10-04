from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@dataclass(frozen=True)
class RawDocumentPayload:
    doc_id: str
    source_uri: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    total_processed: int
    inserted_count: int
    skipped_duplicates: int


class IDocumentSource(ABC):
    @abstractmethod
    def fetch_documents(self) -> Sequence[RawDocumentPayload]:
        pass


class IDeduplicationStore(ABC):
    @abstractmethod
    def is_duplicate(self, doc_hash: str) -> bool:
        pass

    @abstractmethod
    def mark_seen(self, doc_hash: str, doc_id: str) -> None:
        pass


class InMemoryDeduplicationStore(IDeduplicationStore):
    def __init__(self) -> None:
        self._seen_hashes: dict[str, str] = {}

    def is_duplicate(self, doc_hash: str) -> bool:
        return doc_hash in self._seen_hashes

    def mark_seen(self, doc_hash: str, doc_id: str) -> None:
        self._seen_hashes[doc_hash] = doc_id


class DocumentIngestionService:
    def __init__(
        self,
        dedup_store: IDeduplicationStore,
        collection_name: str = "ingestion_collection",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._dedup_store = dedup_store
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def process(self, raw_payloads: Sequence[RawDocumentPayload]) -> IngestionResult:
        docs_to_index: list[Document] = []
        skipped = 0

        for payload in raw_payloads:
            content_hash = self._compute_hash(payload.content)
            if self._dedup_store.is_duplicate(content_hash):
                skipped += 1
                continue

            metadata = {
                **payload.metadata,
                "doc_id": payload.doc_id,
                "source_uri": payload.source_uri,
                "content_hash": content_hash,
            }
            doc = Document(text=payload.content, metadata=metadata)
            docs_to_index.append(doc)
            self._dedup_store.mark_seen(content_hash, payload.doc_id)

        if docs_to_index:
            storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
            VectorStoreIndex.from_documents(
                docs_to_index,
                storage_context=storage_context,
                embed_model=self._embed_model,
            )

        return IngestionResult(
            total_processed=len(raw_payloads),
            inserted_count=len(docs_to_index),
            skipped_duplicates=skipped,
        )
