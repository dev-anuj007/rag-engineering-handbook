from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


class ModalityType(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_DESCRIPTION = "image_description"


@dataclass(frozen=True)
class ParsedElement:
    element_id: str
    modality: ModalityType
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


class IElementParser(ABC):
    @abstractmethod
    def parse(self, raw_content: str, metadata: dict[str, str]) -> Sequence[ParsedElement]:
        pass


class StructuredDocumentParser(IElementParser):
    def parse(self, raw_content: str, metadata: dict[str, str]) -> Sequence[ParsedElement]:
        elements: list[ParsedElement] = []
        lines = raw_content.strip().split("\n")
        
        current_text_block: list[str] = []
        idx = 0

        for line in lines:
            line_str = line.strip()
            if line_str.startswith("|") and line_str.endswith("|"):
                # Flush text block before processing table
                if current_text_block:
                    elements.append(
                        ParsedElement(
                            element_id=f"{metadata.get('doc_id', 'doc')}-text-{idx}",
                            modality=ModalityType.TEXT,
                            content="\n".join(current_text_block),
                            metadata=metadata,
                        )
                    )
                    idx += 1
                    current_text_block = []
                elements.append(
                    ParsedElement(
                        element_id=f"{metadata.get('doc_id', 'doc')}-table-{idx}",
                        modality=ModalityType.TABLE,
                        content=line_str,
                        metadata={**metadata, "is_table": "true"},
                    )
                )
                idx += 1
            elif line_str.startswith("![") and "]" in line_str:
                if current_text_block:
                    elements.append(
                        ParsedElement(
                            element_id=f"{metadata.get('doc_id', 'doc')}-text-{idx}",
                            modality=ModalityType.TEXT,
                            content="\n".join(current_text_block),
                            metadata=metadata,
                        )
                    )
                    idx += 1
                    current_text_block = []
                elements.append(
                    ParsedElement(
                        element_id=f"{metadata.get('doc_id', 'doc')}-img-{idx}",
                        modality=ModalityType.IMAGE_DESCRIPTION,
                        content=line_str,
                        metadata={**metadata, "is_image_ref": "true"},
                    )
                )
                idx += 1
            else:
                if line_str:
                    current_text_block.append(line_str)

        if current_text_block:
            elements.append(
                ParsedElement(
                    element_id=f"{metadata.get('doc_id', 'doc')}-text-{idx}",
                    modality=ModalityType.TEXT,
                    content="\n".join(current_text_block),
                    metadata=metadata,
                )
            )

        return elements


class MultimodalIndexer:
    def __init__(
        self,
        collection_name: str = "multimodal_parsed_docs",
        embedding_model: str = "gemini-embedding-001",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)

    def index_elements(self, elements: Sequence[ParsedElement]) -> VectorStoreIndex:
        nodes = [
            TextNode(
                text=elem.content,
                id_=elem.element_id,
                metadata={
                    **elem.metadata,
                    "modality": elem.modality.value,
                },
            )
            for elem in elements
        ]
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )
