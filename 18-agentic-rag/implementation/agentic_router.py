from __future__ import annotations

from typing import Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


class AgenticRAGManager:
    def __init__(
        self,
        embedding_model: str = "gemini-embedding-001",
        llm_model: str = "gemini-2.5-flash",
    ) -> None:
        self._client = chromadb.EphemeralClient()
        self._embed_model = GoogleGenAIEmbedding(model_name=embedding_model)
        self._llm = GoogleGenAI(model=llm_model)

    def _create_engine(self, collection_name: str, docs: Sequence[Document]):
        col = self._client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=col)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_ctx,
            embed_model=self._embed_model,
        )
        return index.as_query_engine(llm=self._llm)

    def build_router(
        self,
        finance_docs: Sequence[Document],
        tech_docs: Sequence[Document],
    ) -> RouterQueryEngine:
        finance_engine = self._create_engine("finance_coll", finance_docs)
        tech_engine = self._create_engine("tech_coll", tech_docs)

        tools = [
            QueryEngineTool(
                query_engine=finance_engine,
                metadata=ToolMetadata(
                    name="finance_tool",
                    description="Useful for questions regarding revenue, budgets, finance, and financial reports.",
                ),
            ),
            QueryEngineTool(
                query_engine=tech_engine,
                metadata=ToolMetadata(
                    name="tech_tool",
                    description="Useful for technical architecture, infrastructure, API limits, and engineering specs.",
                ),
            ),
        ]

        return RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=self._llm),
            query_engine_tools=tools,
        )
