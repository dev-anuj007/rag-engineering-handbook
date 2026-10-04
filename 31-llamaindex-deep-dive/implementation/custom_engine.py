from __future__ import annotations

from typing import Any, Sequence
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


class CustomChromaRetriever(BaseRetriever):
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 2,
    ) -> None:
        self._index = index
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        retriever = self._index.as_retriever(similarity_top_k=self._similarity_top_k)
        return retriever.retrieve(query_bundle.query_str)


class CustomSOLIDQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: GoogleGenAI,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        super().__init__()

    def _query(self, query_bundle: QueryBundle) -> Response:
        nodes = self._retriever.retrieve(query_bundle)
        context_str = "\n".join(n.node.get_content() for n in nodes)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query_bundle.query_str}\nAnswer:"
        result = self._llm.complete(prompt).text
        return Response(response=result, source_nodes=nodes)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._query(query_bundle)


def create_custom_rag_engine(
    documents: Sequence[Document],
    collection_name: str = "custom_llamaindex_collection",
) -> CustomSOLIDQueryEngine:
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=col)
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
    llm = GoogleGenAI(model="gemini-2.5-flash")

    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        embed_model=embed_model,
    )

    retriever = CustomChromaRetriever(index=index, similarity_top_k=2)
    return CustomSOLIDQueryEngine(retriever=retriever, llm=llm)
