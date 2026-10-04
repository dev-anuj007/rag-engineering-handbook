from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from llama_index.core.schema import NodeWithScore, TextNode


@dataclass(frozen=True)
class ContextEngineeringConfig:
    max_token_budget: int = 1000
    enable_reordering: bool = True  # Lost in the middle mitigation


class IContextPacker(ABC):
    @abstractmethod
    def pack(self, query: str, nodes: Sequence[NodeWithScore]) -> str:
        pass


class LostInTheMiddleReorderer:
    """
    Places highest relevance items at the start and end of prompt context,
    pushing lower relevance nodes to the middle where LLM attention tends to sag.
    """
    @staticmethod
    def reorder(nodes: Sequence[NodeWithScore]) -> list[NodeWithScore]:
        if len(nodes) <= 2:
            return list(nodes)
        
        sorted_nodes = sorted(nodes, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        reordered: list[NodeWithScore] = [None] * len(sorted_nodes)  # type: ignore

        left = 0
        right = len(sorted_nodes) - 1
        for i, item in enumerate(sorted_nodes):
            if i % 2 == 0:
                reordered[left] = item
                left += 1
            else:
                reordered[right] = item
                right -= 1
        return reordered


class ProductionContextPacker(IContextPacker):
    def __init__(self, config: ContextEngineeringConfig = ContextEngineeringConfig()) -> None:
        self._config = config

    def pack(self, query: str, nodes: Sequence[NodeWithScore]) -> str:
        active_nodes = list(nodes)
        if self._config.enable_reordering:
            active_nodes = LostInTheMiddleReorderer.reorder(active_nodes)

        formatted_sections: list[str] = []
        cumulative_tokens = 0

        for idx, item in enumerate(active_nodes):
            text = item.node.get_content().strip()
            # Rough token estimate (~4 chars per token)
            token_est = len(text) // 4
            if cumulative_tokens + token_est > self._config.max_token_budget:
                break

            doc_id = item.node.metadata.get("doc_id", f"doc_{idx+1}")
            formatted = f"--- [Source: {doc_id}] ---\n{text}"
            formatted_sections.append(formatted)
            cumulative_tokens += token_est

        context_str = "\n\n".join(formatted_sections)
        return (
            f"You are a precise technical assistant. Answer the question using ONLY the provided context snippets.\n\n"
            f"CONTEXT SNIPPETS:\n{context_str}\n\n"
            f"QUESTION: {query}\n"
            f"ANSWER:"
        )
