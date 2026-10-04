from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from llama_index.core.response_synthesizers import (
    CompactAndRefine,
    ResponseMode,
    TreeSummarize,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI


class SynthesisMode(Enum):
    COMPACT = "compact"
    TREE_SUMMARIZE = "tree_summarize"
    REFINE = "refine"


class ISynthesizer(ABC):
    @abstractmethod
    def synthesize(self, query: str, nodes: Sequence[NodeWithScore]) -> str:
        pass


class LlamaIndexResponseSynthesizer(ISynthesizer):
    def __init__(
        self,
        mode: SynthesisMode = SynthesisMode.COMPACT,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self._llm = GoogleGenAI(model=model_name)
        mode_mapping = {
            SynthesisMode.COMPACT: ResponseMode.COMPACT,
            SynthesisMode.TREE_SUMMARIZE: ResponseMode.TREE_SUMMARIZE,
            SynthesisMode.REFINE: ResponseMode.REFINE,
        }
        self._synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode=mode_mapping[mode],
        )

    def synthesize(self, query: str, nodes: Sequence[NodeWithScore]) -> str:
        if not nodes:
            return "Insufficient context available to answer the query."
        response = self._synthesizer.synthesize(
            query=query,
            nodes=nodes,
        )
        return str(response)
