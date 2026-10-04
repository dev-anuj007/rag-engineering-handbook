from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class RAGFailureMode(Enum):
    MISSING_CONTENT = "missing_content"
    MISSED_TOP_K = "missed_top_k"
    NOT_IN_CONTEXT = "not_in_context"
    NOT_EXTRACTED = "not_extracted"
    WRONG_FORMAT = "wrong_format"
    INCORRECT_SPECIFICITY = "incorrect_specificity"
    INCOMPLETE = "incomplete"
    NO_FAILURE = "no_failure"


@dataclass(frozen=True)
class FailureDiagnostic:
    detected_mode: RAGFailureMode
    mitigation_strategy: str


class RAGFailureDiagnosticsEngine:
    @staticmethod
    def diagnose(
        is_in_corpus: bool,
        is_retrieved_in_k: bool,
        is_passed_to_llm: bool,
        is_correctly_generated: bool,
    ) -> FailureDiagnostic:
        if not is_in_corpus:
            return FailureDiagnostic(
                detected_mode=RAGFailureMode.MISSING_CONTENT,
                mitigation_strategy="Ingest additional source documents or enable web search fallback.",
            )
        if not is_retrieved_in_k:
            return FailureDiagnostic(
                detected_mode=RAGFailureMode.MISSED_TOP_K,
                mitigation_strategy="Tune chunking strategy, switch to hybrid BM25+dense search, or increase Top-K.",
            )
        if not is_passed_to_llm:
            return FailureDiagnostic(
                detected_mode=RAGFailureMode.NOT_IN_CONTEXT,
                mitigation_strategy="Adjust reranker cutoff thresholds or increase prompt token budget.",
            )
        if not is_correctly_generated:
            return FailureDiagnostic(
                detected_mode=RAGFailureMode.NOT_EXTRACTED,
                mitigation_strategy="Apply lost-in-the-middle prompt reordering or use few-shot extractor prompts.",
            )
        return FailureDiagnostic(
            detected_mode=RAGFailureMode.NO_FAILURE,
            mitigation_strategy="Pipeline executing within nominal performance parameters.",
        )
