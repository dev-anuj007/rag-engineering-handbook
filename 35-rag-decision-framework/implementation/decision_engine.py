from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecommendedPattern(Enum):
    NAIVE_VECTOR_RAG = "naive_vector_rag"
    HYBRID_RRF_RAG = "hybrid_rrf_rag"
    GRAPH_VECTOR_HYBRID = "graph_vector_hybrid"
    AGENTIC_ROUTER_RAG = "agentic_router_rag"
    FINE_TUNING_ONLY = "fine_tuning_only"


@dataclass(frozen=True)
class WorkloadProfile:
    has_exact_part_numbers: bool
    requires_multi_hop_relational_reasoning: bool
    has_multiple_isolated_departments: bool
    requires_style_adaptation_only: bool


class ArchitecturalDecisionEngine:
    @staticmethod
    def recommend_architecture(profile: WorkloadProfile) -> RecommendedPattern:
        if profile.requires_style_adaptation_only:
            return RecommendedPattern.FINE_TUNING_ONLY
        if profile.has_multiple_isolated_departments:
            return RecommendedPattern.AGENTIC_ROUTER_RAG
        if profile.requires_multi_hop_relational_reasoning:
            return RecommendedPattern.GRAPH_VECTOR_HYBRID
        if profile.has_exact_part_numbers:
            return RecommendedPattern.HYBRID_RRF_RAG
        return RecommendedPattern.NAIVE_VECTOR_RAG
