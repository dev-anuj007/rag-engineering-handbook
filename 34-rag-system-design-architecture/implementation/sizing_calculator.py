from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemCapacityEstimation:
    num_documents: int
    avg_tokens_per_doc: int
    vector_dimension: int
    target_qps: int

    def compute_total_chunks(self, chunk_size: int = 256, chunk_overlap: int = 32) -> int:
        stride = chunk_size - chunk_overlap
        chunks_per_doc = max(1, (self.avg_tokens_per_doc - chunk_size) // stride + 1)
        return self.num_documents * chunks_per_doc

    def compute_vector_memory_gb(self, precision_bytes: int = 4) -> float:
        total_vectors = self.compute_total_chunks()
        # Vector raw bytes + ~1.5x HNSW graph index overhead
        raw_bytes = total_vectors * self.vector_dimension * precision_bytes
        total_index_bytes = raw_bytes * 1.5
        return total_index_bytes / (1024 ** 3)

    def compute_daily_llm_cost(self, prompt_cost_per_m: float = 0.15, answer_cost_per_m: float = 0.60) -> float:
        daily_queries = self.target_qps * 86400
        avg_prompt_tokens = 1500  # Query + context
        avg_output_tokens = 300
        
        daily_input_cost = (daily_queries * avg_prompt_tokens / 1_000_000) * prompt_cost_per_m
        daily_output_cost = (daily_queries * avg_output_tokens / 1_000_000) * answer_cost_per_m
        return daily_input_cost + daily_output_cost
