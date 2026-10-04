from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence
from llama_index.llms.google_genai import GoogleGenAI


@dataclass(frozen=True)
class GeneratedQAPair:
    qa_id: str
    query: str
    ground_truth_context: str
    ground_truth_answer: str
    complexity_level: str


class SyntheticQAGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._llm = GoogleGenAI(model=model_name)

    def generate_from_chunk(self, context_chunk: str) -> GeneratedQAPair:
        # Simple extraction logic for offline synthetic dataset building
        lines = context_chunk.strip().split(". ")
        first_line = lines[0] if lines else context_chunk
        generated_query = f"What does the system specify regarding: {first_line[:40]}...?"
        
        return GeneratedQAPair(
            qa_id=str(uuid.uuid4())[:8],
            query=generated_query,
            ground_truth_context=context_chunk,
            ground_truth_answer=first_line,
            complexity_level="factual_direct",
        )
