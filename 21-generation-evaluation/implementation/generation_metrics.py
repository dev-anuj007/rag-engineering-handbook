from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from llama_index.llms.google_genai import GoogleGenAI


@dataclass(frozen=True)
class GenerationMetricScore:
    faithfulness: float
    answer_relevance: float
    hallucination_detected: bool


class GenerationEvaluationCalculator:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._llm = GoogleGenAI(model=model_name)

    def evaluate_response(
        self,
        query: str,
        response: str,
        context: str,
    ) -> GenerationMetricScore:
        # Prompt for binary claim extraction & verification
        eval_prompt = f"""
        Evaluate the generated answer against the source context and question.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        {response}

        Check:
        1. Are all factual statements in the ANSWER directly supported by the CONTEXT? (Output YES or NO)
        2. Does the ANSWER directly address the QUESTION? (Output YES or NO)

        Format output as:
        FAITHFUL: [YES/NO]
        RELEVANT: [YES/NO]
        """
        result = self._llm.complete(eval_prompt).text

        is_faithful = "FAITHFUL: YES" in result.upper()
        is_relevant = "RELEVANT: YES" in result.upper()

        return GenerationMetricScore(
            faithfulness=1.0 if is_faithful else 0.0,
            answer_relevance=1.0 if is_relevant else 0.0,
            hallucination_detected=not is_faithful,
        )
