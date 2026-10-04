from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from llama_index.llms.google_genai import GoogleGenAI


@dataclass(frozen=True)
class JudgeEvaluationResult:
    score: int  # 1 to 5
    reasoning: str
    pass_threshold: bool


class StructuredLLMJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._llm = GoogleGenAI(model=model_name)

    def judge_answer(
        self,
        query: str,
        context: str,
        answer: str,
        criteria: str = "Correctness and Groundedness",
    ) -> JudgeEvaluationResult:
        rubric_prompt = f"""
        You are an impartial, expert evaluation judge. Rate the following answer on a scale from 1 to 5 based on the criteria: {criteria}.

        EVALUATION CRITERIA:
        1: Completely inaccurate or hallucinated.
        2: Major inaccuracies or ungrounded claims.
        3: Partially correct but omits crucial context.
        4: Highly accurate with minor formatting issues.
        5: Perfectly accurate, complete, and strictly grounded in the context.

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
        {answer}

        OUTPUT FORMAT:
        SCORE: [1-5]
        REASONING: [Brief 1-sentence rationale]
        """
        response = self._llm.complete(rubric_prompt).text

        score = 5
        for line in response.split("\n"):
            if "SCORE:" in line.upper():
                for char in line:
                    if char.isdigit() and 1 <= int(char) <= 5:
                        score = int(char)
                        break

        return JudgeEvaluationResult(
            score=score,
            reasoning=response.strip(),
            pass_threshold=(score >= 4),
        )
