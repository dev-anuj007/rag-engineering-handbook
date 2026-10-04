# 22 LLM-as-a-Judge

## Objective

Design rigorous, automated LLM evaluation pipelines using structured Likert rubrics, G-Eval framework techniques, pairwise battle comparisons, and position-bias mitigation.

## Why It Matters

Human evaluation does not scale to thousands of daily enterprise queries. High-capacity LLM judges (e.g. Gemini 2.5 Flash / Pro) correlate with human domain experts at $>85\%$ when provided with explicit step-by-step rubrics.

## Core Concepts

- **Structured Scoring Rubrics**: Explicitly defining criteria for scores 1 through 5 prevents score drift.
- **Position & Verbosity Bias**: LLMs naturally favor longer responses and options presented first; positional swapping cancels this bias.
- **Chain-of-Thought Evaluation**: Forcing the judge to output reasoning *before* emitting the numeric score improves consistency.

## Architecture

```
[Context + Question + Answer] ──> [Structured Rubric Prompt] ──> [LLM Judge Engine] ──> [1-5 Score + Rationale]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [judge_evaluator.py](implementation/judge_evaluator.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Automated, highly scalable, provides human-readable explanations for every score.
- **Weaknesses**: Susceptible to self-enhancement bias when evaluating models from the same family.

## Architectural Decision & Trade-off Questions

1. *How do you mathematically measure the inter-annotator agreement (Cohen's Kappa / Krippendorff's Alpha) between an LLM judge and human experts?*
2. *How do you prevent 'judge hacking' where generated responses exploit the known stylistic preferences of the evaluator LLM?*
