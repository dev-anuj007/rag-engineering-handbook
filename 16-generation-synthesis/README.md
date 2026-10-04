# 16 Generation & Synthesis

## Objective

Master LlamaIndex response synthesizers (Compact, Tree Summarize, Refine), streaming generation, structured citation grounding, and strict refusal mechanisms.

## Why It Matters

Naive string concatenation of 10 chunks into an LLM prompt leads to fragmented or rambling responses. Structured synthesizers generate cohesive, fully-cited answers with zero context hallucination.

## Core Concepts

- **Compact & Refine**: Packs multiple chunks per prompt to minimize LLM roundtrips; refines sequentially if context spans multiple windows.
- **Tree Summarize**: Recursively builds hierarchical summary trees over hundreds of chunks for high-level thematic queries.
- **Strict Grounding Refusal**: Programmatically refuses when retrieved nodes contain zero supporting evidence.

## Architecture

```
[Retrieved Nodes] ──> [Response Synthesizer Engine] ──> [Gemini LLM] ──> [Grounded Cited Answer]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [synthesizer.py](implementation/synthesizer.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Compact**: Fastest latency and lowest token cost for standard QA.
- **Tree Summarize**: Capable of summarizing entire document collections at the expense of multiple LLM calls.

## Architectural Decision & Trade-off Questions

1. *How do you prevent response synthesis latency degradation when dealing with 50+ retrieved nodes?*
2. *How do you enforce inline bracketed citations `[Doc 1, p. 4]` during generation without degrading LLM fluency?*
