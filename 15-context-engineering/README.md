# 15 Context Engineering

## Objective

Optimize prompt packing, mitigate the "Lost in the Middle" attention degradation curve, enforce strict token budgeting, and inject explicit citation metadata.

## Why It Matters

LLMs exhibit high positional bias: information placed in the middle of long multi-chunk prompts is frequently ignored or hallucinated. Reordering context strategically restores attention focus on top-ranked evidence.

## Core Concepts

- **Positional Attention Bias**: LLMs recall evidence best when placed at the exact start or end of context.
- **Lost-in-the-Middle Reordering**: Alternating placement of high-ranking nodes to prompt extremes.
- **Hard Token Budgeting**: Preventing prompt overflow and truncating noisy tail nodes.

## Architecture

```
[Retrieved Nodes (Ranks 1..N)] ──> [U-Curve Reorderer] ──> [Token Budget Truncator] ──> [LLM Prompt]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [context_compressor.py](implementation/context_compressor.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Noticeable reduction in context omission errors without additional model fine-tuning.
- **Weaknesses**: Requires deterministic character/token budgeting calculations.

## Architectural Decision & Trade-off Questions

1. *How does prompt packing order affect reasoning performance in long-context models (e.g. 1M token contexts)?*
2. *How do you dynamically allocate prompt budget between system instructions, few-shot examples, retrieved context, and conversation history?*
