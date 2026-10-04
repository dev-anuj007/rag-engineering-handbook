# 24 Evaluation Datasets

## Objective

Build, curate, and evolve robust synthetic and human-reviewed golden evaluation datasets with explicit SQLite persistence and train/validation/test partitions.

## Why It Matters

Manual annotation of 1,000 QA pairs across complex enterprise manuals is time-prohibitive. Synthetic generation combined with Evol-Instruct complexity mutation produces diverse multi-hop evaluation benches at near-zero cost.

## Core Concepts

- **Corpus-Grounded Synthetic Generation**: LLM-driven generation of questions conditioned on specific corpus nodes.
- **Evol-Instruct Mutation**: Evolving simple queries into complex multi-hop, counterfactual, or comparative questions.
- **Dataset Splitting & Leakage Prevention**: Enforcing strict document-level isolation between test and training sets.

## Architecture

```
[Corpus Chunks] ──> [Synthetic Generator] ──> [Evol-Instruct Mutator] ──> [SQLite Golden DB]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [dataset_generator.py](implementation/dataset_generator.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Enables massive test coverage across all document sections automatically.
- **Weaknesses**: Synthetic questions can lack the conversational messiness and typos of real end-users.

## Architectural Decision & Trade-off Questions

1. *How do you prevent data contamination when using synthetic questions to evaluate a RAG pipeline?*
2. *What is the role of 'negative examples' (queries with no relevant answer in the corpus) in golden evaluation sets?*
