# 04 Document Parsing & Multimodal Data

## Objective

Extract structured content (prose, tabular rows, and visual diagram representations) from complex documents into strongly-typed LlamaIndex nodes indexed into ChromaDB.

## Why It Matters

Naive text extraction destroys table structures and discards diagrams, creating catastrophic retrieval failures for numerical queries or visual schema questions.

## Core Concepts

- **Modality Segregation**: Discriminating prose, tabular structures, and diagram metadata.
- **Node Typing**: Tagging nodes with modality metadata to enable modality-aware retrieval filters.
- **ChromaDB Unified Vector Indexing**: Storing multimodal node embeddings with indexed metadata tags.

## Architecture

```
[Raw Document] ──> [Structured Parser] ──> [Modality Nodes] ──> [ChromaDB Unified Store]
                                                ├── Text Nodes
                                                ├── Table Nodes
                                                └── Image Nodes
```

See [architecture.md](architecture/architecture.md) for full design.

## Implementation

Implemented in [multimodal_parser.py](implementation/multimodal_parser.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Preserves tabular context and visual semantic cues.
- **Weaknesses**: Requires more complex preprocessing pipelines.

## Architectural Decision & Trade-off Questions

1. *How do you represent complex multi-page financial tables in a RAG system without losing column-header associations across chunk boundaries?*
2. *When should you embed image summary text vs using a joint multimodal embedding space (e.g. CLIP / Gemini Multimodal Embeddings)?*
