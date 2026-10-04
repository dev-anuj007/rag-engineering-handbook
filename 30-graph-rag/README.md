# 30 Graph RAG

## Objective

Unify structured Knowledge Graph relational traversals (entities, predicates, multi-hop hops) with unstructured vector retrieval in ChromaDB to solve complex interconnected queries.

## Why It Matters

Standard vector search struggles with global relational questions ("How are all subsidiary holding companies connected to the parent entity?"). Graph RAG connects fragmented entities across disparate documents.

## Core Concepts

- **Entity-Relation Extraction**: Extracting `(Subject, Predicate, Object)` triplets from text during ingestion.
- **Graph Traversal Expansion**: Retrieving $N$-hop neighbor subgraphs when an entity is queried.
- **Hybrid Graph-Vector Synthesis**: Providing both relational topology and raw text excerpts to the LLM.

## Architecture

```
[Entity Query] ──┬──> [Graph Traversal (1..2 Hops)] ──┐
                 └──> [ChromaDB Vector Retrieval] ───┴──> [Context Synthesis]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [graph_retriever.py](implementation/graph_retriever.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Excels at multi-hop inductive reasoning and global holistic synthesis.
- **Weaknesses**: Entity extraction and graph indexing significantly increase ingestion latency.

## Architectural Decision & Trade-off Questions

1. *How do you resolve entity ambiguity (entity disambiguation / linking) across millions of ingested documents?*
2. *When is a dedicated graph database (Neo4j) required versus embedding entity summaries into ChromaDB with relational metadata links?*
