# 06 Advanced Chunking

## Objective

Implement semantic boundary splitting, markdown header extraction, and multi-tier hierarchical parent-child chunking for high-precision retrieval with broad context synthesis.

## Why It Matters

Fixed-size chunking forces a trade-off: small chunks retrieve precisely but lack surrounding context for generation, while large chunks dilute semantic search vectors. Hierarchical chunking indexes small leaf chunks but resolves to larger parent context during synthesis.

## Core Concepts

- **Hierarchical Parent-Child Nodes**: Index leaf nodes in ChromaDB; pass parent nodes to the LLM.
- **Markdown & Structural Chunking**: Parse headers (#, ##, ###) into logically scoped sections.
- **Semantic Chunking**: Split when adjacent sentence embedding cosine similarity drops below a dynamic percentile threshold.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Advanced Chunking Topologies

Advanced chunking decoupling techniques overcome the fundamental tradeoff between **retrieval specificity** (small text) and **generation synthesis context** (large text) through structural hierarchies and embedding distance boundaries.

```
[ HIERARCHICAL / PARENT-CHILD CHUNKING ]

         ┌────────────────────────────────────────────────────────┐
         │ Parent Chunk (Large Context: 1500 Tokens for LLM)      │
         └──────────────────────────┬─────────────────────────────┘
                                    │ Partitioned into
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────────┐┌───────────────────────┐┌───────────────────────┐
│ Child Chunk 1 (250t)  ││ Child Chunk 2 (250t)  ││ Child Chunk 3 (250t)  │
│ (Indexed in ChromaDB) ││ (Indexed in ChromaDB) ││ (Indexed in ChromaDB) │
└───────────┬───────────┘└───────────────────────┘└───────────────────────┘
            │ Query matches Child 1
            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Vector Search retrieves Child 1 ID ──► Resolves Parent ID in SQLite     │
│ Injects FULL PARENT CHUNK (1500t) into LLM Prompt Context               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Mechanics: Semantic & Statistical Chunking

### Semantic Distance Breakpoint Detection
Rather than splitting at fixed token thresholds, semantic chunking monitors the cosine distance between adjacent sentence embeddings:

$$d_i = 1 - \cos(\mathbf{e}(s_i), \mathbf{e}(s_{i+1})) = 1 - \frac{\mathbf{e}(s_i) \cdot \mathbf{e}(s_{i+1})}{\|\mathbf{e}(s_i)\| \|\mathbf{e}(s_{i+1})\|}$$

A chunk breakpoint is inserted at index $i$ when the semantic distance exceeds a dynamic percentile threshold $\tau$:

$$\text{Breakpoint}(i) = \mathbb{I}\left(d_i > \mu_d + \alpha \cdot \sigma_d\right)$$

Where:
- $\mu_d, \sigma_d$ are the mean and standard deviation of semantic distances across the document.
- $\alpha$ is a sensitivity multiplier (typically $\alpha \in [0.5, 1.5]$).

```
Semantic
Distance (d) ▲
             │                  Breakpoint 1               Breakpoint 2
             │                      ▲                          ▲
             │                      │                          │
Threshold τ  │- - - - - - - - - - - ┼ - - - - - - - - - - - - - ┼ - - - - - - - -
             │       /\             │          /\              │
             │  /\  /  \     /\     │    /\   /  \      /\     │     /\
             │ /  \/    \   /  \    │   /  \ /    \    /  \    │    /  \
             └──────────────────────┴──────────────────────────┴──────────────► Sentence (i)
               [   Chunk A        ]   [       Chunk B        ]   [   Chunk C   ]
```

---

## 3. Advanced Chunking Strategy Comparison

| Strategy | Vector Indexed Unit | Context Passed to LLM | Ingestion Latency | Retrieval Recall |
|---|---|---|---|---|
| **Parent-Child (Hierarchical)** | Small Child (200t) | Full Parent (1500t) | Low | Very High |
| **Sentence-Window Retrieval** | Single Sentence (50t) | Sentence + $k$ Surrounding Sentences | Low | High (Pinpoint matches) |
| **Semantic Boundary Chunking** | Coherent Paragraph | Coherent Paragraph | High (Embedding per sentence) | High |
| **Document Summary Indexing** | LLM-generated Chunk Summary | Raw Chunk Text | Very High (LLM per chunk) | Very High (Abstractive) |

---

## 4. Failure Modes & Mitigations

1. **Parent-Child Context Explosion**:
   - *Failure*: 5 child chunks match the query, but all 5 belong to different parents, causing the assembled prompt to exceed the LLM context window.
   - *Mitigation*: Deduplicate parent IDs so that multiple matching children from the same parent only inject the parent once; apply context compression to remaining parents.
2. **High Embedding Latency during Semantic Chunking**:
   - *Failure*: Computing embeddings for every single sentence individually slows down ingestion throughput by $10\times$.
   - *Mitigation*: Batch sentence embeddings in chunks of 128 into the model; use small fast embedding models (e.g. `all-MiniLM-L6-v2`) for boundary detection.

---

## 5. SOLID Principles in Advanced Chunking

- **Single Responsibility (SRP)**: `SemanticSplitter` detects distance anomalies; `HierarchyNodeLinker` maps parent-child graph relationships.
- **Open/Closed (OCP)**: New hierarchy schemas (e.g., Book -> Chapter -> Section -> Paragraph) extend `HierarchyChunkerProtocol`.
- **Liskov Substitution (LSP)**: `HierarchicalChunker` produces standard `ChunkNode` entities carrying parent metadata.
- **Dependency Inversion (DIP)**: Retrievers depend on `MetadataResolverProtocol` to fetch parent content from SQLite.

---

## Implementation

Implemented in [advanced_chunker.py](implementation/advanced_chunker.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Highest signal-to-noise ratio in search; zero context fragmentation.
- **Weaknesses**: Requires more complex storage topology (vector store + doc store lookup).

## Architectural Decision & Trade-off Questions

1. *How does Parent-Child retrieval resolve the 'Lost in the Middle' phenomenon during multi-chunk synthesis?*
2. *How do you compute dynamic threshold cutoffs in semantic embedding-based chunking?*
