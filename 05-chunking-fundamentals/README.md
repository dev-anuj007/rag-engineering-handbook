# 05 Chunking Fundamentals

## Objective

Master foundational text chunking techniques, token vs character splitting geometries, overlap strategies, and their downstream impact on ChromaDB vector retrieval.

## Why It Matters

Arbitrary character slicing cuts sentences in half, severing semantic predicates and ruining embedding fidelity. Proper sentence-aware chunking preserves coherent propositions.

## Core Concepts

- **Token vs Character Splitting**: Token-based chunking aligns directly with LLM context windows, while character splitting is model-agnostic.
- **Overlap Math**: Retaining 10–20% token overlap prevents loss of context across split boundaries.
- **Sentence-Aware Boundaries**: Respecting punctuation stops prevents fragmented clauses.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Chunking

Chunking is the process of partitioning a continuous document stream $\mathcal{T} = [t_1, t_2, \dots, t_M]$ into discrete passages $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$ such that each chunk maximizes internal semantic coherence while adhering to embedding model context windows and LLM prompt budgets.

```
Raw Text Stream: [ ----------------- Document Text (3,000 Tokens) ----------------- ]
                                          │
                                          ▼
                      Chunk Size: S = 500, Overlap: O = 100
  ┌─────────────────────────┐
  │ Chunk 1 (Tokens 1..500) │
  └───────────┬─────────────┘
              │ Overlap Window (Tokens 401..500)
              ▼
        ┌─────────────────────────┐
        │ Chunk 2 (Tokens 401..900)│
        └───────────┬─────────────┘
                    │ Overlap Window (Tokens 801..900)
                    ▼
              ┌───────────────────────────┐
              │ Chunk 3 (Tokens 801..1400)│
              └───────────────────────────┘
```

---

## 2. Mathematical Formalization & Trade-offs

### The Chunk Size Trade-Off Formula

Let $S$ be chunk size in tokens, and $O$ be overlap size in tokens ($O < S$).

$$\text{Total Chunks } K = \left\lceil \frac{M - O}{S - O} \right\rceil$$

$$\text{Overlap Ratio } \rho = \frac{O}{S}$$

### The Retrieval-Generation Tension:
- **Small Chunks ($S = 128 - 256$)**:
  - High embedding specificity: The vector $\mathbf{e}(c_i)$ represents a narrow, focused concept.
  - High retrieval precision (ANN distance is tight).
  - *Risk*: Context fragmentation. Missing surrounding antecedent or resolution logic needed by the generator LLM.
- **Large Chunks ($S = 1024 - 2048$)**:
  - High synthesis context for generation.
  - Low embedding specificity: Dense bi-encoders average across diverse topics within the chunk, causing semantic dilution and lower retrieval recall.

```
       [ Chunk Size vs Performance Dynamics ]
  Precision / Specificity ▲
                          │  \ (Small Chunks: High Specificity)
                          │   \
                          │    \
                          │     \  / (Large Chunks: High Context)
                          │      \/
                          │      /\
                          │     /  \
  Context / Completeness  │    /    \
                          └─────────────────────────────► Chunk Size (S)
                             128    512    1024   2048
```

---

## 3. Chunking Strategies Comparison

| Strategy | Splitting Boundary | Strengths | Weaknesses |
|---|---|---|---|
| **Fixed-Size (Character)** | Hard character count ($N$ chars) | Fastest ($O(1)$ computation) | Slices words and sentences in half |
| **Fixed-Size (Token)** | Tokenizer-aware ($BPE$/WordPiece) | Fits exact embedding token limits | Still splits logical thoughts |
| **Sentence-Aware** | Sentence boundary regex / spaCy | Preserves complete grammatical thoughts | Chunks have variable token lengths |
| **Recursive Character** | Hierarchical separators (`\n\n`, `\n`, `.`, ` `) | Balances paragraph structure with size bounds | Requires tuning separator lists |

---

## 4. Failure Modes & Mitigations

1. **Boundary Severance of Coreference**:
   - *Failure*: Sentence 1 (*"Acme Corp acquired Beta AI in 2021."*) is in Chunk 1. Sentence 2 (*"The purchase price was $500M."*) is in Chunk 2. Querying *"How much did Acme pay for Beta?"* fails because Chunk 2 lacks the subject.
   - *Mitigation*: Maintain an overlap $\rho \ge 20\%$ (e.g. $S=512, O=100$) or apply parent-child / sentence-window chunking.
2. **Table Splitting**:
   - *Failure*: A 50-row financial table is split across 3 chunks without repeating the header row.
   - *Mitigation*: Isolate table blocks during parsing and chunk them as atomic units or prepend table headers to every sliced sub-table.

---

## 5. SOLID Principles in Chunking

- **Single Responsibility (SRP)**: `TokenCounter` computes model-specific BPE tokens; `BoundaryDetector` locates sentence boundaries; `Chunker` coordinates slice creation.
- **Open/Closed (OCP)**: New chunking algorithms (e.g., Markdown-aware, Code AST) implement `ChunkerProtocol`.
- **Liskov Substitution (LSP)**: All chunkers accept `str` and return `list[Chunk]` containing text and metadata offsets.
- **Dependency Inversion (DIP)**: Ingestion pipeline accepts `ChunkerProtocol` interface.

---

## Implementation

Implemented in [chunker.py](implementation/chunker.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Small Chunks (128-256 tokens)**: High retrieval precision, low context noise, but risk losing broader thematic context.
- **Large Chunks (512-1024 tokens)**: Rich context, but introduce irrelevant noise into the retrieval ranking.

## Architectural Decision & Trade-off Questions

1. *How does chunk size inversely correlate with retrieval precision and generation groundedness?*
2. *How do you choose between sentence-level chunking and fixed-token sliding windows for code vs narrative text?*
