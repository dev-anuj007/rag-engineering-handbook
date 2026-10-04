# 14 Reranking

## Objective

Implement a high-precision two-stage retrieval architecture combining fast bi-encoder retrieval (ChromaDB) with cross-encoder / fine-grained rerankers.

## Why It Matters

Bi-encoders generate embeddings for query and passage separately, missing fine-grained token-level cross-attention interactions. Cross-encoders evaluate full query-document attention simultaneously, yielding a 15–30% boost in NDCG@10.

## Core Concepts

- **Bi-Encoder vs Cross-Encoder**: Bi-encoders trade attention depth for offline pre-computation; cross-encoders provide maximum semantic precision on candidate subsets.
- **Two-Stage Retrieval Funnel**: Coarse fetch (top-50) followed by fine neural rerank (top-3).
- **Latency vs Precision Trade-off**: Allocating ~20ms rerank compute budget per query.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics: Bi-Encoder vs Cross-Encoder

Information retrieval systems employ a **Two-Stage Cascade Architecture**:
1. **Stage 1: Bi-Encoder (Dense ANN Search)**: Evaluates millions of documents in milliseconds by computing independent vector representations, but suffers from the **information bottleneck** (query and document cannot interact at token level).
2. **Stage 2: Cross-Encoder (Reranker)**: Feeds query and candidate passage simultaneously into full self-attention layers, computing all-to-all cross-token attention $O(L^2)$ for deep semantic relevance scoring.

```
[ STAGE 1: BI-ENCODER (High Recall, Fast) ]
Query ──► [Encoder] ──► e(q) ┐
                             ├─► Dot Product (No Cross-Token Attention) ──► Top-50 Chunks
Doc   ──► [Encoder] ──► e(d) ┘

[ STAGE 2: CROSS-ENCODER (High Precision, Compute Heavy) ]
[CLS] + Query + [SEP] + Candidate Doc ──► [Full Transformer Cross-Attention] ──► Relevance Score (0..1)
                                         (Tokens attend to each other)
                                                                             │
                                                                             ▼
                                                                     Top-5 Final Context
```

---

## 2. Mathematical Formalization of Cross-Attention

Given query tokens $\mathbf{Q} = [q_1, \dots, q_n]$ and passage tokens $\mathbf{D} = [d_1, \dots, d_m]$ concatenated as input $\mathbf{X} = [\text{[CLS]}, q_1, \dots, q_n, \text{[SEP]}, d_1, \dots, d_m]$:

### Multi-Head Self-Attention Over Joint Sequence:

$$\mathbf{A}_{i, j} = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d_k}}\right)$$

Where query token $q_i$ directly attends to document token $d_j$, capturing complex relationships like negation (*"not covered under warranty"*), conditionality (*"only if signed before 2020"*), and coreferences.

The classification head extracts the final logit:

$$\text{Score}_{\text{rerank}}(q, d) = \sigma\left(\mathbf{W}_{\text{cls}} \mathbf{h}_{\text{[CLS]}} + b\right) \in [0, 1]$$

---

## 3. Reranking Cascade Performance Dynamics

| Stage | Candidates Evaluated | Latency per Candidate | Total Latency | Accuracy / Precision@5 |
|---|---|---|---|---|
| **Stage 1 (ChromaDB ANN)** | $10,000,000$ | $0.000001\text{ ms}$ | $\sim 10\text{ ms}$ | Moderate ($0.65$) |
| **Stage 2 (Cross-Encoder / Cohere / BGE)** | $50$ | $0.8\text{ ms}$ | $\sim 40\text{ ms}$ | Very High ($0.93$) |
| **Combined 2-Stage Cascade** | $10,000,000 \rightarrow 50 \rightarrow 5$ | — | $\mathbf{\sim 50\text{ ms}}$ | $\mathbf{0.93}$ |

---

## 4. Failure Modes & Mitigations

1. **Reranker Latency Explosion from Oversized Candidate Pool**:
   - *Failure*: Passing 500 candidate chunks to a Cross-Encoder pushes p95 query latency past $1200\text{ms}$.
   - *Mitigation*: Cap Stage 1 candidate pool to $N \in [25, 50]$ chunks. Stage 1 recall typically plateaus beyond Top-50.
2. **Context Length Truncation in Long Documents**:
   - *Failure*: Cross-Encoder has a 512-token limit. A candidate chunk of 700 tokens gets its concluding paragraphs truncated before scoring.
   - *Mitigation*: Restrict chunk sizes to $\le 400$ tokens during ingestion or score sliding windows across long chunks.

---

## 5. SOLID Principles in Reranking

- **Single Responsibility (SRP)**: `CrossEncoderReranker` scores query-document pairs; `ScoreFilter` sorts and truncates to Top-$K$.
- **Open/Closed (OCP)**: New reranker models (Cohere Rerank, BGE-Reranker, ColBERT) implement `RerankerProtocol`.
- **Liskov Substitution (LSP)**: All rerankers take `(query: str, docs: list[Document])` and return `list[ScoredDocument]`.
- **Dependency Inversion (DIP)**: Query orchestrator depends on `RerankerProtocol`.

---

## Implementation

Implemented in [reranker.py](implementation/reranker.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Drastically lifts top-1 precision, suppressing false positives.
- **Weaknesses**: Adds a second compute hop with GPU/CPU inference overhead.

## Architectural Decision & Trade-off Questions

1. *Why can't we run cross-encoder scoring across all 10M documents in a database directly?*
2. *How does ColBERT's late-interaction mechanism bridge the latency gap between bi-encoders and full cross-encoders?*
