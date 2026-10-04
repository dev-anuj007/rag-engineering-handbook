# 12 Retrieval Fundamentals

## Objective

Master dense vector retrieval, similarity score normalization, top-k candidate selection, and relevance score thresholding in ChromaDB collections.

## Why It Matters

Retrieving fixed top-$K$ chunks unconditionally injects irrelevant distractor nodes when queries have zero relevant documents in the corpus, causing LLM hallucinations. Score thresholding acts as a critical relevance gate.

## Core Concepts

- **Cosine Distance vs Similarity**: Converting distance metrics into intuitive similarity bounds $[0, 1]$.
- **Top-K Tuning**: Finding the sweet spot between retrieval recall and LLM context bloat.
- **Score Threshold Cutoffs**: Discarding low-confidence candidate chunks.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Dense Vector Retrieval

Dense vector retrieval is the query-time mapping of a natural language question into an embedding space $\mathbf{e}(q) \in \mathbb{R}^D$ and executing similarity scoring against indexed passage vectors $\mathbf{e}(d_i) \in \mathbb{R}^D$ to yield the top-$K$ candidates that maximize semantic proximity.

```
Query String: "How do I configure OAuth2 in production?"
                     │
                     ▼
             [ Bi-Encoder ] ──► Dense Query Vector e(q) (1x768)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────────┐
             │       ChromaDB Vector Store (Indexed HNSW Graph)        │
             │   Distance Computation: 1 - (e(q) · e(d_i))            │
             └──────────────────────────┬──────────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────────┐
             │ Top-K Scored Candidates:                                │
             │   1. doc_982 (Score: 0.94) - "OAuth2 Setup Guide"       │
             │   2. doc_114 (Score: 0.89) - "JWT Token Validation"     │
             │   3. doc_045 (Score: 0.81) - "SSO Security Policy"      │
             └─────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization & Thresholding Strategies

Given query $q$ and collection $\mathcal{C}$:

### 1. Similarity Scoring ($S_i \in [-1, 1]$):

$$S_i = \cos(\mathbf{e}(q), \mathbf{e}(d_i)) = \frac{\mathbf{e}(q) \cdot \mathbf{e}(d_i)}{\|\mathbf{e}(q)\| \|\mathbf{e}(d_i)\|}$$

### 2. Thresholding Mechanisms:
- **Fixed Top-$K$**: Always returns exactly $K$ items.
  - *Risk*: Returns low-quality irrelevant noise when no truly relevant documents exist in the corpus.
- **Score Threshold Cutoff ($\tau_{\text{sim}}$)**:

$$\mathcal{R}_{\text{filtered}} = \{d_i \in \text{Top-}K \mid S_i \ge \tau_{\text{sim}}\}$$

- **Dynamic Range / Radius Search**: Returns all documents within distance radius $\epsilon$:

$$\mathcal{R}_{\epsilon} = \{d_i \in \mathcal{C} \mid \|\mathbf{e}(q) - \mathbf{e}(d_i)\|_2 \le \epsilon\}$$

---

## 3. Retrieval Strategy Comparison

| Retrieval Strategy | Latency Budget | Recall@10 | Precision@10 | Computational Complexity |
|---|---|---|---|---|
| **Fixed Top-$K$ ($K=5$)** | $< 10\text{ ms}$ | Moderate ($0.75$) | Moderate ($0.70$) | $O(\log N)$ |
| **Fixed Top-$K$ with Score Cutoff** | $< 12\text{ ms}$ | High ($0.82$) | High ($0.88$) | $O(\log N)$ + $O(K)$ filter |
| **Auto-Merging Hierarchical Retrieval** | $\sim 25\text{ ms}$ | Very High ($0.91$) | Very High ($0.92$) | $O(\log N)$ + Tree aggregation |

---

## 4. Failure Modes & Mitigations

1. **Semantic Drift on Short Queries**:
   - *Failure*: A 1-word query (*"refunds"*) maps to a broad region in vector space, retrieving generic customer support documents instead of specific transaction policies.
   - *Mitigation*: Implement query expansion / HyDE (Hypothetical Document Embeddings) to enrich short queries into full synthetic passages before embedding.
2. **Irrelevant Context Flooding (Hallucination Driver)**:
   - *Failure*: An off-topic user query retrieves $K=5$ unrelated chunks with low scores ($0.35$), which are passed to the LLM, prompting it to generate hallucinated answers.
   - *Mitigation*: Apply a strict minimum similarity threshold ($\tau \ge 0.70$). If no documents exceed $\tau$, trigger immediate fallback to standard prompt or generic assistant behavior.

---

## 5. SOLID Principles in Retrieval Fundamentals

- **Single Responsibility (SRP)**: `DenseRetriever` queries ChromaDB; `SimilarityFilter` enforces threshold bounds.
- **Open/Closed (OCP)**: Custom similarity scorers or distance transforms implement `ScorerProtocol`.
- **Liskov Substitution (LSP)**: All retrievers implement `RetrieverProtocol` yielding `list[ScoredDocument]`.
- **Dependency Inversion (DIP)**: Synthesizer depends on `RetrieverProtocol`.

---

## Implementation

Implemented in [dense_retriever.py](implementation/dense_retriever.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Ultra-fast, intuitive, native vector database execution.
- **Weaknesses**: Vulnerable to out-of-vocabulary acronyms and exact keyword mismatches.

## Architectural Decision & Trade-off Questions

1. *How do you dynamically calibrate the score threshold across different domains without artificially dropping valid edge-case queries?*
2. *What is the relationship between retrieval Hit Rate@K and downstream generation hallucination rates?*
